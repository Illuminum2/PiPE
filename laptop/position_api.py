from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import time

from position_pipeline import min_distance_calibration, max_distance_calibration, set_min_distance, set_max_distance
from position_pipeline import is_calibrated, people_positions


app = FastAPI()
address = "10.10.0.85:8000"



@app.post("/api/coordinates/calibrate/background")
async def calibrate_coordinates():
    calibrated, message = await asyncio.to_thread(background_calibration)

    return calibration_result(calibrated, message)


@app.post("/api/coordinates/calibrate/min")
async def calibrate_min_distance():
    calibrated, message = await asyncio.to_thread(min_distance_calibration)

    return calibration_result(calibrated, message)


@app.post("/api/coordinates/calibrate/max")
async def calibrate_max_distance():
    calibrated, message = await asyncio.to_thread(max_distance_calibration)

    return calibration_result(calibrated, message)


@app.websocket("/api/coordinates/stream")
async def websocket_coordinates(websocket: WebSocket):
    await websocket.accept()
    position_stream = None

    try:
        while True:
            if not is_calibrated():
                await asyncio.sleep(0.25)
                continue

            position_stream = people_positions()

            try:
                while True:
                    # https://stackoverflow.com/a/65716963
                    positions = await asyncio.to_thread(next, position_stream)
                    data = positions_to_data(positions)

                    await websocket.send_json(data)
            except RuntimeError as error:
                position_stream.close()
                position_stream = None

                # await websocket.send_json({
                #     'message': str(error)
                # })

    except WebSocketDisconnect:
        pass
    finally:
        if position_stream is not None:
            position_stream.close()


def calibration_result(calibrated, message):
    return {
        'calibrated': calibrated,
        'message': message
    }


def positions_to_data(positions):
    data = []

    for x, y, z in positions:
        data.append({
            'x': x,
            'y': y,
            'z': z,
            'timestamp': round(time.time(), 2)
        })

    return data