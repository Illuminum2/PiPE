from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import time

from position_pipeline import min_distance_calibration, max_distance_calibration, set_min_distance, set_max_distance
from position_pipeline import is_calibrated, people_positions


app = FastAPI()


position_websockets = []
position_task = None

async def read_positions():
    global position_task

    position_stream = None

    try:
        while len(position_websockets) > 0:
            if not is_calibrated():
                await asyncio.sleep(0.25)
                continue

            if position_stream is None:
                position_stream = people_positions()

            try:
                # https://stackoverflow.com/a/65716963
                positions = await asyncio.to_thread(next, position_stream)
                data = positions_to_data(positions)

                for websocket in position_websockets:
                    try:
                        await websocket.send_json(data)
                    except (RuntimeError, WebSocketDisconnect):
                        position_websockets.remove(websocket)

            except RuntimeError as error:
                position_stream.close()
                position_stream = None

                # await websocket.send_json({
                #     'message': str(error)
                # })
    finally:
        position_stream.close()
        position_task = None


@app.post("/coordinates/calibrate/min")
async def calibrate_min_distance():
    calibrated, message = await asyncio.to_thread(min_distance_calibration)

    return calibration_result(calibrated, message)

@app.post("/coordinates/calibrate/min/{min_distance}")
async def set_min(min_distance: float):
    calibrated, message = await asyncio.to_thread(set_min_distance, min_distance)

    return calibration_result(calibrated, message)

@app.post("/coordinates/calibrate/max")
async def calibrate_max_distance():
    calibrated, message = await asyncio.to_thread(max_distance_calibration)

    return calibration_result(calibrated, message)

@app.post("/coordinates/calibrate/max/{max_distance}")
async def set_max(max_distance: float):
    calibrated, message = await asyncio.to_thread(set_max_distance, max_distance)

    return calibration_result(calibrated, message)


@app.websocket("/coordinates/stream")
async def websocket_coordinates(websocket: WebSocket):
    global position_task

    await websocket.accept()
    position_websockets.append(websocket)

    if position_task is None:
        position_task = asyncio.create_task(read_positions())

    try:
        while websocket in position_websockets:
            await asyncio.sleep(0.25)

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in position_websockets:
            position_websockets.remove(websocket)


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