from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import time

from depth_pipeline import background_calibration, min_depth_calibration, max_depth_calibration
from depth_pipeline import is_calibrated, people_positions


app = FastAPI()
address = "10.10.0.85:8000"



@app.post("/api/coordinates/calibrate/background")
async def calibrate_coordinates():
    calibrated, message = await asyncio.to_thread(background_calibration)

    return calibration_result(calibrated, message)


@app.post("/api/coordinates/calibrate/min")
async def calibrate_min_depth():
    calibrated, message = await asyncio.to_thread(min_depth_calibration)

    return calibration_result(calibrated, message)


@app.post("/api/coordinates/calibrate/max")
async def calibrate_max_depth():
    calibrated, message = await asyncio.to_thread(max_depth_calibration)

    return calibration_result(calibrated, message)


@app.websocket("/api/coordinates/stream")
async def websocket_coordinates(websocket: WebSocket):
    await websocket.accept()
    positions = people_positions()

    try:
        while True:
            # https://stackoverflow.com/a/65716963
            positions = await asyncio.to_thread(next, positions)
            data = positions_to_data(positions)

            await websocket.send_json(data)

    except WebSocketDisconnect:
        pass
    except RuntimeError as error:
        await websocket.send_json({
            'message': str(error)
        })
    finally:
        positions.close()


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