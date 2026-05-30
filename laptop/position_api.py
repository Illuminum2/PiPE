from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import time

from depth_pipeline import calculate_base_calibration, people_positions


app = FastAPI()
address = "10.10.0.85:8000"


@app.post("/api/coordinates/calibrate")
async def calibrate_coordinates():
    calibrated, message = await asyncio.to_thread(calculate_base_calibration)

    return {
        'calibrated': calibrated,
        'message': message
    }


@app.websocket("/api/coordinates/stream")
async def websocket_coordinates(websocket: WebSocket):
    await websocket.accept()
    positions = people_positions()

    try:
        while True:
            # https://stackoverflow.com/a/65716963
            face_centers, face_depths = await asyncio.to_thread(next, positions)
            data = positions_to_data(face_centers, face_depths)

            await websocket.send_json(data)

    except WebSocketDisconnect:
        pass
    except RuntimeError as error:
        await websocket.send_json({
            'message': error
        })
    finally:
        positions.close()


def positions_to_data(face_centers, face_depths):
    data = []

    for (x, y), z in zip(face_centers, face_depths):
        x, y, z = fix_pos(x, y, z)

        data.append({
            'x': x,
            'y': y,
            'z': z,
            'timestamp': round(time.time(), 2)
        })

    return data

def fix_pos(x, y, z):
    x = (x / 640) * (640 / 1000)
    y = (y / 360) * (360 / 1000)

    x += (1 - (640 / 1000)) / 2
    y += (1 - (360 / 1000)) / 2

    z = z / 3

    return x, y, z