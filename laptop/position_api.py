from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import asyncio
import time

from depth_pipeline import people_positions


app = FastAPI()
address = "10.10.0.85:8000"


@app.websocket("/api/stream/coordinates")
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
    finally:
        positions.close()


def positions_to_data(face_centers, face_depths):
    data = []

    for (x, y), z in zip(face_centers, face_depths):
        data.append({
            'x': x,
            'y': y,
            'z': z,
            'timestamp': round(time.time(), 2)
        })

    return data