from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import time

from depth_pipeline import people_positions


app = FastAPI()
address = "10.10.0.85:8000"


@app.websocket("/stream/coordinates")
async def websocket_coordinates(websocket: WebSocket):
    await websocket.accept()

    try:
        for face_centers, face_depths in people_positions():
            print(face_centers, face_depths)
            data = positions_to_data(face_centers, face_depths)

            await websocket.send_json(data)

    except WebSocketDisconnect:
        pass


def positions_to_data(face_centers, face_depths):
    data = []

    for (x, y), z in zip(face_centers, face_depths):
        data.append({
            'x': x,
            'y': y,
            'z': z,
            'timestamp': time.time()
        })

    return data