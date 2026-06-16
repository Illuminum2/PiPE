import cv2
import json
import numpy as np
import zmq

from dataclasses import dataclass


PI_HOST = "10.12.194.1"
PI_PORT = 5555
UNIT_CELL_SIZE = (1400, 1400)
FOCAL_LENGTH = 4.74


@dataclass
class Frame:
    img: np.ndarray
    unit_cell_size: tuple
    focal_length: float


def frames_from_pi(host=PI_HOST, port=PI_PORT, flip=True):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)  # zmq subscriber mode
    # http://api.zeromq.org/3-0:zmq-getsockopt
    sock.setsockopt(zmq.RCVHWM, 1)  # limit receive queue to only the latest element
    sock.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to everything
    sock.connect(f"tcp://{host}:{port}")

    try:
        while True:
            # https://gist.github.com/Dansyuqri/9eea1c4affa27b9d5ad138fd508e8026
            msg = sock.recv_multipart()

            # Deserialize JSON
            info = json.loads(msg[0].decode("utf-8"))
            jpg = msg[1]

            # https://stackoverflow.com/questions/49511753/python-byte-image-to-numpy-array-using-opencv
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

            if flip:
                frame = cv2.flip(frame, 1)

            if frame.any():
                yield Frame(
                    frame,
                    (info.get("unit_cell_size", UNIT_CELL_SIZE)),
                    float(info.get("focal_length", FOCAL_LENGTH))
                )

    finally:
        sock.close(0)
        ctx.term()