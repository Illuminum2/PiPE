import cv2
import numpy as np
import zmq


PI_HOST = "10.12.194.1"
PI_PORT = 5555


def frames_from_pi(host=PI_HOST, port=PI_PORT, flip=True):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)  # zmq subscriber mode
    sock.setsockopt(zmq.CONFLATE, 1)  # limit receive queue to only the latest element
    sock.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to everything
    sock.connect(f"tcp://{host}:{port}")

    try:
        while True:
            jpg = sock.recv()

            # https://stackoverflow.com/questions/49511753/python-byte-image-to-numpy-array-using-opencv
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

            if flip:
                frame = cv2.flip(frame, 1)

            if frame.any():
                yield frame

    finally:
        sock.close(0)
        ctx.term()