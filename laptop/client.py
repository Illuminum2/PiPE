import cv2
import numpy as np
import zmq


def frames_from_pi(host, port):
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

            if frame.any():
                yield frame

    finally:
        sock.close(0)
        ctx.term()
