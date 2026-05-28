import cv2
import numpy as np
import zmq
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.55.0.1", help="pi ethernet host")
parser.add_argument("--port", default="5555", help="pi ethernet port")
args = parser.parse_args()

PI_HOST = args.host
PI_PORT = int(args.port)


def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB) # zmq subscriber mode
    sock.setsockopt(zmq.CONFLATE, 1) # limit receive queue to only the latest element
    sock.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to everythign
    sock.connect(f"tcp://{PI_HOST}:{PI_PORT}")

    try:
        while True:
            jpg = sock.recv()

            # https://stackoverflow.com/questions/49511753/python-byte-image-to-numpy-array-using-opencv
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow("Pi Camera", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
