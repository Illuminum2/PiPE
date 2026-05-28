import cv2
import zmq
from picamera2 import Picamera2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.55.0.1", help="pi host")
parser.add_argument("--port", default="5555", help="pi port")
parser.add_argument("--width", type=int, default=640, help="camera width")
parser.add_argument("--height", type=int, default=360, help="camera height")
parser.add_argument("--fps", type=int, default=30, help="camera fps")
parser.add_argument("--quality", type=int, default=45, help="camera jpg quality")
args = parser.parse_args()


def main():
    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": args.fps},
            buffer_count=1,
        )
    )
    cam.start()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB) # zmq publish mode
    sock.setsockopt(zmq.SNDHWM, 1) # limit the send queue to only the latest element
    sock.bind(f"tcp://{args.host}:{args.port}")

    print(f"Camera server running on {args.host}:{args.port}")

    try:
        while True:
            rgb = cam.capture_array()

            # https://stackoverflow.com/questions/59259786/how-to-encode-opencv-image-as-bytes-using-python
            # https://stackoverflow.com/questions/64342838/why-is-cv2-imencodejpg-seemingly-changing-the-color-of-an-image
            ok, jpg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, args.quality])

            if ok:
                sock.send(jpg.tobytes())

    finally:
        sock.close(0)
        ctx.term()
        cam.stop()


if __name__ == "__main__":
    main()
