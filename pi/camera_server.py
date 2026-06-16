import cv2
import zmq
from picamera2 import Picamera2
from libcamera import controls
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="*", help="address to bind")
    parser.add_argument("--port", type=int, default=5555, help="pi port")
    parser.add_argument("--width", type=int, default=640, help="camera width")
    parser.add_argument("--height", type=int, default=360, help="camera height")
    parser.add_argument("--fps", type=int, default=30, help="camera fps")
    parser.add_argument("--quality", type=int, default=45, help="camera jpg quality")
    parser.add_argument("--focal-length", type=float, default=4.74, help="lens focal length in mm")
    args = parser.parse_args()

    cam = Picamera2()
    # https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-2-picamera2-manual.pdf
    cam.configure(
        cam.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={
                "FrameRate": args.fps,
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": 0.0,
            },
            buffer_count=1,
        )
    )
    # Set focus to infinity and disable autofocus
    cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})
    cam.start()

    # Serialized JSON
    camera_info = json.dumps({
        "unit_cell_size": cam.camera_properties["UnitCellSize"],
        "focal_length": args.focal_length,
    }).encode("utf-8")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB) # zmq publish mode
    # http://api.zeromq.org/3-0:zmq-getsockopt
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
                # https://gist.github.com/Dansyuqri/9eea1c4affa27b9d5ad138fd508e8026
                sock.send_multipart([camera_info, jpg.tobytes()])

    finally:
        sock.close(0)
        ctx.term()
        cam.stop()


if __name__ == "__main__":
    main()