import cv2
import numpy
from PIL import Image
from transformers import pipeline
import argparse

from client import frames_from_pi


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="10.55.0.1", help="pi ethernet host")
parser.add_argument("--port", default="5555", type=int, help="pi ethernet port")
parser.add_argument("--flip", action="store_false", help="flip image")
args = parser.parse_args()


face_cascade = cv2.CascadeClassifier("./models/frontalface.xml")

pipe = pipeline(
    task='depth-estimation',
    model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
    #model='Intel/dpt-hybrid-midas',
    device='mps'
)


# Detect faces with haar cascades
def detect_faces(img):
    face_img = img.copy()
    return face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=7)

def calculate_face_centers(face_rects):
    face_centers = []

    for (x, y, w, h) in face_rects:
        face_centers.append((x + w/2, y + h/2))

    return face_centers

# Draw face rectangles on image
def draw_face_rects(img, face_rects):
    rect_img = img.copy()

    for (x, y, w, h) in face_rects:
        cv2.rectangle(rect_img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 0), thickness=10)

    return rect_img

def draw_face_centers(img, face_centers):
    center_img = img.copy()

    for (x, y) in face_centers:
        cv2.circle(center_img, center=(int(x), int(y)), radius=10, color=(0, 0, 0), thickness=cv2.FILLED)

    return center_img


# Estimate monocular depth from frame
def depth_estimation(img):
    # Convert image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Process image with pipeline
    result = pipe(pil_img)

    # Normalized display depth
    depth_img = numpy.array(result['depth'])

    # Depth for calculations
    raw_depth = result['predicted_depth'].cpu().numpy()

    return depth_img, raw_depth


def calculate_patch_depths(depth_img, patch_size, margin):
    img_h, img_w = depth_img.shape[:2]

    margin_x = int(img_w * margin)
    margin_y = int(img_h * margin)

    usable_w = img_w - margin_x * 2
    usable_h = img_h - margin_y * 2

    patch_count_x = usable_w // patch_size
    patch_count_y = usable_h // patch_size

    start_x = margin_x + (usable_w - patch_count_x * patch_size) // 2
    start_y = margin_y + (usable_h - patch_count_y * patch_size) // 2

    patch_depth = []

    for y_i in range(patch_count_y):
        for x_i in range(patch_count_x):
            x = start_x + x_i * patch_size
            y = start_y + y_i * patch_size

            patch = depth_img[y:y+patch_size, x:x+patch_size]
            patch_depth.append(float(numpy.median(patch)))

    return patch_depth


def correct_depth_img(depth_img, calibration_patch_depth, patch_size, margin):
    current_patch_depth = calculate_patch_depths(depth_img, patch_size, margin)
    correction = []

    for current_depth, calibration_depth in zip(current_patch_depth, calibration_patch_depth):
        correction.append(calibration_depth / current_depth)

    if len(correction) == 0:
        return depth_img

    return depth_img * float(numpy.median(correction))


def calculate_face_depths(depth_img, face_rect, margin):
    face_depth = []

    for (x, y, w, h) in face_rect:
        x0 = int(x + w * margin)
        x1 = int(x + w * (1 - margin))
        y0 = int(y + h * margin)
        y1 = int(y + h * (1 - margin))

        face_area = depth_img[y0:y1, x0:x1]

        if face_area.size == 0:
            face_depth.append(None)
        else:
            face_depth.append(float(numpy.median(face_area)))

    return face_depth


def main():
    calibration_patch_depth = None
    patch_size = 16
    patch_margin = 0.08

    face_margin = 0.1

    try:
        for frame_img in frames_from_pi(args.host, args.port):
            if args.flip:
                frame_img = cv2.flip(frame_img, 1)

            face_rects = detect_faces(frame_img)
            face_centers = calculate_face_centers(face_rects)

            depth_img, raw_depth = depth_estimation(frame_img)

            if calibration_patch_depth is None:
                calibration_patch_depth = calculate_patch_depths(raw_depth, patch_size, patch_margin)

            corrected_depth_img = correct_depth_img(raw_depth, calibration_patch_depth, patch_size, patch_margin)
            face_depth = calculate_face_depths(corrected_depth_img, face_rects, face_margin)

            print(face_depth)

            final_img = draw_face_rects(depth_img, face_rects)
            final_img = draw_face_centers(final_img, face_centers)
            final_img = cv2.applyColorMap(final_img, cv2.COLORMAP_INFERNO)

            # Display depth result and original frame
            cv2.imshow("Depth", final_img)
            cv2.imshow("Pi Camera", frame_img)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
