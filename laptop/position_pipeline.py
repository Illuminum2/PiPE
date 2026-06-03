import cv2
import numpy
from PIL import Image

from transformers import pipeline
from ultralytics import YOLO

from client import frames_from_pi


head_model = YOLO("./models/head.mlpackage")

depth_pipe = pipeline(
    task='depth-estimation',
    model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
    #model='Intel/dpt-hybrid-midas',
    device='mps'
)

calibration_patches = None
min_depth = None
max_depth = None

patch_size = 16
patch_margin = 0.08
face_margin = 0.1


def process_frame(frame_img):
    raw_depth = depth_estimation(frame_img)

    face_rects, face_centers = detect_heads(frame_img)
    corrected_depth = correct_depth(raw_depth)

    face_depths = calculate_face_depths(corrected_depth, face_rects)

    return face_centers, face_depths


# Detect faces with custom YOLO model
# https://docs.ultralytics.com/usage/python#predict
# https://docs.ultralytics.com/modes/predict#inference-arguments
def detect_heads(img, conf_thresh=0.5):
    results = head_model(source=img, conf=conf_thresh, imgsz=640)[0] # Get first (and only) image

    head_rects = []
    head_centers = []

    for box in results.boxes:
        x, y, w, h = box.xywh[0]

        head_rects.append([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)]) # x and y from xywh are center
        head_centers.append((int(x), int(y)))

    return head_rects, head_centers

def frame_contains_person(frame_img):
    return len(detect_heads(frame_img)[0]) > 0

# Estimate monocular depth from frame
def depth_estimation(img):
    # Convert image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Process image with pipeline
    result = depth_pipe(pil_img)

    # Normalized display depth
    #depth_img = numpy.array(result['depth'])

    # Raw depth for calculations
    raw_depth = result['predicted_depth'].cpu().numpy()

    return raw_depth

def correct_depth(raw_depth):
    current_patches = calculate_patch_depths(raw_depth)
    correction = []

    for current_patch, calibration_patch in zip(current_patches, calibration_patches):
        if current_patch != 0:
            correction.append(calibration_patch / current_patch)
        else:
            correction.append(current_patch)

    if len(correction) == 0:
        return raw_depth

    return raw_depth * float(numpy.median(correction))

def calculate_face_depths(depth, face_rects):
    face_depths = []

    for (x, y, w, h) in face_rects:
        x0 = int(x + w * face_margin)
        x1 = int(x + w * (1 - face_margin))
        y0 = int(y + h * face_margin)
        y1 = int(y + h * (1 - face_margin))

        face_area = depth[y0:y1, x0:x1]
        face_depths.append(float(numpy.median(face_area)))

    return face_depths

def normalize_positions(face_centers, face_depths, frame_img):
    img_h, img_w = frame_img.shape[:2]
    positions = []

    for (x, y), z in zip(face_centers, face_depths):
        if z < min_depth or z > max_depth:
            continue

        positions.append((
            x / img_w,
            y / img_h,
            (z - min_depth) / (max_depth - min_depth)
        ))

    return positions


def background_calibration():
    global calibration_patches, min_depth, max_depth

    frames = frames_from_pi()

    try:
        frame_img = next(frames)

        raw_depth = depth_estimation(frame_img)

        if frame_contains_person(frame_img):
            return False, "No person must be in frame."

        calibration_patches = calculate_patch_depths(raw_depth)
        min_depth = None
        max_depth = None

        return True, "Background calibrated."

    finally:
        frames.close()

def calculate_patch_depths(depth):
    img_h, img_w = depth.shape[:2]

    margin_x = int(img_w * patch_margin)
    margin_y = int(img_h * patch_margin)

    usable_w = img_w - margin_x * 2
    usable_h = img_h - margin_y * 2

    patch_count_x = usable_w // patch_size
    patch_count_y = usable_h // patch_size

    start_x = margin_x + (usable_w - patch_count_x * patch_size) // 2
    start_y = margin_y + (usable_h - patch_count_y * patch_size) // 2

    patch_depths = []

    for patch_x in range(patch_count_x):
        for patch_y in range(patch_count_y):
            x = start_x + patch_x * patch_size
            y = start_y + patch_y * patch_size

            patch = depth[y:y+patch_size, x:x+patch_size]
            patch_depths.append(float(numpy.median(patch)))

    return patch_depths


def min_depth_calibration():
    global min_depth

    depth, message = depth_calibration()

    if depth is None:
        return False, message
    if max_depth is not None and depth >= max_depth:
        return False, "Min depth must be smaller than max depth."

    min_depth = depth

    return True, "Min depth calibrated."

def max_depth_calibration():
    global max_depth

    depth, message = depth_calibration()

    if depth is None:
        return False, message
    if min_depth is not None and depth <= min_depth:
        return False, "Max depth must be larger than min depth."

    max_depth = depth

    return True, "Max depth calibrated."

def depth_calibration():
    if calibration_patches is None:
        return None, "Background is not calibrated."

    frames = frames_from_pi()

    try:
        frame_img = next(frames)

        face_centers, face_depths = process_frame(frame_img)

        if len(face_centers) != 1:
            return None, "Exactly one person must be in frame."

        return face_depths[0], "Depth calibrated."

    finally:
        frames.close()

def is_calibrated():
    return (
        calibration_patches is not None
        and min_depth is not None
        and max_depth is not None
    )


def people_positions():
    if not is_calibrated():
        raise RuntimeError("Not calibrated.")

    frames = frames_from_pi()

    try:
        for frame_img in frames:
            face_centers, face_depths = process_frame(frame_img)

            positions = normalize_positions(face_centers, face_depths, frame_img)

            yield positions

    finally:
        frames.close()