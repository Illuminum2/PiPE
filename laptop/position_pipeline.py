from ultralytics import YOLO

from client import frames_from_pi
from head_distance import head_distance_from_frame


head_model = YOLO("./models/head.mlpackage")


min_distance = None
max_distance = None


def process_frame(frame_img):
    head_rects, head_centers = detect_heads(frame_img)
    head_distances = calculate_head_distances(frame_img, head_rects)

    return head_centers, head_distances


# Detect heads with custom YOLO model
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


def calculate_head_distances(frame_img, head_rects):
    head_distances = []
    focal_length = frame_img.shape[1] # Temporary focal length

    for head_rect in head_rects:
        distance = head_distance_from_frame(frame_img, head_rect, focal_length)
        head_distances.append(distance)

    return head_distances


def normalize_positions(head_centers, head_distances, frame_img):
    img_h, img_w = frame_img.shape[:2]
    positions = []

    for (x, y), z in zip(head_centers, head_distances):
        if z is None or z < min_distance or z > max_distance:
            continue

        positions.append((
            x / img_w,
            y / img_h,
            (z - min_distance) / (max_distance - min_distance)
        ))

    return positions


def set_min_distance(distance):
    global min_distance

    if distance is None:
        return False, "Min distance must be set."
    if max_distance is not None and distance >= max_distance:
        return False, "Min distance must be smaller than max distance."

    min_distance = distance
    return True, f"Min distance set to {min_distance}m."

def set_max_distance(distance):
    global max_distance

    if distance is None:
        return False, "Max distance must be set."
    if min_distance is not None and distance <= min_distance:
        return False, "Max distance must be smaller than max distance."

    max_distance = distance
    return True, f"Max distance set to {max_distance}m."

def min_distance_calibration():
    global min_distance

    distance, message = distance_calibration()

    if distance is None:
        return False, message
    if max_distance is not None and distance >= max_distance:
        return False, "Min distance must be smaller than max distance."

    min_distance = distance

    return True, f"Min distance calibrated to {min_distance}m."

def max_distance_calibration():
    global max_distance

    distance, message = distance_calibration()

    if distance is None:
        return False, message
    if min_distance is not None and distance <= min_distance:
        return False, "Max distance must be larger than min distance."

    max_distance = distance

    return True, f"Max distance calibrated to {max_distance}m."


def distance_calibration():
    frames = frames_from_pi()

    try:
        frame_img = next(frames)

        head_centers, head_distances = process_frame(frame_img)

        if len(head_centers) != 1:
            return None, "Exactly one person must be in frame."
        if head_distances[0] is None:
            return None, "Head distance could not be calculated."

        return head_distances[0], "Distance calibrated."

    finally:
        frames.close()

def is_calibrated():
    return (
        min_distance is not None and
        max_distance is not None
    )


def people_positions():
    if not is_calibrated():
        raise RuntimeError("Not calibrated.")

    frames = frames_from_pi()

    try:
        for frame_img in frames:
            head_centers, head_distances = process_frame(frame_img)

            positions = normalize_positions(head_centers, head_distances, frame_img)

            yield positions

    finally:
        frames.close()