from client import frames_from_pi
from head_positioning import estimate_head_positions


min_distance = None
max_distance = None


def normalize_positions(heads, frame_img):
    img_h, img_w = frame_img.shape[:2]
    positions = []

    for h in heads:
        x, y, z = h.position.x, h.position.y, h.position.z

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

        heads = estimate_head_positions(frame_img, 500)

        if len(heads) != 1:
            return None, "Exactly one person must be in frame."
        if heads[0].position.z is None:
            return None, "Head distance could not be calculated."

        return heads[0].position.z, "Distance calibrated."

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
            heads = estimate_head_positions(frame_img, 500)
            positions = normalize_positions(heads, frame_img)

            yield positions

    finally:
        frames.close()