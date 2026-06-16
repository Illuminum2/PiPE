from ultralytics import YOLO
import coremltools as ct

import cv2
import numpy
import math

from dataclasses import dataclass

# Custom head detection model
head_model = YOLO("./models/head.mlpackage", task="detect")

# https://github.com/thohemp/6DRepNet360
sixdrepnet_model = ct.models.MLModel("./models/6DRepNet360.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

# https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/test.py
sixdrepnet_mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
sixdrepnet_std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)


# ********************************************
# Classes representing a head and bounding box
# *******************************************

@dataclass
class Bbox:
    x: int
    y: int
    width: int
    height: int

@dataclass
class EulerAngles:
    yaw: float
    pitch: float
    roll: float

@dataclass
class Vec3D:
    x: float
    y: float
    z: float

@dataclass
class Head:
    bbox: Bbox
    angles: EulerAngles
    position: Vec3D | None = None


# ******************************
# Head bounding boxes estimation
# ******************************

# https://docs.ultralytics.com/usage/python#predict
# https://docs.ultralytics.com/modes/predict#inference-arguments
def estimate_head_bounding_boxes(img, conf_thresh=0.6):
    results = head_model(source=img, conf=conf_thresh, imgsz=640, verbose=False)[0] # Get first (and only) image

    bboxes = []

    for box in results.boxes:
        bx0, by0, bx1, by1 = box.xyxy[0]
        bboxes.append(clip_to_bbox(img, bx0, by0, bx1, by1))

    return bboxes

def clip_to_bbox(img, x0, y0, x1, y1):
    img_h, img_w = img.shape[:2]

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(img_w, int(x1))
    y1 = min(img_h, int(y1))

    return Bbox(x0, y0, x1-x0, y1-y0)


# ******************************
# Head view direction estimation
# ******************************

# https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/test.py
# https://github.com/thohemp/6DRepNet360/blob/master/sixdrepnet360/utils.py
def estimate_head_direction(head_img):
    #input_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
    input_img = head_img.copy()
    input_img = cv2.resize(input_img, (256, 256))
    input_img = input_img[16:240, 16:240]
    input_img = numpy.expand_dims(input_img, axis=0)
    input_img = input_img / 255
    input_img = (input_img - sixdrepnet_mean) / sixdrepnet_std
    input_img = numpy.transpose(input_img, (0, 3, 1, 2))

    # https://apple.github.io/coremltools/docs-guides/source/model-prediction.html#image-prediction
    result = sixdrepnet_model.predict({"input": input_img})
    pitch, yaw, roll = result["euler"][0]

    return EulerAngles(float(-yaw), float(pitch), float(roll))


# *******************************************************
# Extract head bounding box and view direction from frame
# *******************************************************

def heads_from_frame(img):
    bboxes = estimate_head_bounding_boxes(img)

    heads = []

    for bbox in bboxes:
        head_img = img[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
        angles = estimate_head_direction(head_img)

        heads.append(Head(bbox, angles))

    return heads


# **************************************************************
# Rotated and projected head ellipsis for real life bounding box
# **************************************************************

# https://math.stackexchange.com/questions/1403126/what-is-the-general-equation-for-rotated-ellipsoid
# https://math.stackexchange.com/questions/4442006/orthogonal-projection-of-an-ellipso%C3%AFd-from-n-to-2-dimensional-space
# https://math.stackexchange.com/questions/1835198/largest-rotated-ellipse-inscribed-in-a-rectangle
def calculate_physical_bounding_box(angles, head_x=0.16, head_y=0.2, head_z=0.20):
    # https://www.physicsforums.com/threads/the-very-very-general-equation-of-an-ellipsoid-who-knows-it.379555/
    # https://www.youtube.com/watch?v=lFkVv_EJ6AM
    rotation_matrix = get_rotation_matrix(angles.yaw, angles.pitch, angles.roll)
    a = numpy.diag([(head_x / 2)**2, (head_y / 2)**2, (head_z / 2)**2])

    c = rotation_matrix @ a @ rotation_matrix.T

    # AABB width/height after orthographic projection to 2D
    # https://blog.yiningkarlli.com/2013/02/bounding-boxes-for-ellipsoids.html
    # https://stackoverflow.com/questions/4368961/calculating-an-aabb-for-a-transformed-sphere/4369956#4369956
    # https://tavianator.com/2014/ellipsoid_bounding_boxes.html
    return Bbox(0, 0, math.sqrt(c[0, 0]) * 2, math.sqrt(c[1, 1]) * 2) # x and y don't mean anything

# https://gist.github.com/jamesgregson/67eb5509af0d8b372f25146d5e3c5149
def get_rotation_matrix(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
    # 6DRepNet returns rotations in xyz order
    return numpy.array([
        [cy*cr, -cy*sr, sy],
        [cp*sr + sp*sy*cr, cp*cr - sp*sy*sr, -sp*cy],
        [sp*sr - cp*sy*cr, sp*cr + cp*sy*sr, cp*cy]
    ])


# **************************************************
# Position estimation using the pinhole camera model
# **************************************************

# https://en.wikipedia.org/wiki/Pinhole_camera_model
# https://mayavan95.medium.com/3d-position-estimation-of-a-known-object-using-a-single-camera-7a82b37b326b
def estimate_head_positions(frame):
    img = frame.img.copy()
    img_h, img_w = img.shape[:2]

    fx, fy = pixel_focal_lengths(frame)

    cx = img_w / 2
    cy = img_h / 2

    heads = heads_from_frame(img)

    for h in heads:
        e_bbox = calculate_physical_bounding_box(h.angles)

        width_factor = (fx * e_bbox.width) / h.bbox.width
        height_factor = (fy * e_bbox.height) / h.bbox.height

        z = (width_factor + height_factor) / 2

        x = ((h.bbox.x + h.bbox.width/2 - cx) * z) / fx
        y = ((h.bbox.y + h.bbox.height/2 - cy) * z) / fy

        h.position = Vec3D(x, y, z)

    return heads

# https://gist.github.com/Shrinks99/04ecc04da478ee6bd4a53c935971e3c5
def pixel_focal_lengths(frame):
    img_h, img_w = frame.img.shape[:2]

    sensor_width = frame.unit_cell_size[0] / 1000000 * img_w
    sensor_height = frame.unit_cell_size[1] / 1000000 * img_h

    return (
        frame.focal_length / sensor_width * img_w,
        frame.focal_length / sensor_height * img_h
    )