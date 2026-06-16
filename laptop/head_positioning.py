from ultralytics import YOLO
import coremltools as ct

import cv2
import numpy
import math

from dataclasses import dataclass

from unicodedata import east_asian_width

# Custom head detection model
head_model = YOLO("./models/head.mlpackage")

# https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino
whenet_model = ct.models.MLModel("./models/whenet.mlpackage")

# https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/blob/master/whenet.py
whenet_mean = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
whenet_std = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
whenet_idx = numpy.array([idx for idx in range(66)], dtype=numpy.float32)
whenet_idx_yaw = numpy.array([idx for idx in range(120)], dtype=numpy.float32)


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
    results = head_model(source=img, conf=conf_thresh, imgsz=640)[0] # Get first (and only) image

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

# https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/blob/master/demo.py
# https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/blob/master/whenet.py
def estimate_head_direction(head_img):
    input_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (224, 224))
    input_img = numpy.expand_dims(input_img, axis=0)
    input_img = input_img / 255
    input_img = (input_img - whenet_mean) / whenet_std

    # https://apple.github.io/coremltools/docs-guides/source/model-prediction.html#image-prediction
    # https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/blob/main/convert_script.txt
    result = whenet_model.predict({"input_1": input_img}) # Model requires input_1

    # https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/blob/main/convert_script.txt#L177
    yaw_predicted = softmax(result["Identity"][0])
    pitch_predicted = softmax(result["Identity_1"][0])
    roll_predicted = softmax(result["Identity_2"][0])
    yaw = float(numpy.sum(yaw_predicted * whenet_idx_yaw) * 3 - 180)
    pitch = float(numpy.sum(pitch_predicted * whenet_idx) * 3 - 99)
    roll = float(numpy.sum(roll_predicted * whenet_idx) * 3 - 99)

    return EulerAngles(yaw, pitch, roll)

# https://medium.com/@amit25173/understanding-softmax-with-numpy-b7273d8ab205
def softmax(x):
    exp_x = numpy.exp(x - numpy.max(x))
    return exp_x / numpy.sum(exp_x)


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
    return numpy.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


# **************************************************
# Position estimation using the pinhole camera model
# **************************************************

def estimate_head_positions(img, focal_length):
    heads = heads_from_frame(img)

    for h in heads:
        e_bbox = calculate_physical_bounding_box(h.angles)

        # https://en.wikipedia.org/wiki/Pinhole_camera_model
        width_factor = focal_length * (e_bbox.width / h.bbox.width)
        height_factor = focal_length * (e_bbox.height / h.bbox.height)

        z = (width_factor + height_factor) / 2

        h.position = Vec3D(h.bbox.x + h.bbox.width/2, h.bbox.y + h.bbox.height/2, z)

    return heads