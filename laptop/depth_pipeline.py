import cv2
import numpy
from PIL import Image
from transformers import pipeline

face_cascade = cv2.CascadeClassifier("./models/frontalface.xml")

pipe = pipeline(
    task='depth-estimation',
    #model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
    model='Intel/dpt-hybrid-midas',
    device='mps'
)

cam = cv2.VideoCapture(0)

# Detect faces with haar cascades
def detect_faces(img):
    face_img = img.copy()
    return face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=7)

def calculate_face_centers(face_rect):
    face_center = []

    for (x, y, w, h) in face_rect:
        face_center.append((x + w/2, y + h/2))

    return face_center

# Draw face rectangles on image
def draw_face_rects(img, face_rect):
    rect_img = img.copy()

    for (x, y, w, h) in face_rect:
        cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 0, 255), 10)

    return rect_img

def draw_face_centers(img, face_center):
    center_img = img.copy()

    for (x, y) in face_center:
        cv2.circle(center_img, center=(int(x), int(y)), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)

    return center_img;

# Estimate monocular depth from frame
def depth_estimation(img):
    # Convert image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Process image with pipeline
    result = pipe(pil_img)

    # Convert depth result to CV2 format
    return numpy.array(result['depth'])

while True:
    ret, frame = cam.read()
    cam_img = cv2.flip(frame, 1)

    face_rect = detect_faces(cam_img)
    face_center = calculate_face_centers(face_rect)
    depth_img = depth_estimation(cam_img)

    final_img = draw_face_rects(depth_img, face_rect)
    final_img = draw_face_centers(final_img, face_center)

    # Display depth result
    cv2.imshow('Depth', final_img)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()