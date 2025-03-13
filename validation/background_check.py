import cv2
import numpy as np
from retinaface import RetinaFace

def is_white_background(image_path, threshold=0.65):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 160], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    faces = RetinaFace.detect_faces(image_path)
    if isinstance(faces, dict):
        for key in faces:
            x1, y1, x2, y2 = faces[key]["facial_area"]
            mask[y1:y2, x1:x2] = 0

    white_ratio = np.count_nonzero(mask) / mask.size
    return white_ratio > threshold
