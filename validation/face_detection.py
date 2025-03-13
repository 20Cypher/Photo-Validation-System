import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)


def detect_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces, _ = mtcnn.detect(image_rgb)

    if faces is None:
        return False, "No face detected"

    return True, "Face detected"
