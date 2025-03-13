import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)


def detect_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces, _ = mtcnn.detect(image_rgb)

    if faces is None:
        return False, "No face detected"

    detected_accessories = detect_accessories(image)

    return True, detected_accessories


def detect_accessories(image):
    faces = app.get(image)

    if len(faces) == 0:
        return {"error": "No face detected"}

    glasses_detected = False
    head_cover_detected = False
    glasses_confidence = 0.0
    head_cover_confidence = 0.0

    for face in faces:
        landmarks = face.kps
        left_eye, right_eye = landmarks[0], landmarks[1]

        eye_region = image[int(left_eye[1]) - 10:int(right_eye[1]) +
                           10, int(left_eye[0]) - 10:int(right_eye[0]) + 10]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_intensity = np.mean(gray_eye)

        if eye_intensity < 100:
            glasses_detected = True
            glasses_confidence = min(1.0, (100 - eye_intensity) / 50)

        bbox = face.bbox
        face_height = bbox[3] - bbox[1]
        forehead_region = image[int(
            bbox[1]) - int(0.2 * face_height): int(bbox[1]), int(bbox[0]): int(bbox[2])]
        forehead_intensity = np.mean(forehead_region)

        if forehead_intensity < 80:
            head_cover_detected = True
            head_cover_confidence = min(1.0, (80 - forehead_intensity) / 40)

    return {
        "glasses_detected": glasses_detected,
        "glasses_confidence": round(glasses_confidence, 2),
        "head_cover_detected": head_cover_detected,
        "head_cover_confidence": round(head_cover_confidence, 2)
    }
