import torch
import cv2
from yolov5 import detect

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def detect_accessories(image_path):
    results = model(image_path)
    labels = results.pandas().xyxy[0]["name"].tolist()

    glasses_present = "glasses" in labels
    head_cover_present = "hat" in labels or "headscarf" in labels

    return {"glasses": glasses_present, "head_covering": head_cover_present}


if __name__ == "__main__":
    image_path = "test_images\img2.jpg"
    result = detect_accessories(image_path)
    print(result)
