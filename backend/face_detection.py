import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(keep_all=True, device=device)


def detect_face(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces, _ = mtcnn.detect(image_rgb)

    if faces is None:
        print("No face detected")
        return None

    for (x_min, y_min, x_max, y_max) in faces:
        cv2.rectangle(image_rgb, (int(x_min), int(y_min)),
                      (int(x_max), int(y_max)), (0, 255, 0), 2)

    # Display Image with Detected Faces using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.axis("off")  # Hide axis for better visualization
    plt.show()

    return faces


if __name__ == "__main__":
    result = detect_face("test_images/img1.webp")
    print(result)