import cv2
import numpy as np


def is_white_background(image_path, white_ratio_threshold=0.80):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    white_ratio = np.sum(mask == 255) / mask.size

    return white_ratio > white_ratio_threshold


if __name__ == "__main__":
    result = is_white_background("test_images/img1.webp")
    print("White Background:", result)
