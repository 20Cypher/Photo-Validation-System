import cv2


def validate_dimensions(image_path):
    REQUIRED_WIDTH = 413
    REQUIRED_HEIGHT = 531

    image = cv2.imread(image_path)
    if image is None:
        return False

    height, width, _ = image.shape
    return width == REQUIRED_WIDTH and height == REQUIRED_HEIGHT
