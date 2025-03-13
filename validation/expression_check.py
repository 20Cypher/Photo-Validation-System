from deepface import DeepFace


def detect_expression(image_path):
    result = DeepFace.analyze(img_path=image_path, actions=[
                              "emotion"], enforce_detection=False)
    dominant_emotion = result[0]['dominant_emotion'].lower()

    return dominant_emotion == "neutral"
