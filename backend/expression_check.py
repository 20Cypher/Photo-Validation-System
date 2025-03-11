from deepface import DeepFace

image_path = "test_images\img2.jpg"

# Analyze facial expressions
result = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)

# Extract emotion
dominant_emotion = result[0]['dominant_emotion']

# Check if expression is neutral
is_neutral = dominant_emotion.lower() == "neutral"

print(f"Detected Expression: {dominant_emotion}")
print(f"Neutral Expression: {is_neutral}")
