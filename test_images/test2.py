import base64
import requests

API_KEY = "dc764cc198d0f6677db81dbae9dc8ac72fc28136ba629b0e43a4e51979d9d830"

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_photo(image_path, model="qwen/qwen2-vl-72b-instruct"):
    image_base64 = encode_image(image_path)

    url = "https://api.together.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a vision-language model that verifies ID photos for official documentation."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze this image and verify if it meets these requirements:\n"
                                           "- Is the person looking directly at the camera?\n"
                                           "- Are their shoulders visible?\n"
                                           "- Are their ears unobstructed?\n"
                                           "- Is the background white?\n"
                                           "- Does the person have a neutral facial expression (no smile, no open mouth)?\n"
                                           "- Are there any glasses or head coverings?\n"
                                           "Provide a detailed validation report."},
                    {"type": "image", "image": image_base64}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 512
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

image_path = "test_images/img3.jpg"
result = analyze_photo(image_path)

print("Photo Validation Result:\n", result)
