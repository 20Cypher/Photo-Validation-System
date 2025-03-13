from together import Together
import base64

from together import Together

# Initialize Together.AI client with API key
client = Together(api_key="dc764cc198d0f6677db81dbae9dc8ac72fc28136ba629b0e43a4e51979d9d830")  # Replace with your API key

getDescriptionPrompt = "Analyze this image and verify if it meets these requirements: Is the person looking directly at the camera? Are their shoulders visible? Are their ears unobstructed? Is the background white? Does the person have a neutral facial expression(no smile, no open mouth)? Are there any glasses or head coverings? Provide a detailed validation report."


imagePath = "test_images\img2.jpg"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


base64_image = encode_image(imagePath)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": getDescriptionPrompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
