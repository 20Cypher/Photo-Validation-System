from together import Together
import base64

# Initialize Together.AI Client
client = Together(api_key="dc764cc198d0f6677db81dbae9dc8ac72fc28136ba629b0e43a4e51979d9d830")  # Ensure API key is set

# Path to Image
image_path = "test_images/img3.jpg"  # Replace with your image path

# Function to Encode Image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode Image
base64_image = encode_image(image_path)

# Photo Validation Prompt
validation_prompt = """
Analyze this image of a person and determine if it meets these criteria for an ID/passport photo:

1. Is the person looking directly at the camera?
2. Are their shoulders visible?
3. Are their ears unobstructed?
4. Is the background white?
5. Does the person have a neutral facial expression (no smile, no open mouth)?
6. Are there any glasses or head coverings?

For each criterion, provide a Yes/No answer and a brief explanation.
Then give an overall assessment of whether the photo meets ID/passport standards.
"""

# Call Together.AI API
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",  # Latest VLM model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": validation_prompt},
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

# Print Validation Result
print("\n=== PHOTO VALIDATION REPORT ===\n")
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
