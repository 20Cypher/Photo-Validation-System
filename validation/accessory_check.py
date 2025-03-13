from together import Together
import base64

client = Together(api_key="dc764cc198d0f6677db81dbae9dc8ac72fc28136ba629b0e43a4e51979d9d830")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_accessories(image_path):

    base64_image = encode_image(image_path)

    accessory_prompt = """
    Analyze this image and determine if the person is wearing any accessories:
    - Are there any glasses (prescription glasses, sunglasses)?
    - Is there any headwear (hats, caps, headscarves, or any other head coverings)?

    Provide a clear Yes/No answer for each. Make sure your assessment are always correct since it holds high importance
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": accessory_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        stream=False,
    )

    full_response = response.choices[0].message.content if response.choices else "No response received."

    accessories_present = "Yes" in full_response
    accessory_data = {
        "Glasses Detected": "Yes" in full_response and "glasses" in full_response.lower(),
        "Headwear Detected": "Yes" in full_response and "headwear" in full_response.lower(),
    }

    return accessories_present, accessory_data