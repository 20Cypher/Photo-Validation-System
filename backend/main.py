from fastapi import FastAPI, UploadFile, File
import uvicorn
from face_detection import detect_face
from background_check import is_white_background
from pose_estimation import check_pose
from expression_check import detect_expression
from accessory_check import detect_accessories

app = FastAPI()


@app.post("/validate/")
async def validate_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run all validation checks
    face_detected = detect_face(file_path, method="mtcnn")
    white_bg = is_white_background(file_path)
    pose_ok = check_pose(file_path)
    expression = detect_expression(file_path)
    accessories = detect_accessories(file_path)

    return {
        "face_detected": face_detected,
        "white_background": white_bg,
        "pose_valid": pose_ok,
        "expression": expression,
        "accessories": accessories
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
