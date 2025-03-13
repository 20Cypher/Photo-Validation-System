from fastapi import FastAPI, UploadFile, File
import uvicorn
import shutil
from validation.face_detection import detect_face
from validation.background_check import is_white_background
from validation.pose_estimation import check_pose
from validation.expression_check import detect_expression
from validation.accessory_check import detect_accessories
from validation.dimension_check import validate_dimensions

app = FastAPI()

@app.post("/validate/")
async def validate_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    dimensions_ok = validate_dimensions(file_path)
    face_detected, face_message = detect_face(file_path)
    white_bg = is_white_background(file_path)
    pose_result = check_pose(file_path)

    expression_ok = detect_expression(file_path)
    accessories_ok, accessories_data = detect_accessories(file_path)

    return {
        "Dimensions Valid": dimensions_ok,
        "Face Detected": face_message,
        "White Background": white_bg,
        "Pose Valid": pose_result["status"],
        "Pose Checks": pose_result["checks"],
        "Neutral Expression": expression_ok,
        "Accessories": accessories_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
