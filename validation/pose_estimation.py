import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def check_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"status": False, "message": "Image not found or corrupted"}

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_image)
    pose_results = pose.process(rgb_image)

    if not face_results.multi_face_landmarks or not pose_results.pose_landmarks:
        return {"status": False, "message": "Face or pose not detected"}

    face_landmarks = face_results.multi_face_landmarks[0].landmark
    pose_landmarks = pose_results.pose_landmarks.landmark

    result = {"status": True, "message": "Pose is valid", "checks": {}}

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulders_visible = left_shoulder.visibility > 0.6 and right_shoulder.visibility > 0.6

    result["checks"]["shoulders_visible"] = shoulders_visible
    if not shoulders_visible:
        result["status"] = False
        result["message"] = "Shoulders not visible"

    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    eye_distance = abs(left_eye.x - right_eye.x)

    looking_straight = eye_distance > 0.035
    result["checks"]["looking_directly_at_camera"] = looking_straight

    if not looking_straight:
        result["status"] = False
        result["message"] = "Not looking directly at the camera"

    left_ear = pose_landmarks[7]
    right_ear = pose_landmarks[8]
    ears_visible = left_ear.visibility > 0.5 and right_ear.visibility > 0.5

    result["checks"]["ears_unobstructed"] = ears_visible
    if not ears_visible:
        result["status"] = False
        result["message"] = "Ears not unobstructed"

    return result