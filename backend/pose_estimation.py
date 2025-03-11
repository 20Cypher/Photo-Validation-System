import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh()
pose = mp_pose.Pose()


def check_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False, "Image not found or corrupted"

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    face_results = face_mesh.process(rgb_image)
    if not face_results.multi_face_landmarks:
        return False, "No face detected"
    face_landmarks = face_results.multi_face_landmarks[0]

    # Convert face landmarks to numpy array
    face_points = np.array([(lm.x, lm.y, lm.z)
                           for lm in face_landmarks.landmark])

    # Pose detection (for shoulders)
    pose_results = pose.process(rgb_image)
    if not pose_results.pose_landmarks:
        return False, "No pose detected"
    pose_landmarks = pose_results.pose_landmarks.landmark

    # Criteria 1: Looking directly at the camera
    nose = face_points[1]
    left_eye = face_points[33]  # Left eye center
    right_eye = face_points[263]  # Right eye center

    eye_alignment = abs(left_eye[1] - right_eye[1]
                        ) < 0.02  # Ensure eyes are level
    nose_centered = abs(nose[0] - 0.5) < 0.05  # Ensure nose is near center
    looking_straight = eye_alignment and nose_centered

    # Criteria 2: Ears unobstructed
    left_ear = face_points[234]
    right_ear = face_points[454]

    # Ensure ears are within frame bounds
    ears_visible = (left_ear[0] > 0.05 and right_ear[0] < 0.95)

    # Criteria 3: Shoulders visible
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    shoulders_visible = (left_shoulder.visibility >
                         0.5 and right_shoulder.visibility > 0.5)

    # Generate validation report
    validation_results = {
        "Looking Straight": looking_straight,
        "Ears Unobstructed": ears_visible,
        "Shoulders Visible": shoulders_visible
    }

    return all(validation_results.values()), validation_results


if __name__ == "__main__":
    result = check_pose("test_images/img1.webp")
    print(result)
