import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def draw_styled_landmarks(image, results):
    # REMOVED: The heavy 468-point Face Mesh.
    # The Pose Landmarks below will still draw the Nose, Eyes, Ears, and Mouth.
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    features = np.concatenate([pose, lh, rh])
    
    # Normalization: Center coordinates around the Nose (Landmark 0)
    if results.pose_landmarks:
        nose_x = results.pose_landmarks.landmark[0].x
        nose_y = results.pose_landmarks.landmark[0].y
        nose_z = results.pose_landmarks.landmark[0].z
        
        num_points = len(features) // 3
        reshaped = features.reshape(num_points, 3)
        reshaped[:, 0] -= nose_x
        reshaped[:, 1] -= nose_y
        reshaped[:, 2] -= nose_z
        return reshaped.flatten()
        
    return features