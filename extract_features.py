import cv2
import numpy as np
import os
import glob
from utils import extract_landmarks, mediapipe_detection, mp_holistic

if __name__ == "__main__":
    DATA_PATH = os.path.join('data', 'raw')
    OUTPUT_PATH = os.path.join('data', 'processed')

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Please place your sentence folders here.")
        exit()

    sentences = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    print(f"Found {len(sentences)} sentence classes. Starting frame extraction...")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as holistic:
        for sentence in sentences:
            save_dir = os.path.join(OUTPUT_PATH, sentence)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            sentence_path = os.path.join(DATA_PATH, sentence)
            variations = [v for v in os.listdir(sentence_path) if os.path.isdir(os.path.join(sentence_path, v))]
            
            for variation in variations:
                variation_path = os.path.join(sentence_path, variation)
                
                # Sorted reads frames in order: 01.jpg, 02.jpg, 03.jpg
                frame_files = sorted(glob.glob(os.path.join(variation_path, '*.jpg')))
                
                sequence = []
                
                for frame_path in frame_files:
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        continue
                        
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image, results = mediapipe_detection(image, holistic)
                    keypoints = extract_landmarks(results)
                    sequence.append(keypoints)
                
                if len(sequence) > 0:
                    npy_path = os.path.join(save_dir, f"var_{variation}.npy")
                    np.save(npy_path, np.array(sequence))
                    print(f"Processed: '{sentence}' -> Variation {variation} ({len(sequence)} frames)")

    print("✅ Frame extraction complete.")