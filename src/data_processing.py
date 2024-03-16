import os
import cv2
import mediapipe as mp
import json
import splitfolders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/raw")
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/processed")
SPLIT_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/split")

def process_data():
    mp_hands = mp.solutions.hands.Hands()

    for subcategory in os.listdir(RAW_DATA_PATH):
        subcategory_path = os.path.join(RAW_DATA_PATH, subcategory)
        if not os.path.isdir(subcategory_path):
            continue

        for image in os.listdir(subcategory_path):
            image_path = os.path.join(subcategory_path, image)
            if os.path.exists(os.path.join(PROCESSED_DATA_PATH, subcategory, image)):
                continue

            img = cv2.imread(image_path)
            landmarks = mp_hands.process(img)

            if landmarks.multi_hand_landmarks:
                processed_path = os.path.join(PROCESSED_DATA_PATH, subcategory)
                if not os.path.exists(processed_path):
                    os.makedirs(processed_path)
                
                hand_landmarks = landmarks.multi_hand_landmarks[0]  # for each hand
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    landmark_dict = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                    hand_data.append(landmark_dict)
            
                processed_image_path = os.path.join(processed_path, image.split('.')[0] + '.json')
                with open(processed_image_path, 'w') as f:
                    json.dump(hand_data, f, indent=4)


if __name__ == "__main__":
    print("Processing data...")
    process_data()
    print("Data processing complete!")
    if not os.path.exists(SPLIT_DATA_PATH):
        os.makedirs(SPLIT_DATA_PATH)
    splitfolders.ratio(PROCESSED_DATA_PATH, output=SPLIT_DATA_PATH, seed=1337, ratio=(.7, .15, .15))
    print("Data split complete!")