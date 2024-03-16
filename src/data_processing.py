import os
import cv2
import mediapipe as mp
import json

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

"""
Plan:
go through each image in each subcategory
keeping track of what subcategory we are in
for each image, we will:
    - read the image
    - use the mediapipe library to detect the hand landmarks
    - save the landmarks to "processed" folder as a json file
    - and into the subcategory folder
    - we will need to check and make sure the folder exists
    - if it doesn't, we will create it
"""

def process_data():
    """
    grabs the landmarks of each image in the raw data folder and saves them to the processed data folder
    """
    mp_hands = mp.solutions.hands.Hands()

    RAW_DATA_PATH = "C:/Users/imani/Documents/gesture_tracker/data/raw"
    PROCESSED_DATA_PATH = "C:/Users/imani/Documents/gesture_tracker/data/processed"

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
                
                landmarks_data = []
                for hand_landmarks in landmarks.multi_hand_landmarks:  # for each hand
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        landmark_dict = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                        hand_data.append(landmark_dict)

                    landmarks_data.append(hand_data)
            
                processed_image_path = os.path.join(processed_path, image.split('.')[0] + '.json')
                with open(processed_image_path, 'w') as f:
                    json.dump(landmarks_data, f)
                print(f"Processed {image_path} and saved to {processed_image_path}")


if __name__ == "__main__":
    process_data()
