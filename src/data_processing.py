import os
import cv2
import mediapipe as mp
import json
import shutil
import splitfolders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/raw")
RAW_VIDEO_PATH = os.path.join(SCRIPT_DIR, "../data/raw_videos")
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/processed")
SPLIT_DATA_PATH = os.path.join(SCRIPT_DIR, "../data/split")


def clean_directory(dir_path):
    """
    Delete all files and subdirectories in a directory

    Args:
    dir_path (str): path to the directory

    Returns:
    None
    """
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def read_video(video_path, category):
    """ 
    Read video and save every 5th frame as an image in the raw data folder

    Args:
    video_path (str): path to the video file
    category (str): category of the video
    save_rate (int): save every nth frame

    Returns:
    None
    """
    frame_count = 0
    save_count = 0
    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    save_rate = int(frame_rate / 5)

    if not os.path.exists(os.path.join(RAW_DATA_PATH, category)):
        os.makedirs(os.path.join(RAW_DATA_PATH, category))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % save_rate == 0:
            save_count += 1
            frame_path = os.path.join(RAW_DATA_PATH, category, f"{save_count}.jpg")
            cv2.imwrite(frame_path, frame)

    cap.release()
    cv2.destroyAllWindows()


def process_data(total_images):
    """
    Process the raw data and save the hand landmarks as json files in the processed data folder

    Args:
    total_images (int): total number of images to process

    Returns:
    None
    """
    mp_hands = mp.solutions.hands.Hands()
    count = 0

    for subcategory in os.listdir(RAW_DATA_PATH):
        subcategory_path = os.path.join(RAW_DATA_PATH, subcategory)
        if not os.path.isdir(subcategory_path):
            continue

        for image in os.listdir(subcategory_path):
            image_path = os.path.join(subcategory_path, image)

            img = cv2.imread(image_path)
            img = cv2.flip(img, 1)
            landmarks = mp_hands.process(img)

            if landmarks.multi_hand_landmarks:
                count += 1
                if count >= total_images:
                    break

                processed_path = os.path.join(PROCESSED_DATA_PATH, subcategory)
                if not os.path.exists(processed_path):
                    os.makedirs(processed_path)
                
                hand_landmarks = landmarks.multi_hand_landmarks[0]  # Only one hand is detected
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    landmark_dict = {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                    hand_data.append(landmark_dict)

                processed_image_path = os.path.join(processed_path, image.split('.')[0] + '.json')
                with open(processed_image_path, 'w') as f:
                    json.dump(hand_data, f, indent=4)
        
        if count >= total_images:
            break


def main():
    """ 
    Read videos, process data, and split data into training, validation, and test sets
    Cleans the directories before starting
    """
    print("Cleaning directories...")
    clean_directory(RAW_DATA_PATH)
    clean_directory(PROCESSED_DATA_PATH)
    clean_directory(SPLIT_DATA_PATH)
    print("Directories cleaned!")

    print("Reading videos...")
    for video in os.listdir(RAW_VIDEO_PATH):
        video_path = os.path.join(RAW_VIDEO_PATH, video)
        category = video.split('.')[0]
        read_video(video_path, category)
    print("Video reading complete!")

    print("Processing data...")
    process_data(500)
    print("Data processing complete!")

    if not os.path.exists(SPLIT_DATA_PATH):
        os.makedirs(SPLIT_DATA_PATH)

    splitfolders.ratio(PROCESSED_DATA_PATH, output=SPLIT_DATA_PATH, seed=1337, ratio=(.7, .15, .15))
    print("Data split complete!")


if __name__ == "__main__":
    main()