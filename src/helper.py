import os
import json

import numpy as np
import cv2
import mediapipe as mp
from screeninfo import get_monitors
from sklearn.preprocessing import MinMaxScaler

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(50, 205, 50), thickness=2, circle_radius=2)
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, "../data/processed")


################################## Screen related functions ##################################


def get_screen_resolution():
    """
    Gets screen dimensions
    
    returns: total_width: the total width of all monitors
             total_height: the height of the primary monitor
    """
    monitors = get_monitors()
    total_width = sum(monitor.width for monitor in monitors)
    total_height = max(monitor.height for monitor in monitors)

    return total_width, total_height


def process_frame(frame):
    """
    Process the frame using the mediapipe hands model. Will flip the frame and convert it to RGB.
    
    args: frame: the frame to process

    returns: results: the results from the mediapipe hands model
             frame: the processed frame
    """
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    return results, frame


def draw_landmarks(frame, hand_landmarks):
    """
    Draw the landmarks on the frame.
    
    args: frame: the frame to draw the landmarks on
          hand_landmarks: the landmarks to draw on the frame
    
    returns None
    """
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=hand_landmarks, 
        connections=mp.solutions.hands.HAND_CONNECTIONS, 
        landmark_drawing_spec=drawing_spec, 
        connection_drawing_spec=drawing_spec
    )


################################## Machine learning related functions ##################################
    

def compute_min_max(data_dir):
    """Compute the minimum and maximum values of the data."""
    min_val = float('inf')
    max_val = float('-inf')

    all_data = []
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for subdir in subdirs:
        data_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.json')]

        for data_file in data_files:
            with open(data_file, 'r') as f:
                data = json.load(f)

            data = [value for coordinate_dict in data for value in coordinate_dict.values()]  # flatten list of dictionaries
            min_val = min(min_val, min(data))
            max_val = max(max_val, max(data))
            all_data.extend(data)
    return data, min_val, max_val


data, min_val, max_val = compute_min_max(processed_dir)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(np.array(data).reshape(-1, 1))

def lable_dict(data_dir):
    """
    Create a dictionary of labels to indices and indices to labels.
    When training the model, the labels must be converted to indices.
    When predicting with the model, the indices must be converted to labels.

    args: data_dir: the directory containing the data

    returns: label_to_index: a dictionary mapping labels to indices
                index_to_label: a dictionary mapping indices to labels
    """
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subdirs.sort()
    label_to_index = {os.path.basename(label): i for i, label in enumerate(subdirs)}
    index_to_label = {i: os.path.basename(label) for label, i in label_to_index.items()}
    return label_to_index, index_to_label