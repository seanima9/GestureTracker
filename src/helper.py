import os
import json

import numpy as np
import cv2
import mediapipe as mp
from screeninfo import get_monitors
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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
    Process the frame to get the hand landmarks.
    
    args: frame: the frame to process
    returns: results: the hand landmarks
    """
    flipped_frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    return results, flipped_frame


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
    """
    Compute the minimum and maximum values of the data in the data directory.

    args: data_dir: the directory containing the data

    returns: all_data: a list of all the data in the data directory
                min_val: the minimum value of the data
    """
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

def label_dict(data_dir):
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


def data_generator(data_dir):
    """ 
    Generator function that yields data and labels from the given data directory.

    Args:
        data_dir (str): The directory containing the data.
    
    Yields:
        tuple: A tuple containing the data and the corresponding label. (data, label)
        data (list): A data sample.
        label (int): The label corresponding to the data sample.
    """
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    label_to_index, _ = label_dict(data_dir)
    for subdir in subdirs:
        data_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.json')]
        
        for data_file in data_files:
            with open(data_file, 'r') as f:
                data = json.load(f)

            data = [value for coordinate_dict in data for value in coordinate_dict.values()] 
            data = scaler.transform(np.array(data).reshape(-1, 1))
            data = data.flatten().tolist()
            label_index = label_to_index[os.path.basename(os.path.dirname(data_file))]

            yield (data, tf.constant(label_index, dtype=tf.int32))


def create_and_prepare_dataset(data_dir, batch_size, shuffle_size=1000):
    """ 
    Create and prepare a dataset from the given data directory.
    Will shuffle and batch the dataset.
    
    Args:
        data_dir (str): The directory containing the data.
        batch_size (int): The batch size.
        shuffle_size (int): The shuffle size.
    
    Returns:
        tf.data.Dataset: The dataset containing the data.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data_dir),
        output_signature=(
            tf.TensorSpec(shape=(63,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return dataset.shuffle(shuffle_size).batch(batch_size)


def get_data_size(data):
    """ 
    Get the number of data samples in the given data directory.
    
    Args:
        data (str): The directory containing the data.
    
    Returns:
        int: The number of data samples.
    """
    data_size = 0
    for dir in os.listdir(data):
        for _ in os.listdir(os.path.join(data, dir)):
            data_size += 1

    return data_size