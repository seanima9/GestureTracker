import os

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tensorflow as tf

from src.helper import get_screen_resolution, process_frame, draw_landmarks, lable_dict, scaler

FRAME_WIDTH_ID = 3
FRAME_HEIGHT_ID = 4
PRIMARY_CAMERA_ID = 0
pyautogui.FAILSAFE = False
prev_x, prev_y = 0, 0

screen_width, screen_height = get_screen_resolution()
current_dir = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(current_dir, "../models/model_acc_0.91_loss_0.18.h5"))
train_dir = os.path.join(current_dir, "../data/split/train")


def move_mouse(hand_landmarks):
    """
    Move the mouse based on the hand landmarks. Will use the index finger tip to move the mouse.
    Only moves if the thumb is below the index finger tip.
    Has a "fast mode" if the middle finger is below the thumb.
    
    args: hand_landmarks: mediapipe hand landmarks
    
    returns None
    """
    global prev_x, prev_y

    x = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x
    y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y

    thumb_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y
    middle_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y

    if thumb_y * 0.9 < y:  # (0,0) is at the top left corner so we need to invert the y axis
        prev_x, prev_y = x, y  # Reset previous positions when hand is not in control position to avoid sudden jumps
        return
    
    sensitivity = 1
    smoothing = 0.5

    if middle_y > thumb_y:  # If middle finger is below thumb, increase sensitivity
        sensitivity = 3

    x = prev_x * smoothing + x * (1 - smoothing)
    y = prev_y * smoothing + y * (1 - smoothing)
    
    dx = x - prev_x
    dy = y - prev_y
    if abs(dx) > 0.1 or abs(dy) > 0.1:
        dx = dy = 0

    mouse_dx = int(dx * screen_width * sensitivity)
    mouse_dy = int(dy * screen_height * sensitivity)

    pyautogui.move(mouse_dx, mouse_dy, duration=0.1)

    prev_x, prev_y = x, y


def perform_action(hand_landmarks):
    """
    Gets hand landmarks, processes them, then puts them through the model to predict the action.
    Then uses the predicted action to perform the action.

    args: hand_landmarks: mediapipe hand landmarks

    returns None
    """
    data = [value for landmark in hand_landmarks.landmark for value in [landmark.x, landmark.y, landmark.z]]

    processed_data = scaler.transform(np.array(data).reshape(-1, 1))
    processed_data = processed_data.flatten().tolist()
    threshold = 0.5

    input_data = np.array([processed_data])
    predictions = model.predict(input_data)
    max_prediction = np.max(predictions[0])
    _, index_to_label = lable_dict(train_dir)
    print(index_to_label)  # TODO: Remove this line
    print(predictions)  # TODO: Remove this line

    if max_prediction > threshold:
        predicted_class = index_to_label[np.argmax(predictions[0])]
    else:
        predicted_class = None

    if predicted_class == "left_click":
        pyautogui.click()
        print(predicted_class)  # TODO: Remove this line

    elif predicted_class == "cursor":
        print(predicted_class)  # TODO: Remove this line

    else:
        print("No action")  # TODO: Remove this line


def stream():
    """
    Stream the camera and process the frames. Will draw landmarks and move the mouse based on the hand landmarks.
    Will also move mouse and perform actions based on the hand landmarks.

    returns None
    """
    cap = cv2.VideoCapture(PRIMARY_CAMERA_ID, cv2.CAP_DSHOW)
    if cap.isOpened():
        print("Camera is ready")
    else:
        print("Camera is not ready. Exiting...")
        return
    
    cap.set(FRAME_WIDTH_ID, 1280)
    cap.set(FRAME_HEIGHT_ID, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    frame_count = 0
    prediction_frequency = 10  # Make a prediction every 10 frames

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        results, frame = process_frame(frame)

        if results.multi_hand_landmarks:
            # print(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                draw_landmarks(frame, hand_landmarks)
                move_mouse(hand_landmarks)
                
                if frame_count % prediction_frequency == 0:
                    perform_action(hand_landmarks)
                
                break
        
        frame_count += 1

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream()
