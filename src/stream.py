import cv2
import mediapipe as mp
import pyautogui
from helper import get_screen_resolution, process_frame, draw_landmarks

FRAME_WIDTH_ID = 3
FRAME_HEIGHT_ID = 4
PRIMARY_CAMERA_ID = 0
pyautogui.FAILSAFE = False
prev_x, prev_y = 0, 0


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
    Perform action based on hand landmarks. Will pass landmarks into a model to get the action.
    Then use pyautogui to perform the action.

    args: hand_landmarks: mediapipe hand landmarks

    returns None
    """
    pass


def stream():
    """
    Stream the camera and process the frames. Will draw landmarks and move the mouse based on the hand landmarks.

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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        results, frame = process_frame(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
                move_mouse(hand_landmarks)
                perform_action(hand_landmarks)
                break
        
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    screen_width, screen_height = get_screen_resolution()
    stream()
