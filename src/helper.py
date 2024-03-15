import cv2
import mediapipe as mp
from screeninfo import get_monitors

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(50, 205, 50), thickness=2, circle_radius=2)


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

