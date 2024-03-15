import cv2
import mediapipe as mp


FRAME_WIDTH_ID = 3
FRAME_HEIGHT_ID = 4
PRIMARY_CAMERA_ID = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(50, 205, 50), thickness=2, circle_radius=2)

def stream():
    """
    Stream the video from the primary camera and track the hand gestures using the MediaPipe library.
    Get the hand landmarks and use gesture recognition model to detect the gestures.
    """

    cap = cv2.VideoCapture(PRIMARY_CAMERA_ID)
    if cap.isOpened():
        print("Camera is ready")
    else:
        print("Camera is not ready. Exiting...")
        return
    
    cap.set(FRAME_WIDTH_ID, 1280)
    cap.set(FRAME_HEIGHT_ID, 720)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image=frame, 
                landmark_list=hand_landmarks, 
                connections=mp_hands.HAND_CONNECTIONS, 
                landmark_drawing_spec=drawing_spec, 
                connection_drawing_spec=drawing_spec
            )
        
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream()
