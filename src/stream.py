import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Camera is ready")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break