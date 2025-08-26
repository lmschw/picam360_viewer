import cv2

# Try different indices if 0 doesn't work (e.g., 1 or 2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open video device")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("PiCam360 Feed", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
