import cv2
import numpy as np

# Load a pretrained MobileNet SSD (COCO classes)
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# List of class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open camera (try 0, 1, 2 depending on your system)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for DNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
