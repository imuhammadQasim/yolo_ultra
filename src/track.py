from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("model/yolo26n.pt")

# Open video
cap = cv2.VideoCapture("data/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Draw detections + tracking IDs
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
