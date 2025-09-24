from djitellopy import Tello
import cv2
from ultralytics import YOLO
import torch

# Load your trained YOLO model
model = YOLO("/content/runs/detect/train2/weights/best.pt")
classes = ['Healthy', 'Infected']

# Connect to Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

while True:
    frame = frame_read.frame
    if frame is None:
        continue

    # YOLO inference
    results = model(frame, stream=True)

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            label = classes[int(cls)]
            confidence = conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tello YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
