import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
classes = ['Healthy', 'Infected']

def detect_and_plot(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            label = classes[int(cls)]
            confidence = conf.item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_and_plot("test.jpg")
