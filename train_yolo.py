from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml")

# Train
model.train(data="configs/palm_disease.yaml", epochs=50, imgsz=640)

# Export
model.export(format="onnx")
