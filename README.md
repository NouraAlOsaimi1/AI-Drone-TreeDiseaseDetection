# AI-Drone-TreeDiseaseDetection
This project uses a DJI Tello drone and YOLOv8 to detect palm tree diseases (Healthy vs Infected) in real time.
## Features
- YOLOv8 training on custom palm tree dataset
- Exported model in ONNX and PyTorch format
- Real-time detection with DJI Tello drone
- Visualization of detection results
## Project Structure
AI-Drone-TreeDiseaseDetection/
├── train_yolo.py # Training YOLO model
├── detect_yolo.py # Testing detection on images
├── tello_yolo.py # Real-time detection with DJI Tello drone
├── configs/
│ └── palm_disease.yaml
├── requirements.txt
└── README.md

---

### `requirements.txt`
```txt
ultralytics
torch
opencv-python
djitellopy
matplotlib
pyyaml
