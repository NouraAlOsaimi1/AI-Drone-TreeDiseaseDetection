# AI-Drone-TreeDiseaseDetection
This project uses a DJI Tello drone and YOLOv8 to detect palm tree diseases (Healthy vs Infected) in real time.
## Features
- YOLOv8 training on custom palm tree dataset
- Exported model in ONNX and PyTorch format
- Real-time detection with DJI Tello drone
- Visualization of detection results
## Project Structure
- `train_yolo.py` → Train YOLO model
- `detect_yolo.py` → Test detection on images
- `tello_yolo.py` → Real-time detection with Tello drone
- `palm_disease.yaml` → Dataset config file
- ## How to Run?
```bash
pip install -r requirements.txt
python tello_yolo.py
