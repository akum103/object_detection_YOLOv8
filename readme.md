# 🚀 Object Detection using YOLOv8

## 📌 Overview
This project implements **YOLOv8 (You Only Look Once v8)** for **object detection** using the **VOC2012 dataset**. It detects objects in images using the latest YOLOv8 model.

## 📂 Dataset
- **Dataset:** PASCAL VOC 2012
- **Location:** Stored in Google Drive
- **Total Images:** 📸 (Update after running the count in your notebook)

## 🛠️ Installation
To run this project in **Google Colab**, follow these steps:

```bash
!pip install ultralytics opencv-python-headless
Mount your Google Drive:

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
🏗️ Model Setup
Using the pre-trained YOLOv8 model:

python
Copy
Edit
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Run detection on a sample image
results = model("/path/to/sample.jpg", save=True)
📊 Results
After running inference, detected objects are saved with bounding boxes.

🚀 Future Improvements
Train on custom dataset 📈
Improve accuracy with data augmentation
Deploy as a web app 🌍
💡 Author: akum103
🎯 GitHub Repo: object_detection_YOLOv8
