🚀 Object Detection using YOLOv8

📌 Overview

This project implements YOLOv8 (You Only Look Once v8) for object detection using the VOC2012 dataset. YOLOv8 is a state-of-the-art object detection model that enables real-time detection and classification of multiple objects in an image. This project provides a step-by-step guide to setting up the environment, training, and evaluating the model.

📂 Dataset

Dataset Name: PASCAL VOC 2012

Dataset Type: Object Detection

Location: Stored in Google Drive

Number of Images: 📸 (Update after running the count in your notebook)

Classes:

aeroplane

bicycle

bird

boat

bottle

bus

car

cat

chair

cow

dining table

dog

horse

motorbike

person

potted plant

sheep

sofa

train

TV monitor

🛠️ Installation

1️⃣ Clone the Repository

git clone https://github.com/akum103/object_detection_YOLOv8.git
cd object_detection_YOLOv8

2️⃣ Install Required Packages

Install dependencies using pip:

pip install ultralytics opencv-python-headless matplotlib numpy pandas

3️⃣ Mount Google Drive (for dataset access in Colab)

from google.colab import drive
drive.mount('/content/drive')

🏗️ Model Setup

1️⃣ Load the YOLOv8 Model

Using the pre-trained YOLOv8 model:

from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'n' refers to Nano version; change to 's', 'm', 'l' for larger models

2️⃣ Running Object Detection

# Path to the image for inference
sample_image = "/path/to/sample.jpg"

# Run inference on the image
results = model(sample_image, save=True)

3️⃣ Training YOLOv8 on Custom Dataset

If you want to train YOLOv8 on a custom dataset, follow these steps:

# Train YOLOv8 model on custom dataset for 50 epochs
model.train(data="/path/to/dataset.yaml", epochs=50, imgsz=640)

4️⃣ Evaluate Model Performance

# Evaluate model on validation dataset
eval_results = model.val()
print(eval_results)

📊 Results

Once inference is run, the detected objects will be saved with bounding boxes. The output images with annotations will be stored in the runs/detect/predict/ directory.

📈 Performance Metrics

Precision: Measures how many of the detected objects are correct.

Recall: Measures how well the model detects actual objects.

mAP (Mean Average Precision): Measures overall model accuracy.

# Get model performance metrics
eval_results = model.val()
print(f"mAP: {eval_results['mAP']}")

🚀 Future Improvements

Train on custom datasets to improve accuracy 📈

Apply data augmentation techniques for better generalization

Optimize the model for edge devices for real-time deployment 🖥️

Develop a web application to interact with the model 🌍

💡 Author: akum103🎯 GitHub Repo: object_detection_YOLOv8
