# 🔍 X-Ray Dangerous Object Detection

### Airport Baggage Scanner using YOLOv8 and RT-DETR Transformer

## 👨‍💻 Author

**Harjot Singh**\
B.Tech -- Artificial Intelligence & Machine Learning\
**Thapar Institute of Engineering and Technology**

------------------------------------------------------------------------

# 📌 Project Overview

Airport security relies heavily on X-ray baggage scanners to detect
dangerous objects inside luggage. Manual inspection is time-consuming
and prone to human error.

This project builds an **AI-powered X-ray baggage scanning system**
capable of detecting dangerous objects automatically using deep
learning.

The system compares two modern object detection architectures:

• **YOLOv8 (CNN based detector)** -- optimized for real-time detection\
• **RT-DETR (Transformer based detector)** -- optimized for higher
accuracy

The models are trained on the **OPIXray dataset**, which contains real
airport X-ray baggage images.

The project also includes an **interactive scanner interface**
simulating an airport security inspection system.

------------------------------------------------------------------------

# 🎯 Objectives

The main objectives of this project were:

-   Detect dangerous objects in X-ray baggage images
-   Train deep learning models on the **OPIXray dataset**
-   Compare **CNN vs Transformer detection models**
-   Evaluate performance using standard detection metrics
-   Build an **interactive scanning system**
-   Provide **model explainability using EigenCAM**

------------------------------------------------------------------------

# 🧠 Models Implemented

## 1️⃣ YOLOv8 (CNN Detector)

YOLOv8 is a convolutional neural network designed for **fast real-time
object detection**.

Advantages:

-   Very fast inference
-   Lightweight architecture
-   Suitable for real-time scanning systems

Model file:

model/model_trained.pt

------------------------------------------------------------------------

## 2️⃣ RT-DETR (Transformer Detector)

RT-DETR is a **Transformer-based object detector** using attention
mechanisms to capture global image context.

Advantages:

-   Higher detection accuracy
-   Better detection under occlusion
-   Stronger feature understanding

Model file:

model/model_trained2.pt

------------------------------------------------------------------------

# 🏷️ Detected Object Classes

The system detects **five dangerous object categories**.

  ID   Class
  ---- ------------------
  0    Folding Knife
  1    Straight Knife
  2    Scissors
  3    Utility Knife
  4    Multi-tool Knife

------------------------------------------------------------------------

# 📊 Model Performance

  Model     Precision   Recall   mAP@0.5   Speed
  --------- ----------- -------- --------- ----------
  YOLOv8    0.85        0.81     0.78      Fast
  RT-DETR   0.91        0.86     0.84      Moderate

Observation:

-   RT-DETR provides higher detection accuracy
-   YOLOv8 provides much faster inference

------------------------------------------------------------------------

# 📁 Project Structure

xray-dangerous-object-detection │ ├── main_enhanced.py \# Interactive
X-ray scanner interface ├── main.py \# Basic scanner interface ├──
train.py \# Model training script ├── evaluate.py \# Model evaluation
script ├── prepare_dataset.py \# Dataset conversion script ├──
eigencam_visualizer.py \# Explainable AI visualization │ ├── model │ ├──
model_trained.pt │ ├── model_trained2.pt │ ├── model_exported.pt │ ├──
model_exported.onnx │ └── classes.txt │ ├── images ├── data ├── runs ├──
evaluation_results │ ├── requirements.txt └── README.md

------------------------------------------------------------------------

# ⚡ Quick Start

### Clone repository

git clone
https://github.com/YOUR_USERNAME/xray-dangerous-object-detection.git

cd xray-dangerous-object-detection

------------------------------------------------------------------------

### Create virtual environment

python -m venv venv

Activate:

Windows

venv`\Scripts`{=tex}`\activate`{=tex}

Linux / Mac

source venv/bin/activate

------------------------------------------------------------------------

### Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

### Run scanner

python main_enhanced.py

Controls:

  Key     Action
  ------- ---------------------------
  R       Load random baggage image
  P       Load positive image
  N       Load negative image
  SPACE   Pause scanner
  Q       Quit

------------------------------------------------------------------------

# 📦 Dataset

The project uses the **OPIXray dataset**.

Dataset repository:

https://github.com/OPIXray-author/OPIXray

After downloading the dataset:

python prepare_dataset.py

------------------------------------------------------------------------

# 🏋️ Training

Train YOLOv8:

python train.py --data data/dataset.yaml --model yolov8n --epochs 50

Train RT-DETR:

yolo detect train model=rtdetr-l.pt data=data/dataset.yaml epochs=50

------------------------------------------------------------------------

# 📈 Evaluation

Evaluate YOLOv8:

python evaluate.py --model model/model_trained.pt

Evaluate RT-DETR:

python evaluate.py --model model/model_trained2.pt

Results are saved in:

evaluation_results/

------------------------------------------------------------------------

# 🔬 Explainable AI

EigenCAM visualization helps understand which regions influence the
model predictions.

Run:

python eigencam_visualizer.py

------------------------------------------------------------------------

# ⭐ If you found this project useful

Please consider giving the repository a star.
