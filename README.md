🔍 X-Ray Dangerous Object Detection System

An end-to-end Computer Vision pipeline that detects dangerous objects inside airport baggage using deep learning models trained on X-ray scan images.

The system processes X-ray images from the OPIXray dataset and identifies weapons such as knives and scissors using state-of-the-art object detection architectures.

This project demonstrates modern computer vision and machine learning engineering concepts, including:

• Deep learning model training
• Dataset preprocessing pipelines
• CNN vs Transformer model comparison
• Object detection evaluation metrics
• Explainable AI visualization
• Interactive scanning interface
• GPU-accelerated training using PyTorch

📦 Dataset

Dataset used:

OPIXray – Airport X-ray baggage dataset

Dataset repository:

https://github.com/OPIXray-author/OPIXray

Dataset contains real airport baggage scans with dangerous objects hidden inside luggage.

Dataset Classes
Class ID	Object
0	Folding Knife
1	Straight Knife
2	Scissors
3	Utility Knife
4	Multi-tool Knife
🏗 System Architecture
OPIXray Dataset
        │
        ▼
Dataset Preparation
(Python Scripts)
        │
        ▼
Data Preprocessing
(Label conversion + split)
        │
        ▼
Model Training
 ├── YOLOv8 (CNN Detector)
 └── RT-DETR (Transformer Detector)
        │
        ▼
Model Evaluation
(Precision / Recall / mAP)
        │
        ▼
Explainable AI
(EigenCAM Visualization)
        │
        ▼
Interactive Scanner Interface
(Pygame Application)
📂 Project Structure
xray-dangerous-object-detection
│
├── main_enhanced.py
├── main.py
├── train.py
├── evaluate.py
├── prepare_dataset.py
├── eigencam_visualizer.py
│
├── model
│   ├── model_trained.pt
│   ├── model_trained2.pt
│   ├── model_exported.pt
│   ├── model_exported.onnx
│   └── classes.txt
│
├── images
│   └── test
│
├── runs
│
├── evaluation_results
│
├── requirements.txt
└── README.md
⚙️ Prerequisites

Install the following:

• Python 3.10+
• pip
• Git
• CUDA GPU (recommended)

📥 Clone the Repository
git clone https://github.com/Harjotsingh0311/xray-dangerous-object-detection.git
cd xray-dangerous-object-detection
🖥 Installation

Create virtual environment

python -m venv venv

Activate environment

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate

Install dependencies

pip install -r requirements.txt
🔄 Dataset Preparation

Download dataset from:

https://github.com/OPIXray-author/OPIXray

Place dataset in:

OPIXray/

Convert dataset into YOLO format:

python prepare_dataset.py

Output structure:

data/
 ├── images
 │   ├── train
 │   ├── val
 │   └── test
 └── labels
🧠 Train Models
Train YOLOv8
python train.py --data data/dataset.yaml --model yolov8n --epochs 50
Train RT-DETR Transformer
yolo detect train model=rtdetr-l.pt data=data/dataset.yaml epochs=50
📈 Model Evaluation

Evaluate YOLOv8

python evaluate.py --model model/model_trained.pt

Evaluate RT-DETR

python evaluate.py --model model/model_trained2.pt

Outputs generated:

evaluation_results/
 ├── metrics.csv
 ├── confusion_matrix.png
 └── sample_detections.png
📊 Example Model Performance
Model	Precision	Recall	mAP@0.5
YOLOv8	0.85	0.81	0.78
RT-DETR	0.91	0.86	0.84

Observation:

• RT-DETR achieves higher detection accuracy
• YOLOv8 provides faster inference

🔬 Explainable AI

EigenCAM visualization is used to understand which regions of the X-ray image influence model predictions.

Run visualization:

python eigencam_visualizer.py

This generates heatmaps showing model attention regions.

🖥 Run Scanner Interface

Launch interactive scanner:

python main_enhanced.py

Keyboard Controls:

Key	Action
R	Load random image
P	Load positive sample
N	Load negative sample
SPACE	Pause scanner
Q	Quit
📊 Key Insights

Model comparison shows:

• Transformer models achieve higher detection accuracy
• CNN models provide real-time detection performance

This demonstrates the trade-off between accuracy and speed in object detection systems.

🧰 Technologies Used
Technology	Purpose
Python	Core development
PyTorch	Deep learning framework
Ultralytics YOLOv8	CNN object detection
RT-DETR	Transformer detection
Pygame	Scanner interface
EigenCAM	Explainable AI
Jupyter	Exploratory analysis
🚀 Future Improvements

Possible extensions:

• Real-time CCTV baggage scanner integration
• Edge deployment using ONNX / TensorRT
• Cloud deployment on AWS / GCP
• Support for additional datasets (SIXray, PIDray)

👨‍💻 Author

Harjot Singh
B.Tech Artificial Intelligence & Machine Learning
Thapar Institute of Engineering and Technology

⭐ If you found this project useful, consider starring the repository.
