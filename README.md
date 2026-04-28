  ------------------
  X-Ray Dangerous Object Detection
  Airport Baggage Scanner using YOLOv8 and RT-DETR
  ------------------
  
Author: Harjot Singh B.Tech – Artificial Intelligence & Machine Learning
Thapar Institute of Engineering and Technology

  ------------------
  PROJECT OVERVIEW
  ------------------

Airport security relies heavily on X‑ray baggage scanners to detect
dangerous objects inside luggage. Manual inspection is slow and prone to
human error.

This project builds an AI‑based X‑ray baggage inspection system capable
of detecting dangerous objects automatically using deep learning.

The system compares two modern detection architectures:

YOLOv8 – Fast CNN‑based object detector RT‑DETR – Transformer‑based
object detector with higher accuracy

Both models are trained on the OPIXray dataset.

  ------------
  OBJECTIVES
  ------------

• Detect dangerous objects in X‑ray baggage images • Train deep learning
models using the OPIXray dataset • Compare CNN vs Transformer detection
models • Evaluate performance using detection metrics • Build an
interactive scanning interface • Provide explainability using EigenCAM

  -------------
  MODELS USED
  -------------

1)  YOLOv8 (CNN Detector)

Advantages: - Very fast inference - Lightweight architecture - Suitable
for real‑time scanning systems

Model file: model/model_trained.pt

2)  RT-DETR (Transformer Detector)

Advantages: - Higher detection accuracy - Better detection under
occlusion - Stronger feature understanding

Model file: model/model_trained2.pt

  ------------------
  DETECTED CLASSES
  ------------------

0 Folding Knife 1 Straight Knife 2 Scissors 3 Utility Knife 4 Multi‑tool
Knife

  -------------------
  PROJECT STRUCTURE
  -------------------

xray-dangerous-object-detection

main_enhanced.py main.py train.py evaluate.py prepare_dataset.py
eigencam_visualizer.py

model/ model_trained.pt model_trained2.pt model_exported.pt
model_exported.onnx classes.txt

images/ runs/ evaluation_results/

requirements.txt README.md

  -------------
  QUICK START
  -------------

1)  Clone repository

git clone
https://github.com/Harjotsingh0311/xray-dangerous-object-detection.git
cd xray-dangerous-object-detection

2)  Create virtual environment

python -m venv venv

Activate environment

Windows: venv

Linux/Mac: source venv/bin/activate

3)  Install dependencies

pip install -r requirements.txt

4)  Run scanner

python main_enhanced.py

Scanner Controls:

R Load random baggage image P Load positive image N Load negative image
SPACE Pause scanner Q Quit application

  ---------
  DATASET
  ---------

Dataset used: OPIXray

Download dataset from: https://github.com/OPIXray-author/OPIXray

After downloading:

python prepare_dataset.py

  ----------
  TRAINING
  ----------

Train YOLOv8

python train.py –data data/dataset.yaml –model yolov8n –epochs 50

Train RT-DETR

yolo detect train model=rtdetr-l.pt data=data/dataset.yaml epochs=50

  ------------
  EVALUATION
  ------------

Evaluate YOLOv8

python evaluate.py –model model/model_trained.pt

Evaluate RT-DETR

python evaluate.py –model model/model_trained2.pt

Results will be saved in:

evaluation_results/

  ----------------
  EXPLAINABLE AI
  ----------------

EigenCAM is used to visualize which regions of the X‑ray image influence
model predictions.

Run:

python eigencam_visualizer.py

  -----
  END
  -----
