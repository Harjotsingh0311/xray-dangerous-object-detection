import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "runs/detect/runs/train/opixray_20260427_002657/weights/best.pt"

# Load YOLO model
model = YOLO(MODEL_PATH)
net = model.model

# Choose a deep convolutional layer (second‑last layer is typical)
target_layer = net.model[-2]   # Adjust if needed (use -1 or print(net.model) to see)

# Wrapper that captures the target layer's output via a hook
class FeatureExtractor(nn.Module):
    def __init__(self, original_model, target_layer):
        super().__init__()
        self.original_model = original_model
        self.target_layer = target_layer
        self.feature_maps = None

        def hook(module, inp, out):
            self.feature_maps = out

        target_layer.register_forward_hook(hook)

    def forward(self, x):
        _ = self.original_model(x)   # Run YOLO, we don't need its output
        return self.feature_maps     # Return only the captured feature maps

# Create feature extractor and cam
feature_extractor = FeatureExtractor(net, target_layer)
cam = EigenCAM(model=feature_extractor, target_layers=[target_layer])

def generate_eigencam(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    # Preprocess
    img = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0).float()

    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Overlay and show
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    cv2.imshow("EigenCAM Visualization", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
generate_eigencam("images/test/pos/010821.jpg")