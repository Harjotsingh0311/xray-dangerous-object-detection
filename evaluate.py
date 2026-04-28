"""
evaluate.py
===========
Run comprehensive evaluation of a trained YOLOv8 model on the OPIXray test set.

Outputs
-------
• Per-class Precision, Recall, F1-score
• mAP@0.5 and mAP@0.5:0.95
• Confusion matrix (saved as PNG)
• Full metrics CSV
• Detection sample grid (saved as PNG)

Usage
-----
    python evaluate.py --model model/model_trained.pt --data data/dataset.yaml

Optional flags
--------------
    --conf      Confidence threshold  (default: 0.25)
    --iou       NMS IoU threshold     (default: 0.5)
    --imgsz     Inference image size  (default: 640)
    --device    cuda / cpu / 0        (default: auto)
    --samples   Number of sample detections to visualise (default: 12)
    --output    Folder for saved figures (default: evaluation_results)
"""

import argparse
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on OPIXray test set")
    parser.add_argument("--model",   default="model/model_trained.pt",
                        help="Path to trained .pt weights")
    parser.add_argument("--data",    default="data/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.50)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="")
    parser.add_argument("--samples", type=int,   default=12,
                        help="Number of sample images to visualise")
    parser.add_argument("--output",  default="evaluation_results")
    return parser.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Folding_Knife":    (255,  80,  80),
    "Straight_Knife":   (255, 180,  30),
    "Scissors":         ( 60, 200,  60),
    "Utility_Knife":    ( 80, 140, 255),
    "Multi-tool_Knife": (200,  60, 220),
}
DEFAULT_COLOR = (200, 200, 200)


def color_for_class(name):
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


def draw_detections_on_image(img_bgr, boxes, class_names):
    """Draw bounding boxes on a BGR numpy image (OpenCV style)."""
    import cv2
    img = img_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls_i = int(box.cls[0])
        name  = class_names.get(cls_i, f"cls{cls_i}")
        r, g, b = color_for_class(name)
        bgr = (b, g, r)
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)
        label = f"{name}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def save_confusion_matrix(cm_array, class_names, output_path):
    """Save a normalised confusion matrix as a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm_array.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix (normalised)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {output_path}")


def save_metrics_table(results_dict, per_class, class_names, output_path):
    """Write a human-readable metrics CSV."""
    lines = ["metric,value\n"]
    for k, v in results_dict.items():
        lines.append(f"{k},{v:.6f}\n")
    lines.append("\nclass,precision,recall,f1,ap50\n")
    for i, name in enumerate(class_names):
        p  = per_class["precision"][i]
        r  = per_class["recall"][i]
        f1 = 2 * p * r / (p + r + 1e-9)
        ap = per_class["ap50"][i]
        lines.append(f"{name},{p:.4f},{r:.4f},{f1:.4f},{ap:.4f}\n")

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"  Metrics CSV       → {output_path}")


def save_sample_detections(model, image_paths, class_names, conf, output_path, n=12):
    """Save a grid of sample detections."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_paths = random.sample(image_paths, min(n, len(image_paths)))
    cols = 4
    rows = (len(sample_paths) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for i, img_path in enumerate(sample_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        results = model(img_bgr, conf=conf, verbose=False)
        for result in results:
            img_bgr = draw_detections_on_image(img_bgr, result.boxes, class_names)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(Path(img_path).name, fontsize=8)
        axes[i].axis("off")

    plt.suptitle("Sample Detections on Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sample detections → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("❌  ultralytics not installed.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {data_path}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))
    class_names = model.names   # {0: 'Folding_Knife', ...}

    # ── Run validation ────────────────────────────────────────────────────
    print("📊  Running validation on test split …")
    val_results = model.val(
        data   = str(data_path.resolve()),
        split  = "test",
        conf   = args.conf,
        iou    = args.iou,
        imgsz  = args.imgsz,
        device = args.device if args.device else None,
        plots  = True,
        save_json = True,
        project   = str(output_dir),
        name      = "val_run",
        verbose   = True,
    )

    results_dict = val_results.results_dict

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  📋  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    metric_labels = {
        "metrics/precision(B)": "Precision (all classes)",
        "metrics/recall(B)":    "Recall    (all classes)",
        "metrics/mAP50(B)":     "mAP @ 0.5",
        "metrics/mAP50-95(B)":  "mAP @ 0.5:0.95",
        "fitness":              "Fitness",
    }
    for key, label in metric_labels.items():
        val = results_dict.get(key, float("nan"))
        print(f"  {label:<30} {val:>10.4f}")

    # ── Per-class metrics ─────────────────────────────────────────────────
    print(f"\n  {'Class':<22} {'P':>7} {'R':>7} {'F1':>7} {'AP@0.5':>8}")
    print(f"  {'-'*55}")

    per_class = {"precision": [], "recall": [], "ap50": []}
    try:
        # Ultralytics stores per-class results in val_results.box
        box = val_results.box
        for i in range(len(class_names)):
            p  = float(box.p[i])  if hasattr(box, "p")  else 0.0
            r  = float(box.r[i])  if hasattr(box, "r")  else 0.0
            ap = float(box.ap50[i]) if hasattr(box, "ap50") else 0.0
            per_class["precision"].append(p)
            per_class["recall"].append(r)
            per_class["ap50"].append(ap)
            f1 = 2 * p * r / (p + r + 1e-9)
            print(f"  {class_names[i]:<22} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {ap:>8.4f}")
    except Exception as e:
        print(f"  [WARN] Could not extract per-class metrics: {e}")
        # Fill with zeros so downstream code works
        for i in range(len(class_names)):
            per_class["precision"].append(0.0)
            per_class["recall"].append(0.0)
            per_class["ap50"].append(0.0)

    print(f"{'='*60}\n")

    # ── Save artefacts ────────────────────────────────────────────────────
    save_metrics_table(
        results_dict, per_class, list(class_names.values()),
        output_dir / "metrics.csv"
    )

    # Confusion matrix (Ultralytics already saves one, but we make our own)
    try:
        cm = val_results.confusion_matrix.matrix
        save_confusion_matrix(
            cm, list(class_names.values()) + ["background"],
            output_dir / "confusion_matrix.png"
        )
    except Exception as e:
        print(f"  [WARN] Could not save confusion matrix: {e}")

    # Sample detections
    import yaml
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", data_path.parent))
    test_img_dir = dataset_root / data_cfg.get("test", "images/test")

    if test_img_dir.exists():
        test_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
        if test_images:
            save_sample_detections(
                model, test_images, class_names, args.conf,
                output_dir / "sample_detections.png",
                n=args.samples
            )
    else:
        print(f"  [WARN] Test image folder not found: {test_img_dir}")

    print(f"\n✅  All evaluation results saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
