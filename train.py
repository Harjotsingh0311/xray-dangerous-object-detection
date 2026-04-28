"""
train.py
========
Train a YOLOv8 model on the OPIXray dataset (5 dangerous-object classes).

Features
--------
• YOLOv8m backbone (good accuracy / speed trade-off)
• Loads pre-trained ImageNet weights automatically via Ultralytics
• X-ray–friendly augmentations (no hue shift, gentle colour jitter)
• Cosine LR schedule + early stopping
• Saves best.pt and last.pt, plus a full metrics CSV
• Prints mAP@0.5 and mAP@0.5:0.95 after training

Usage
-----
    python train.py --data data/dataset.yaml

Optional flags
--------------
    --model        YOLOv8 variant: yolov8n/s/m/l/x  (default: yolov8m)
    --epochs       Max training epochs               (default: 100)
    --imgsz        Input image size                  (default: 640)
    --batch        Batch size                        (default: 16)
    --workers      DataLoader workers                (default: 4)
    --device       cuda / cpu / 0 / 0,1 …            (default: auto)
    --patience     Early-stop patience               (default: 20)
    --project      Output folder                     (default: runs/train)
    --name         Experiment name                   (default: opixray)
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on OPIXray")
    parser.add_argument("--data",     default="data/dataset.yaml",
                        help="Path to dataset.yaml (created by prepare_dataset.py)")
    parser.add_argument("--model",    default="yolov8m",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        help="YOLOv8 backbone variant")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--workers",  type=int,   default=4)
    parser.add_argument("--device",   default="",
                        help="Training device: '' = auto, '0' = GPU 0, 'cpu'")
    parser.add_argument("--patience", type=int,   default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--project",  default="runs/train")
    parser.add_argument("--name",     default="opixray")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("❌  ultralytics not installed. Run:  pip install ultralytics")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Run prepare_dataset.py first, or adjust --data."
        )

    # ── Model ──────────────────────────────────────────────────────────────
    if args.resume:
        resume_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint to resume from: {resume_ckpt}")
        print(f"🔄  Resuming from {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
    else:
        model_weights = f"{args.model}.pt"   # downloads pretrained weights if needed
        print(f"🚀  Starting fresh training with {model_weights}")
        model = YOLO(model_weights)

    # ── X-ray–tuned augmentation hyper-params ──────────────────────────────
    # X-ray images are greyscale-ish; avoid aggressive colour distortions.
    xray_aug = dict(
        hsv_h    = 0.0,    # no hue shift (x-ray is pseudo-colour)
        hsv_s    = 0.2,    # mild saturation
        hsv_v    = 0.3,    # moderate brightness
        degrees  = 5.0,    # slight rotation
        translate= 0.1,
        scale    = 0.4,
        shear    = 2.0,
        flipud   = 0.3,    # vertical flip sometimes realistic
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.1,
        copy_paste = 0.1,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{args.name}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"  Dataset  : {data_path.resolve()}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}  (patience={args.patience})")
    print(f"  Image sz : {args.imgsz}px")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device or 'auto'}")
    print(f"  Output   : {args.project}/{run_name}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    results = model.train(
        data      = str(data_path.resolve()),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,
        device    = args.device if args.device else None,
        patience  = args.patience,
        project   = args.project,
        name      = run_name,
        exist_ok  = True,
        pretrained= True,
        optimizer = "AdamW",
        lr0       = 0.001,
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        cos_lr    = True,           # cosine LR annealing
        label_smoothing = 0.05,
        val       = True,
        save      = True,
        save_period = 10,           # save checkpoint every N epochs
        plots     = True,
        verbose   = True,
        **xray_aug
    )

    # ── Results summary ───────────────────────────────────────────────────
    best_weights = Path(args.project) / run_name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print("  ✅  Training complete!")
    print(f"  Best weights : {best_weights}")

    # Extract key metrics from the results object
    try:
        metrics = results.results_dict
        print(f"\n  📊  Final validation metrics:")
        print(f"      Precision (P)   : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"      Recall    (R)   : {metrics.get('metrics/recall(B)',    0):.4f}")
        print(f"      mAP@0.5         : {metrics.get('metrics/mAP50(B)',     0):.4f}")
        print(f"      mAP@0.5:0.95    : {metrics.get('metrics/mAP50-95(B)',  0):.4f}")
    except Exception:
        pass

    print(f"\n  Next step: python evaluate.py --model {best_weights} --data {data_path}")
    print(f"{'='*60}\n")

    # ── Copy best model to ./model/ ───────────────────────────────────────
    dest = Path("model") / "model_trained.pt"
    dest.parent.mkdir(exist_ok=True)
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"  📦  Best model also saved to {dest}")


if __name__ == "__main__":
    main()
