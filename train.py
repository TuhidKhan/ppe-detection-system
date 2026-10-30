"""
==============================================================================
PPE Detection Project - Model Training Script
==============================================================================
This script fine-tunes a pre-trained YOLOv8 Nano model (yolov8n.pt)
on our custom PPE dataset defined in data.yaml.

PDF Requirements:
  - Base Model: yolov8n.pt
  - Epochs: 30 - 50
  - Target Accuracy: ~80% mAP
  - Output Model: models/best.pt
==============================================================================
"""

import os
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml="data.yaml",
    base_model="yolov8n.pt",
    epochs=35,
    imgsz=640,
    batch_size=16,
    project_name="runs/train",
    experiment_name="ppe_yolov8"
):
    """
    Fine-tunes the YOLOv8 model using Transfer Learning.

    Parameters:
        data_yaml (str): Path to data.yaml dataset configuration.
        base_model (str): Name or path of the base model ('yolov8n.pt').
        epochs (int): Number of complete passes over the dataset.
        imgsz (int): Resolution of images during training (default 640).
        batch_size (int): Number of images processed per batch.
        project_name (str): Directory where training logs are stored.
        experiment_name (str): Subfolder name for this training run.
    """
    print("=" * 70)
    print("STARTING YOLOv8 PPE MODEL TRAINING")
    print("=" * 70)
    print(f"Base Model:       {base_model}")
    print(f"Dataset Config:   {data_yaml}")
    print(f"Target Epochs:    {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Image Resolution: {imgsz}x{imgsz}")
    print("=" * 70)

    # 1. Load the pre-trained YOLOv8 Nano model
    # If yolov8n.pt is not present locally, Ultralytics downloads it automatically.
    print(f"\n[Step 1] Loading base model '{base_model}'...")
    model = YOLO(base_model)

    # 2. Start the training process
    print(f"\n[Step 2] Training model on PPE dataset for {epochs} epochs...")
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project_name,
        name=experiment_name,
        plots=True,          # Generates confusion matrix and loss curves
        save=True,           # Saves best.pt checkpoint
        exist_ok=True,       # Overwrite existing experiment run folder
        verbose=True
    )

    # 3. Locate the best model weights
    best_weights_path = Path(project_name) / experiment_name / "weights" / "best.pt"

    # 4. Copy best.pt directly to models/best.pt (as required by PDF Page 5)
    target_models_dir = Path("models")
    target_models_dir.mkdir(parents=True, exist_ok=True)
    target_best_pt = target_models_dir / "best.pt"

    if best_weights_path.exists():
        shutil.copy2(best_weights_path, target_best_pt)
        print("\n" + "=" * 70)
        print(f"SUCCESS: Best trained weights saved to: {target_best_pt.resolve()}")
        print("=" * 70)
    else:
        print(f"\n[Warning] Could not find weights at {best_weights_path}")

    # 5. Run validation to report final metrics (mAP, Precision, Recall)
    print("\n[Step 3] Evaluating model on 1,000 validation images...")
    val_results = model.val()

    # Extract accuracy metrics
    map50 = getattr(val_results.box, "map50", 0.0)
    map50_95 = getattr(val_results.box, "map", 0.0)
    precision = getattr(val_results.box, "mp", 0.0)
    recall = getattr(val_results.box, "mr", 0.0)

    print("\n" + "=" * 70)
    print("FINAL ACCURACY REPORT:")
    print("=" * 70)
    print(f"  mAP@50 (Target: ~80%): {map50 * 100:.2f}%")
    print(f"  mAP@50-95:             {map50_95 * 100:.2f}%")
    print(f"  Mean Precision:        {precision * 100:.2f}%")
    print(f"  Mean Recall:           {recall * 100:.2f}%")
    print("=" * 70)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on PPE Dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs (default: 35)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (default: yolov8n.pt)")

    args = parser.parse_args()

    train_model(
        data_yaml="data.yaml",
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz
    )
