"""
YOLOv8 Training Script for Teeth Braces Placement Detection
=============================================================
This script trains a YOLOv8 model to detect whether orthodontic 
teeth braces brackets are correctly or incorrectly placed.

The model distinguishes between:
- Class 0: correct_brace - Properly positioned brackets
- Class 1: incorrect_brace - Misaligned or improperly positioned brackets

Usage:
    python train.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import YOLOv8 from Ultralytics
from ultralytics import YOLO


def train_model(
    data_yaml: str = 'data.yaml',
    model_name: str = 'yolov8s.pt',
    epochs: int = 150,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = 'runs/detect',
    name: str = 'train',
    exist_ok: bool = False,
    device: str = 'cpu',
    patience: int = 50,
    save: bool = True,
    plots: bool = True,
    resume: bool = False,
    resume_path: str = None,
    lr0: float = 0.0001,
    lrf: float = 0.001,
    optimizer: str = 'AdamW',
    weight_decay: float = 0.0005,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    degrees: float = 15.0,
    translate: float = 0.15,
    scale: float = 0.6,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.15,
    copy_paste: float = 0.1,
    amp: bool = True,
):
    """
    Train YOLOv8 model for braces detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        model_name: Name of pretrained YOLOv8 model (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        imgsz: Input image size for training
        batch: Batch size for training
        project: Directory to save training results
        name: Name of the training run
        exist_ok: Whether to overwrite existing results
        device: Device to use for training ('cpu', '0', '0,1,2,3', 'mps')
        patience: Early stopping patience (epochs without improvement)
        save: Whether to save trained model checkpoints
        plots: Whether to generate training plots
        resume: Whether to resume training from a checkpoint
        resume_path: Path to checkpoint file (e.g., 'runs/detect/train/weights/last.pt')
        
    Returns:
        Trained YOLO model
    """
    print("=" * 60)
    print("Teeth Braces Placement Detection - YOLOv8 Training")
    print("=" * 60)
    
    # Load or resume YOLOv8 model
    if resume and resume_path:
        print(f"\nResuming training from checkpoint: {resume_path}")
        model = YOLO(resume_path)
    else:
        # Load pretrained YOLOv8 model
        # Using yolov8n.pt (nano) - smallest and fastest version
        # Other options: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
        print(f"\nLoading pretrained model: {model_name}")
        model = YOLO(model_name)
    
    # Training configuration
    # These parameters are optimized for braces detection - HIGHER ACCURACY
    results = model.train(
        data=data_yaml,           # Dataset configuration file
        epochs=epochs,            # Number of training epochs
        imgsz=imgsz,              # Input image size (1280 for better accuracy)
        batch=batch,              # Batch size
        project=project,          # Save directory
        name=name,                # Experiment name
        exist_ok=exist_ok,        # Overwrite existing
        device=device,            # Computing device
        patience=patience,        # Early stopping
        save=save,                # Save checkpoints
        plots=plots,              # Generate plots
        verbose=True,              # Detailed output
        # Save directory fix - prevent nested directories
        save_dir=f"{project}/{name}",
        # Learning rate settings - optimized for better convergence
        lr0=lr0,                 # Initial learning rate
        lrf=lrf,                 # Final learning rate factor
        # Optimizer settings
        optimizer=optimizer,     # AdamW optimizer
        weight_decay=weight_decay, # L2 regularization
        # Loss function weights - balanced for classification
        box=box,                 # Box loss weight
        cls=cls,                 # Classification loss weight
        dfl=dfl,                 # DFL loss weight
        # Data augmentation settings - enhanced for better generalization
        degrees=degrees,         # Random rotation
        translate=translate,     # Random translation
        scale=scale,             # Random scaling
        flipud=flipud,           # No vertical flip
        fliplr=fliplr,           # Horizontal flip
        mosaic=mosaic,           # Mosaic augmentation
        mixup=mixup,             # Mixup augmentation
        copy_paste=copy_paste,   # Copy-paste augmentation
        # Training stability
        amp=amp,                 # Automatic mixed precision
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved to: runs/detect/{name}/weights/best.pt")
    print(f"Last model saved to: runs/detect/{name}/weights/last.pt")
    
    return model


def validate_model(weights_path: str = 'runs/detect/train/weights/best.pt'):
    """
    Validate trained model on validation set.
    
    Args:
        weights_path: Path to trained model weights
    """
    print("\n" + "=" * 60)
    print("Running Validation")
    print("=" * 60)
    
    # Load trained model
    model = YOLO(weights_path)
    
    # Run validation
    metrics = model.val()
    
    # Print metrics
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(weights_path: str = 'runs/detect/train/weights/best.pt', format: str = 'onnx'):
    """
    Export trained model to different formats.
    
    Args:
        weights_path: Path to trained model weights
        format: Export format (onnx, torchscript, coreml, etc.)
    """
    print(f"\nExporting model to {format} format...")
    
    model = YOLO(weights_path)
    exported_path = model.export(format=format)
    
    print(f"Model exported to: {exported_path}")
    
    return exported_path


if __name__ == "__main__":
    import torch
    default_device = 'mps' if torch.backends.mps.is_available() else ('0' if torch.cuda.is_available() else 'cpu')
    
    # Training configuration - Optimized for Apple Silicon MPS & Class Imbalance
    CONFIG = {
        'data_yaml': 'data.yaml',
        'model_name': 'yolov8n.pt',     # Base nano model for realistic training time on Apple Silicon
        'epochs': 40,                   # Total epoch target (can resume from checkpoint)
        'imgsz': 512,                  # 512x512 resolution (1.5-2x faster than 640 with high precision)
        'batch': 16,                   # Batch size for Apple M4 memory bandwidth
        'device': default_device,      # Auto-selects 'mps' on Apple Silicon
        'patience': 10,                # Early stopping patience
        'resume': False,               # Set to True to resume from checkpoint
        'resume_path': 'runs/detect/train_final/weights/last.pt',  # Checkpoint path for resuming
        'lr0': 0.001,                 # Initial learning rate
        'lrf': 0.01,                  # Final learning rate factor
        'optimizer': 'AdamW',          # AdamW optimizer for convergence
        'weight_decay': 0.0005,       # L2 regularization
        # Loss function weights addressing ~2:1 class imbalance
        'box': 7.5,                    # Box loss weight
        'cls': 1.2,                    # Classification loss weight (boosted for minority class penalty)
        'dfl': 1.5,                   # DFL loss weight
        # Augmentation settings - enhanced for small object generalization
        'degrees': 10,               # Random rotation
        'translate': 0.1,            # Random translation
        'scale': 0.5,                # Random scaling
        'flipud': 0.0,               # No vertical flip
        'fliplr': 0.5,               # Horizontal flip
        'mosaic': 1.0,               # Mosaic augmentation
        'mixup': 0.1,                # Mixup augmentation
        'copy_paste': 0.3,           # Copy-paste augmentation (enhanced for small objects)
        # Training stability
        'close_mosaic': 10,          # Disable mosaic in last 10 epochs
        'amp': True,                 # Automatic mixed precision
        'val': True,                 # Validate during training
        'plots': True,               # Generate training plots
    }
    
    print("\nTraining Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    
    # Check if resuming from checkpoint
    if CONFIG['resume'] and CONFIG['resume_path'] and os.path.exists(CONFIG['resume_path']):
        print(f"Resuming from checkpoint: {CONFIG['resume_path']}")
        resume_config = CONFIG.copy()
        resume_config['resume'] = True
        model = train_model(**resume_config)
    else:
        # Train from scratch or with pretrained model
        model = train_model(**CONFIG)
    
    # After training, validate the model
    print("\nValidating trained model...")
    validate_model()
    
    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)
    print("\nTo run detection on an image:")
    print("  python detect.py --image path/to/image.jpg")
    print("\nTo run the web app:")
    print("  streamlit run app.py")

