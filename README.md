# Dental Braces Detection AI

<div align="center">

[![Live App](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://dental-braces-ai-v2-nq9c8flmqzu8oeb2jeqytz.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/sourabh-sk1/Dental-Braces-AI-V2)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00599C?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An enterprise-grade computer vision solution for automated detection and placement accuracy assessment of orthodontic braces using YOLOv8 deep neural networks.

[Live Demo](https://dental-braces-ai-v2-nq9c8flmqzu8oeb2jeqytz.streamlit.app/) | [Repository](https://github.com/sourabh-sk1/Dental-Braces-AI-V2) | [System Architecture](#system-architecture) | [Model Performance](#model-performance) | [Deployment](#deployment-guide)

</div>

---

## Executive Summary

Dental Braces Detection AI is a specialized deep-learning application designed to assist dental professionals and researchers in identifying orthodontic brackets and evaluating their positioning accuracy. Powered by an optimized YOLOv8 neural network architecture, the system provides automated detection of orthodontic hardware, categorizing brackets into correct or incorrect placements with high precision and sub-30ms inference latency.

### Live System Access
- **Web Application**: [https://dental-braces-ai-v2-nq9c8flmqzu8oeb2jeqytz.streamlit.app/](https://dental-braces-ai-v2-nq9c8flmqzu8oeb2jeqytz.streamlit.app/)
- **Source Code Repository**: [https://github.com/sourabh-sk1/Dental-Braces-AI-V2](https://github.com/sourabh-sk1/Dental-Braces-AI-V2)

---

## System Capabilities

- **Automated Bracket Detection**: High-resolution localized detection of individual orthodontic brackets across maxillary and mandibular dental arches.
- **Placement Assessment**: Real-time classification of bracket positioning into two distinct categories:
  - **Correct Brace (Class 0)**: Properly aligned bracket placement according to clinical standards.
  - **Incorrect Brace (Class 1)**: Misaligned, rotated, or improperly positioned bracket placement.
- **Anatomical Tooth Mapping**: Positional mapping engine supporting Central Incisor (CI), Lateral Incisor (LI), Canine (C), First Premolar (P1), and Second Premolar (P2) identification.
- **Interactive Streamlit Web Dashboard**:
  - Single-image and batch image evaluation workflows.
  - Configurable confidence thresholding (0.00 to 1.00).
  - Dynamic visual overlays with color-coded bounding boxes.
  - Structured CSV export for diagnostic reporting.
- **Cross-Platform Hardware Optimization**:
  - Apple Silicon Metal Performance Shaders (MPS) acceleration support.
  - CPU and CUDA GPU compatibility.
- **Production-Ready Web Service**: Headless OpenCV compatibility, automated fallback weight downloaders, and unit testing suite.

---

## Model Performance

The deep learning model was evaluated on a held-out test split of 251 clinical dental images. The evaluation results demonstrate high object localization accuracy and class discrimination capabilities.

### Global Evaluation Metrics

| Metric | Score | Definition |
| :--- | :---: | :--- |
| **mAP@50** | **0.990** | Mean Average Precision at 0.50 IoU threshold |
| **mAP@50-95** | **0.745** | Mean Average Precision averaged across 0.50 to 0.95 IoU thresholds |
| **Precision** | **0.965** | Ratio of true positive bracket detections over total positive predictions |
| **Recall** | **0.986** | Ratio of true positive bracket detections over ground truth annotations |

### Class-Specific Breakdown

| Class Name | Target Category | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Correct Brace** | Properly placed bracket | 0.966 | 0.984 | 0.991 | 0.721 |
| **Incorrect Brace** | Misaligned placement | 0.964 | 0.988 | 0.990 | 0.768 |

---

## Dataset Characteristics

- **Total Annotated Images**: 2,490 high-resolution dental photographs.
- **Unique Base Photographs**: 513 distinct clinical source photos.
- **Data Splits**:
  - **Training Set**: 1,741 images (6,627 correct brace annotations, 3,268 incorrect brace annotations).
  - **Validation Set**: 498 images.
  - **Test Set**: 251 images.
- **Class Balance Mitigation**: Loss weight balancing (`cls=1.2`) and mosaic data augmentations applied during training to handle class distribution ratios.

---

## System Architecture

```
Dental-Braces-Detection-AI/
├── README.md                              # Main system documentation
├── requirements.txt                       # Production python dependencies
├── packages.txt                           # Linux system dependencies (libgl1, libglib)
├── best.pt                            # Fused model weights (~6.2 MB)
├── teeth-braces-ai/                       # Streamlit Application Service
│   ├── streamlit_app.py                   # Main web dashboard interface
│   ├── best.pt                            # Tracked production model weights
│   ├── detect.py                          # Independent inference module
│   ├── evaluate.py                        # Model validation engine
│   ├── requirements.txt                   # App dependency definitions
│   ├── packages.txt                       # Linux container system packages
│   ├── tests/                             # Pytest automated test suite
│   └── utils/                             # Tooth mapping & weight utilities
├── braces_dataset_fixed_yolov8/           # Dataset & Training Pipeline
│   ├── data.yaml                          # Dataset specification file
│   ├── train.py                           # Training pipeline script
│   ├── detect.py                          # CLI inference interface
│   ├── webcam_detect.py                   # Real-time webcam feed module
│   ├── train/                             # Training dataset split
│   ├── valid/                             # Validation dataset split
│   └── test/                              # Testing dataset split
├── evaluation_report.csv                  # Quantitative evaluation summary
├── confusion_matrix.png                   # Confusion matrix visualization
└── metrics.png                            # Metric progression charts
```

---

## Installation & Setup

### Prerequisites

- **Operating System**: macOS (Apple Silicon or Intel), Linux (Debian/Ubuntu), or Windows 10/11.
- **Python**: Version 3.10 or higher.
- **Package Manager**: `pip` and `virtualenv`.

### Step 1: Clone Repository

```bash
git clone https://github.com/sourabh-sk1/Dental-Braces-AI-V2.git
cd Dental-Braces-AI-V2
```

### Step 2: Configure Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Execution Guide

### Launch Web Dashboard Locally

To start the interactive Streamlit dashboard:

```bash
streamlit run teeth-braces-ai/streamlit_app.py
```

Access the application in your browser at `http://localhost:8501`.

### Command Line Inference

To run object detection on a single image file via CLI:

```bash
python teeth-braces-ai/detect.py --image path/to/dental_image.jpg --conf 0.40
```

### Automated Unit Verification

To run the automated test suite:

```bash
PYTHONPATH=teeth-braces-ai pytest teeth-braces-ai/tests/
```

---

## Deployment Guide

### Streamlit Community Cloud

1. Push or fork the repository to GitHub: `https://github.com/sourabh-sk1/Dental-Braces-AI-V2`.
2. Navigate to [Streamlit Cloud](https://share.streamlit.io/).
3. Click **New App** and configure the following parameters:
   - **Repository**: `sourabh-sk1/Dental-Braces-AI-V2`
   - **Branch**: `main`
   - **Main file path**: `teeth-braces-ai/streamlit_app.py`
4. Deploy the application.

*Note: The repository includes `packages.txt` (`libgl1`, `libglib2.0-0`) and `opencv-python-headless` to ensure seamless Linux container initialization.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for complete details.

---

## Citation & Acknowledgments

- **Ultralytics YOLOv8**: Real-time object detection framework.
- **Streamlit**: Application framework for machine learning workflows.
- **PyTorch**: Deep learning backend framework.
