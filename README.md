# 🦷 Dental Braces Detection AI

<div align="center">

[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/cloud)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge&logo=python)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS%20Supported-purple?style=for-the-badge&logo=apple)](https://developer.apple.com/metal/)

A professional AI-powered computer vision system for real-time detection and classification of orthodontic braces placement using **YOLOv8** with Apple Silicon MPS acceleration.

[Demo](#-live-demo) • [Features](#-key-features) • [Installation](#-installation) • [Deployment](#-deployment) • [Performance](#-model-performance)

</div>

---

## 🌟 Key Features

- **🎯 Real-Time Bracket Detection**: Detects individual orthodontic brackets with 26.8ms latency on Apple Silicon M4
- **🔍 Placement Classification**: 
  - 🟢 **Correct Brace** (Class 0): Properly aligned orthodontic bracket
  - 🔴 **Incorrect Brace** (Class 1): Misaligned or improperly placed bracket
- **🖥️ Interactive Streamlit Web App**:
  - Single-image and batch image upload
  - Adjustable confidence threshold controls
  - Tooth position mapping (Central Incisor, Lateral Incisor, Canine, Premolar)
  - Color-coded detection results with confidence scores
  - CSV export for clinical reporting
- **⚡ Apple Silicon MPS Hardware Acceleration**:
  - PyTorch MPS backend for 3-6x speedup over CPU
  - Optimized for M1/M2/M3/M4 chips
- **🚀 Production-Ready Deployment**:
  - Git-tracked model weights for seamless deployment
  - Streamlit Cloud compatible
  - External model hosting support via `MODEL_URL`
- **🔬 Comprehensive Training Pipeline**:
  - Resume training from checkpoints
  - Class imbalance handling
  - Early stopping with configurable patience
  - Enhanced data augmentation

---

## 📊 Model Performance

Final training results from **Epoch 40** (Apple Silicon MPS):

| Metric | Initial (Epoch 10) | Final (Epoch 40) | Improvement |
|--------|-------------------|------------------|-------------|
| **mAP@50** | 0.939 | **0.991** | **+5.2%** ⭐ |
| **mAP@50-95** | 0.607 | **0.740** | **+13.2%** ⭐⭐ |
| **Precision** | 0.859 | **0.968** | **+10.9%** ⭐ |
| **Recall** | 0.896 | **0.983** | **+8.7%** ⭐⭐ |

### Per-Class Performance Breakdown

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **Correct Brace** (Class 0) | 0.966 | 0.984 | 0.991 | 0.721 |
| **Incorrect Brace** (Class 1) | 0.964 | 0.988 | 0.990 | 0.768 |

---

## 📂 Repository Structure

```
Dental-Braces-Detection-AI/
├── README.md                              # Top-level project documentation
├── .gitignore                             # Python/OS artifacts, runs exclusion
├── teeth-braces-ai/                       # Streamlit Web Application
│   ├── streamlit_app.py                   # Streamlit UI entry point
│   ├── best.pt                            # Tracked model weights (~6.2 MB)
│   ├── requirements.txt                   # Web app dependencies
│   ├── data.yaml                          # Dataset config pointer
│   ├── .gitignore                         # Track best.pt, exclude runs/cache
│   ├── README.md                          # App-specific documentation
│   ├── tests/                             # Pytest test suite
│   └── utils/                             # Model download & tooth mapping utilities
├── braces_dataset_fixed_yolov8/           # Training Pipeline & Dataset
│   ├── data.yaml                          # Dataset YAML configuration
│   ├── train.py                           # MPS-optimized YOLOv8 training script
│   ├── detect.py                          # Single image inference CLI
│   ├── webcam_detect.py                   # Real-time webcam detection
│   ├── train/                             # 1,741 training images & labels
│   ├── valid/                             # 498 validation images & labels
│   ├── test/                              # 251 test images & labels
│   └── runs/detect/train_final/           # Final training outputs & results.csv
└── evaluation_report.csv                   # Evaluation metrics
```

---

## 🚀 Installation

### Prerequisites

- **macOS** (Apple Silicon M1/M2/M3/M4 recommended for MPS) or Linux/Windows
- **Python 3.10+** and `pip`
- **Git** for cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/sourabh-sk1/Dental-Braces-Detection-AI.git
cd Dental-Braces-Detection-AI
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r teeth-braces-ai/requirements.txt
```

### Step 4: Verify Installation

```bash
# Check MPS availability (Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Run tests
cd teeth-braces-ai
pytest tests/
```

---

## 💻 Quick Start

### Run Streamlit Web App

```bash
cd teeth-braces-ai
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your web browser.

### Run CLI Inference

```bash
# From workspace root
.venv/bin/python braces_dataset_fixed_yolov8/detect.py --image braces_dataset_fixed_yolov8/test/images/sample.jpg
```

### Run Training

```bash
# Train from scratch (Apple Silicon MPS)
.venv/bin/python braces_dataset_fixed_yolov8/train.py

# Resume from checkpoint
# Edit train.py and set: 'resume': True, 'resume_path': 'runs/detect/train_final/weights/last.pt'
```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. **Fork the repository** on GitHub
2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app" and connect your GitHub repository
   - Set main script: `teeth-braces-ai/streamlit_app.py`
   - Click "Deploy"

### Alternative: External Model Hosting

If file size limits require external hosting:

1. Upload `best.pt` to GitHub Releases, Hugging Face, or your server
2. Set `MODEL_URL` environment variable in Streamlit Cloud Settings
3. The app will auto-download weights on startup

### Docker Deployment

```bash
# Build Docker image
docker build -t dental-braces-ai .

# Run container
docker run -p 8501:8501 dental-braces-ai
```

---

## ⚙️ Configuration

### Training Configuration

- **Base Model**: `yolov8n.pt` (nano for speed)
- **Device**: `device="mps"` (Apple Silicon) or `device="cpu"`
- **Image Size**: `512` (balance of speed and accuracy)
- **Class Loss Weighting**: `cls=1.2` (addresses ~2.03:1 class imbalance)
- **Early Stopping**: `patience=10` epochs
- **Resume Capability**: Set `resume=True` in `train.py` to continue from checkpoint

### Inference Configuration

- **Image Size**: 640, 960, or 1280 (higher = better accuracy, slower)
- **Confidence Threshold**: 0.0 to 1.0 (default: 0.40 for high precision)
- **Model Weights**: Auto-detects from multiple locations

---

## 📈 Dataset Statistics

- **Total Images**: 2,490 images
- **Source Photos**: 513 unique source photographs
- **Training Set**: 1,741 images with ~6,627 correct_brace + ~3,268 incorrect_brace instances
- **Validation Set**: 498 images
- **Test Set**: 251 images
- **Class Distribution**: ~2.03:1 imbalance (correct:incorrect)

---

## ⚠️ Known Limitations

1. **Dataset Source Diversity**: The dataset contains 2,490 images derived from **513 unique source photographs**. Limited diversity in clinical scenarios.

2. **Cross-Split Augmentation Leakage**: Roboflow applied geometric augmentations prior to splitting. Consequently, 82.1% (421 of 513) of source photo bases have augmented variations distributed across train, valid, and test sets. High validation mAP reflects performance on similar clinical setups.

3. **Class Imbalance**: Training labels contain ~6,627 `correct_brace` instances vs ~3,268 `incorrect_brace` instances (~2.03:1 ratio). While loss weighting (`cls=1.2`) successfully balanced precision and recall, expanding minority class samples remains recommended for production deployment.

---

## 🧪 Testing

```bash
# Run pytest test suite
cd teeth-braces-ai
pytest tests/ -v

# Run MPS availability check
python -c "import torch; assert torch.backends.mps.is_available() == True"

# Run end-to-end inference test
python -c "
from ultralytics import YOLO
model = YOLO('teeth-braces-ai/best.pt')
results = model('braces_dataset_fixed_yolov8/test/images/sample.jpg', conf=0.40)
print(f'Detections: {len(results[0].boxes)}')
"
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent [YOLOv8](https://github.com/ultralytics/ultralytics) framework
- **Streamlit** for the amazing web app framework
- **PyTorch** team for the MPS backend support
- **Open-source dental imaging community** for dataset contributions

---

## 📞 Support

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/sourabh-sk1/Dental-Braces-Detection-AI/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/sourabh-sk1/Dental-Braces-Detection-AI/discussions)

---

<div align="center">

**Built with ❤️ for Orthodontic AI**

[⭐ Star this repo](https://github.com/sourabh-sk1/Dental-Braces-Detection-AI) • [🍴 Fork this repo](https://github.com/sourabh-sk1/Dental-Braces-Detection-AI/fork) • [📢 Share this repo](https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20AI%20project%20for%20dental%20braces%20detection!%20https://github.com/sourabh-sk1/Dental-Braces-Detection-AI)

</div>
