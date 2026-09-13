# 🦷 Teeth Braces Placement Detection AI — Web App

A Streamlit web application for real-time detection of orthodontic teeth braces brackets and determining if they are correctly or incorrectly placed using YOLOv8.

---

## 📋 Features

- **Single Image Detection**: Upload a dental photo for instant bounding box detection and placement classification.
- **Batch Processing**: Process multiple dental images simultaneously.
- **Tooth Position Mapping**: Position-based tooth identification (Central Incisor, Lateral Incisor, Canine, Premolars).
- **Color-Coded Classification**:
  - 🟢 **Correct Brace** (Class 0): Properly aligned orthodontic bracket.
  - 🔴 **Incorrect Brace** (Class 1): Misaligned or improperly placed bracket.
- **Confidence Threshold Slider**: Interactively filter detections.
- **CSV Result Export**: Download detection metrics and bracket status as CSV.

---

## 📊 Final Model Performance

Metrics from the completed training run (Epoch 40):

|| Metric | Value |
|| :--- | :---: |
|| **mAP@50** | **`0.991`** |
|| **mAP@50-95** | **`0.740`** |
|| **Precision** | **`0.968`** |
|| **Recall** | **`0.983`** |

---

## 🛠️ Local Development

### 1. Installation

```bash
cd teeth-braces-ai
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run streamlit_app.py
```

Access the app at `http://localhost:8501`.

### Model Weights

The app automatically loads model weights from the following locations (in order of priority):
1. `best.pt` (local weights in `teeth-braces-ai/`)
2. `../braces_dataset_fixed_yolov8/runs/detect/train_final/weights/best.pt` (final training weights)
3. `../braces_dataset_fixed_yolov8/runs/detect/train/weights/best.pt` (previous training weights)

---

## 🚀 Streamlit Cloud Deployment

### Option 1: Git-Tracked Weights (Recommended)

1. Fork this repository on GitHub.
2. Connect your repository on [Streamlit Cloud](https://streamlit.io/cloud) with main script set to `teeth-braces-ai/streamlit_app.py`.
3. Model weights (`teeth-braces-ai/best.pt`) are tracked in Git and will be automatically available.

### Option 2: External Model Hosting

If Streamlit Cloud has file size limits or you prefer external hosting:

1. Upload `best.pt` to GitHub Releases, Hugging Face, or your own server.
2. Set the `MODEL_URL` environment variable in Streamlit Cloud Settings pointing to the download URL.
3. The app will automatically download `best.pt` on startup if not found locally.

---

## 🔧 Configuration

### Model Weights Path

You can specify a custom model weights path in the app sidebar:
- Default: `best.pt`
- Alternative: Path to any YOLOv8 `.pt` weights file

### Inference Settings

- **Image Size**: 640, 960, or 1280 (higher = better accuracy but slower)
- **Confidence Threshold**: 0.0 to 1.0 (default: 0.40 for high precision)

---

## 📄 License

MIT License
