# Teeth Braces Placement Detection AI - Web Dashboard

A Streamlit web application for real-time detection of orthodontic braces brackets and placement accuracy assessment using YOLOv8 deep neural networks.

[Live Application Demo](https://dental-braces-ai-v2-6wnplzjxjpdx5bb4ojxjts.streamlit.app/) | [Source Code Repository](https://github.com/sourabh-sk1/Dental-Braces-AI-V2)

---

## Capabilities

- **Single Image Evaluation**: Upload a dental image for real-time localized detection and placement classification.
- **Batch Processing**: Simultaneous evaluation of multiple dental images.
- **Anatomical Tooth Mapping**: Positional mapping engine supporting Central Incisor (CI), Lateral Incisor (LI), Canine (C), First Premolar (P1), and Second Premolar (P2) identification.
- **Classification Categories**:
  - **Correct Brace (Class 0)**: Properly aligned bracket placement.
  - **Incorrect Brace (Class 1)**: Misaligned or improperly placed bracket.
- **Interactive Threshold Controls**: Real-time slider for confidence threshold adjustment.
- **Diagnostic Export**: Structured CSV reporting of bracket coordinates, placement status, and confidence scores.

---

## Quantitative Evaluation Metrics

Metrics from the evaluated model test set:

| Metric | Score |
| :--- | :---: |
| **mAP@50** | **0.990** |
| **mAP@50-95** | **0.745** |
| **Precision** | **0.965** |
| **Recall** | **0.986** |

---

## Local Environment Setup

### 1. Installation

```bash
cd teeth-braces-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Access the dashboard at `http://localhost:8501`.

### Model Weights Resolution

The application automatically resolves model weights from the following paths (in order of priority):
1. `best.pt` (local weights in `teeth-braces-ai/` or root directory)
2. `../braces_dataset_fixed_yolov8/runs/detect/train_final/weights/best.pt`
3. Download via `MODEL_URL` environment variable if configured.

---

## Streamlit Cloud Deployment

1. Connect your repository on [Streamlit Cloud](https://share.streamlit.io/) with the main script path set to `teeth-braces-ai/streamlit_app.py`.
2. Model weights (`teeth-braces-ai/best.pt`) are tracked in Git and will be loaded automatically.
3. Linux system dependencies (`libgl1`, `libglib2.0-0`) are auto-installed via `packages.txt`.

---

## Configuration

- **Inference Image Size**: 640, 960, or 1280 pixels.
- **Confidence Threshold**: 0.00 to 1.00 (default: 0.40).

---

## License

MIT License
