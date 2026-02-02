# 💧 Water Meter AI Detection System

**AI-powered water meter digit detection and reading extraction with YOLOv8-OBB (Oriented Bounding Box)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-green.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange.svg)](https://gradio.app)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Demo Application](#-demo-application)
- [Dataset](#-dataset)

---

## 🎯 Features

### 🚀 Demo Application (`app_demo/`)

- **📊 Dashboard**: Real-time KPI monitoring
- **📷 Single Image**: Individual analysis with JSON output
- **📂 Batch Processing**: Multiple images with CSV export
- **🎥 Video Analysis**: Frame-by-frame detection
- **📹 Live Webcam**: Real-time camera feed

### 🎓 Training System (`training/`)

- **📈 Training Dashboard**: Live metrics monitoring
- **⚙️ Config Management**: YAML-based system
- **📝 Scripts**: Local + Google Colab support
- **📊 Logging**: Comprehensive training logs

---

## 🏗️ Project Structure

```
Water-Meter-Dataset/
│
├── 📱 app_demo/                    # Demo Application
│   ├── main.py                     # Run demo
│   ├── assets/models/
│   │   └── water_meter_model.pt   # Trained model
│   └── src/                        # Modular architecture
│       ├── app.py                  # Gradio UI
│       ├── core.py                 # Main orchestrator
│       ├── inference/              # Detection parsing
│       ├── visualization/          # Annotation rendering
│       ├── utils/                  # Video I/O, helpers
│       └── preprocessing/          # Enhancement (placeholder)
│
├── 🎓 training/                    # Training System (Complete)
│   ├── training_dashboard.py      # Monitor training
│   ├── scripts/                    # Training scripts
│   │   ├── train.py               # Main training
│   │   ├── export.py              # Model export
│   │   └── resplit_dataset.py    # Dataset splitting
│   ├── configs/                    # YAML configs
│   │   ├── train_config.yaml      # Training config
│   │   └── export_config.yaml     # Export config
│   ├── notebooks/                  # Jupyter notebooks
│   │   ├── 02_training_local.ipynb
│   │   └── 03_training_colab.ipynb
│   ├── runs/                       # Training outputs
│   ├── logs/                       # Log files
│   ├── requirements.txt            # Training dependencies
│   └── README.md                   # Training docs
│
├── 📊 Dataset                      # YOLO Dataset
│   ├── data.yaml                   # Dataset config
│   ├── train/                      # 80% training images
│   ├── valid/                      # 15% validation images
│   └── test/                       # 5% test images
│
├── 🔧 Shared Resources
│   ├── src/utils/                  # Config, logging utilities
│   ├── assets/                     # Shared assets
│   ├── deployment/                 # Deployment configs
│   ├── requirements.txt            # Base dependencies
│   └── setup.py                    # Package setup
│
└── 📦 Model Files
    ├── yolo11n.pt                  # YOLO11 base model
    └── yolov8s-obb.pt             # YOLOv8 OBB base model
```

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

**Demo Application:**

```bash
cd app_demo
pip install -r requirements.txt
```

**Training:**

```bash
cd training
pip install -r requirements.txt
```

### 2️⃣ Run Demo

```bash
cd app_demo
python main.py
```

Open: `http://localhost:7860`

---

## 🎓 Training

### Local Training

```bash
cd training
python scripts/train.py --config configs/train_config.yaml
```

### Google Colab

1. Upload `notebooks/03_training_colab.ipynb`
2. Select **T4 GPU** runtime
3. Run all cells

### Monitor Progress

```bash
cd training
python training_dashboard.py
```

---

## 📱 Demo Features

### Architecture Highlights

**Modular Design:**

- ✅ OBB detection parsing & spatial sorting
- ✅ License plate style annotations
- ✅ FFmpeg H.264 video conversion
- ✅ Ready for enhancement pipeline

**Key Features:**

- Left-to-right digit sorting
- Navy Blue theme (#112D4E)
- Browser-compatible video
- JSON + CSV exports

---

## 📊 Dataset

### Classes (11 Total)

- **0-9**: `digit_0` - `digit_9`
- **10**: `border_water_meter_number`

### Split Ratio

- Training: 80%
- Validation: 15%
- Test: 5%

### Format

- **YOLOv8-OBB** (Oriented Bounding Box)
- Coordinates: `xyxyxyxy` (4 corners)

---

## 📈 Performance

**Based on 100 epochs training with YOLOv8s-OBB:**

| Metric             | Value       | Notes                              |
| ------------------ | ----------- | ---------------------------------- |
| **mAP50 (Box)**    | **94.44%**  | @ epoch 100 (final)                |
| **mAP50-95 (Box)** | **77.96%**  | @ epoch 100 (final)                |
| **Precision**      | **94.16%**  | @ epoch 100                        |
| **Recall**         | **85.26%**  | @ epoch 100                        |
| **Model Size**     | **22.2 MB** | YOLOv8s-OBB (water_meter_model.pt) |
| **Training Time**  | **~10.8h**  | 100 epochs on GPU                  |
| **Inference**      | ~30ms       | GPU (estimated)                    |

### 📊 Training Configuration

- **Base Model**: YOLOv8s-OBB (Oriented Bounding Box)
- **Image Size**: 512x512
- **Batch Size**: 24
- **Optimizer**: AdamW
- **Device**: CUDA (GPU)
- **Epochs**: 100
- **Dataset Split**: Train 80% / Valid 15% / Test 5%

### 📈 Key Achievements

- ✅ **Best mAP50**: 94.44% (excellent detection accuracy)
- ✅ **Consistent Training**: Loss decreased steadily from epoch 1 to 100
- ✅ **High Precision**: 94.16% (low false positives)
- ✅ **Good Recall**: 85.26% (captures most digits)
- ✅ **Production Ready**: Model ready for deployment

### 📁 Training Results Available

Full training artifacts available in `training/runs/exp_archive/`:

- `results.csv` - Complete metrics per epoch
- `confusion_matrix.png` - Class confusion analysis
- `results.png` - Training curves visualization
- `BoxPR_curve.png` - Precision-Recall curve
- `weights/water_meter_model.pt` - Best trained model

---

## 👨‍💻 Author

**Arsenius Purbandono**

- GitHub: [@xzars-git](https://github.com/xzars-git)

---

## 🙏 Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Gradio](https://gradio.app/)
- [Roboflow](https://roboflow.com/)

---

<div align="center">

**⭐ Star this repo if it helps you!**

Made with ❤️ by Arsenius Purbandono

</div>
