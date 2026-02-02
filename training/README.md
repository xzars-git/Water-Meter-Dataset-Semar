# 🎓 Water Meter AI - Training System

Training pipeline for YOLOv8-OBB water meter digit detection model.

**Latest Model Performance:**

- ✅ mAP50: **94.44%** @ 100 epochs
- ✅ Precision: **94.16%** | Recall: **85.26%**
- ✅ Model: `training/runs/exp_archive/weights/water_meter_model.pt` (22.2 MB)

---

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

**Local Training:**

```bash
python scripts/train.py --config configs/train_config.yaml
```

**Google Colab:**

- Upload `notebooks/03_training_colab.ipynb` to Colab
- Select **T4 GPU** runtime
- Run all cells

### 3. Monitor Progress

```bash
python training_dashboard.py
```

Open: `http://localhost:7860`

---

## 📁 Folder Structure

```
training/
├── training_dashboard.py      # Live monitoring dashboard
├── requirements.txt            # Training dependencies
│
├── configs/                    # Training configurations
│   └── train_config.yaml      # Main config file
│
├── scripts/                    # Training scripts
│   ├── train.py               # Main training script
│   ├── export.py              # Model export (ONNX, TFLite)
│   └── resplit_dataset.py    # Dataset splitting utility
│
├── notebooks/                  # Jupyter notebooks
│   ├── 02_training_local.ipynb
│   └── 03_training_colab.ipynb
│
└── runs/                       # Training outputs
    └── exp*/                   # Experiment folders
        ├── weights/
        │   ├── best.pt        # Best model
        │   └── last.pt        # Last epoch
        └── results.csv        # Training metrics
```

---

## ⚙️ Configuration

Edit `configs/train_config.yaml`:

```yaml
model:
  type: "yolov8n-obb" # Options: n, s, m, l, x

training:
  epochs: 100
  batch_size: 16
  patience: 20
  lr0: 0.01
  lrf: 0.01

augmentation:
  degrees: 15.0 # Rotation
  fliplr: 0.0 # ⚠️ DISABLED for digits!
  mosaic: 1.0 # Mosaic augmentation

hardware:
  device: "" # Auto-detect GPU
  workers: 8 # Data loading workers
  amp: true # Mixed precision
```

---

## 📊 Training Dashboard

Monitor training in real-time:

**Features:**

- 📈 mAP50 progress chart
- 📊 Loss values (box, cls, dfl)
- 💻 System stats (GPU, CPU, RAM)
- 📂 Experiment tracking
- 🔄 Auto-refresh every 3s

**Usage:**

```bash
python training_dashboard.py
```

---

## 📝 Scripts

### 1. Train Model

```bash
python scripts/train.py --config configs/train_config.yaml
```

**Options:**

- `--config`: Config file path
- `--resume`: Resume from checkpoint
- `--device`: Device (cuda:0, cpu, etc.)

### 2. Export Model

```bash
python scripts/export.py --weights runs/exp/weights/best.pt --format onnx
```

**Supported Formats:**

- `onnx`: ONNX (cross-platform)
- `tflite`: TensorFlow Lite (mobile)
- `engine`: TensorRT (NVIDIA GPU)
- `coreml`: CoreML (Apple devices)

### 3. Resplit Dataset

```bash
python scripts/resplit_dataset.py --train-ratio 0.80 --valid-ratio 0.15 --test-ratio 0.05
```

---

## 📈 Expected Results

| Model       | mAP50 | Speed (GPU) | Size  |
| ----------- | ----- | ----------- | ----- |
| YOLOv8n-OBB | 95%+  | ~30ms       | ~6MB  |
| YOLOv8s-OBB | 96%+  | ~45ms       | ~22MB |
| YOLOv8m-OBB | 97%+  | ~70ms       | ~50MB |

---

## 🐛 Troubleshooting

**Out of Memory:**

```yaml
training:
  batch_size: 8 # Reduce batch size
```

**Slow Training:**

```yaml
hardware:
  workers: 4 # Reduce workers
  amp: true # Enable mixed precision
```

**Poor Performance:**

```yaml
training:
  epochs: 150 # Increase epochs
  patience: 30 # Increase patience
```

---

## 📧 Support

For issues: [GitHub Issues](https://github.com/xzars-git/Water-Meter-Dataset-Semar/issues)

---

Made with ❤️ by Arsenius Purbandono
