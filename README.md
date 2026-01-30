# Water Meter AI - On-Device Detection

**Author:** Arsenius Purbandono  
**Hardware:** RTX 3080 Ti (12GB) | Ryzen 7 9700X | 32GB DDR5  
**Dataset:** 23,986 images (OBB) | Model: YOLOv8s-obb  
**Target:** TFLite INT8 (<10MB) for Flutter mobile deployment

---

## 🎯 Executive Summary & Strategic Value

Proyek ini adalah solusi **Digital Transformation (DX)** untuk mengubah meteran air analog (legacy) menjadi data digital secara real-time menggunakan **On-Device AI (Edge Computing)**.

### Problem Statement

- ❌ Pencatatan manual rawan _human error_
- ❌ Proses lambat dan biaya operasional tinggi
- ❌ Tidak ada real-time monitoring

### Solution

- ✅ _On-Device AI_ yang berjalan **offline** di HP petugas
- ✅ Real-time detection dengan **< 100ms latency**
- ✅ **Zero cloud dependency** (Edge Computing)

### Key Performance Indicators (KPI)

| Metric             | Target        | Impact                                   |
| ------------------ | ------------- | ---------------------------------------- |
| **Efficiency**     | 70% faster    | Memangkas waktu input data               |
| **Cost Saving**    | 60% reduction | Mengurangi kebutuhan server cloud        |
| **Accuracy**       | >95% mAP50    | Menangani rotasi angka & kondisi ekstrem |
| **Inference Time** | <100ms        | Real-time experience                     |
| **Model Size**     | <10MB         | Fast deployment ke mobile                |

---

## 📊 Dataset Technical Specifications

### Dataset Statistics

```
Total Images: 23,986 images
├── Train: 22,512 images (94%)
├── Valid: 795 images (3%)
└── Test: 679 images (3%)

Annotation Type: Oriented Bounding Box (OBB)
Classes: 0-9 (10 digit classes)
```

### Preprocessing Strategy

- **Auto-Orient:** Applied (prevent metadata rotation errors)
- **Resize:** Fit (black edges) in 512x512
  - _Preserves aspect ratio_ - Critical untuk geometri angka

### Augmentation Strategy

**Multiplier:** 3x per image

| Augmentation | Range       | Purpose                      |
| ------------ | ----------- | ---------------------------- |
| Blur         | Up to 2.5px | Simulasi motion blur         |
| Noise        | Up to 1.96% | Simulasi ISO tinggi          |
| Brightness   | ±25%        | Variasi kondisi cahaya       |
| Exposure     | ±20%        | Handle backlight             |
| Rotation     | ±15°        | Toleransi kemiringan         |
| Shear        | ±10° H/V    | Perspektif tidak tegak lurus |

---

## Quick Start

**Install:**

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Monitor (Terminal 1):**

```powershell
python app_monitor.py
# Opens http://localhost:7860
```

**Train (Terminal 2):**

```powershell
python scripts/train.py --config configs/train_config.yaml
```

**Export:**

```powershell
python scripts/export.py --weights runs/train/exp/weights/best.pt
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA 11.8+ (for GPU training)
nvidia-smi
```

### Installation

#### Local Setup

```bash
# Clone repository
git clone <your-repo-url>
cd water-meter-ai

# Create virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Google Colab Setup

```python
# Upload to Colab and run:
!git clone <your-repo-url>
%cd water-meter-ai
!pip install -r requirements.txt
```

### Training

#### Local Training

```bash
python scripts/train.py --config configs/train_config.yaml
```

#### Colab Training

Open `notebooks/03_training_colab.ipynb` in Google Colab

### Evaluation

```bash
python scripts/evaluate.py --weights runs/train/exp/weights/best.pt
```

### Export to TFLite

```bash
python scripts/export.py --weights runs/train/exp/weights/best.pt --format tflite
```

---

## 📈 Training Pipeline

### Phase 1: Baseline Training

- Model: YOLOv8n-obb
- Epochs: 100
- Image Size: 512x512
- Batch Size: 16 (adjust based on GPU)

### Phase 2: Hyperparameter Tuning

- Learning Rate: [0.001, 0.01]
- Augmentation Intensity: [0.5, 1.0]
- Model Size: [n, s, m]

### Phase 3: Model Optimization

- Quantization: Int8
- Pruning (if needed)
- Final Format: TFLite

---

## 🎯 Model Performance Targets

| Metric                  | Target | Minimum Acceptable |
| ----------------------- | ------ | ------------------ |
| mAP50                   | >95%   | >90%               |
| mAP50-95                | >85%   | >80%               |
| Inference Time (Mobile) | <100ms | <150ms             |
| Model Size              | <10MB  | <15MB              |
| False Positive Rate     | <5%    | <10%               |

---

## 🔬 Experiment Tracking

We use **Weights & Biases** for experiment tracking:

```python
import wandb
wandb.init(project="water-meter-ai", name="experiment-1")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py -v
```

---

## 📱 Deployment

### Flutter Integration

See [deployment/flutter/README.md](deployment/flutter/README.md)

### Model Files

- **Training Checkpoint:** `runs/train/exp/weights/best.pt`
- **ONNX:** `exports/best.onnx`
- **TFLite:** `exports/best.tflite` (Int8 Quantized)

---

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit PR

---

## 📝 License

[Your License Here]

---

## 📧 Contact

**Arsenius Purbandono**  
AI & Mobile Engineer  
[Your Email/LinkedIn]
