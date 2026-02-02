# 💧 Water Meter AI Detection System

Professional AI-powered water meter detection and reading extraction system with modern web interface.

## 🎯 Features

- **📊 Model Dashboard**: Real-time performance metrics and KPI monitoring
- **📷 Single Image Detection**: Analyze individual water meter images
- **📂 Batch Processing**: Process multiple images simultaneously
- **🎥 Video Analysis**: Track water meters in video footage
- **📹 Live Webcam**: Real-time detection from webcam feed

## 🏗️ Project Structure

```text
water-meter-ai/
│
├── main.py                  # 🚀 Entry point - Run this file
├── requirements.txt         # 📦 Python dependencies
├── README.md               # 📖 Documentation (this file)
│
├── assets/                 # 📁 Static assets
│   └── models/
│       └── water_meter_model.pt  # YOLOv8-OBB model weights
│
├── src/                    # 💻 Source code package
│   ├── __init__.py         # Package initialization
│   ├── app.py              # Gradio web interface
│   ├── core.py             # Business logic & ML inference
│   └── theme.py            # UI design system (SeaSide Theme)
│
├── water-meter-ai/         # 🔬 Training artifacts
│   └── exp/
│       ├── weights/        # Model checkpoints
│       └── results.csv     # Training metrics
│
├── train/                  # 📊 Training dataset
├── valid/                  # ✅ Validation dataset
└── test/                   # 🧪 Test dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
python main.py
```

### 3. Access Web Interface

Open your browser and navigate to:

```
http://localhost:7860
```

## 📦 Dependencies

- **Gradio** ≥ 6.0: Web interface framework
- **Pandas**: Data manipulation
- **Pillow**: Image processing
- **OpenCV**: Video processing
- **NumPy**: Numerical operations
- **Ultralytics** (optional): YOLOv8 inference

## 🎨 Design System

The application uses **SeaSide Theme** - a professional Japanese corporate design with:

- **Primary Color**: Navy Blue (#112D4E)
- **Accent Color**: Medium Blue (#3F72AF)
- **Background**: Dark gradient (#0a1929 → #0f2744)
- **Typography**: Roboto font family
- **Effects**: Glass morphism with blur effects

## 🔧 Architecture

### Core Components

1. **WaterMeterSystem** (`src/core.py`)
   - Handles all ML inference operations
   - Supports single image, batch, video, and webcam
   - Currently in MOCK mode for development

2. **Gradio Interface** (`src/app.py`)
   - 5-tab navigation system
   - LinkedIn-style app bar
   - Responsive layout with glass morphism

3. **SeaSide Theme** (`src/theme.py`)
   - Custom Gradio theme
   - Japanese corporate color palette
   - Prevents dark mode conflicts

## 📊 Model Information

- **Architecture**: YOLOv8-OBB (Oriented Bounding Boxes)
- **Training**: 100 epochs
- **Performance**:
  - mAP@50: 94.4%
  - Precision: 94.2%
  - Recall: 85.3%
  - mAP@50-95: 78.0%

## 🛠️ Development

### Running in Development Mode

The system automatically runs in MOCK mode if the model file is not found. This allows frontend development without requiring the actual model.

### Project Structure Explained

- **`main.py`**: Single entry point - handles all launch configuration
- **`src/`**: Clean Python package with proper imports
- **`assets/`**: All static resources (models, images, styles)
- **No root-level clutter**: Professional package organization

### Code Organization

```python
# Import the application
from src.app import demo
from src.theme import get_seaside_theme
from src.core import WaterMeterSystem

# Everything is properly packaged
```

## 🔒 Security & Privacy

- **Webcam Processing**: All processing happens locally in browser
- **No Data Transmission**: Video data is never sent to external servers
- **Local Inference**: Model runs entirely on your hardware

## 📝 License

© 2026 Bapenda Water Meter Detection Project. All Rights Reserved.

## 👨‍💻 Author

**Bapenda Water Meter Detection Project**  
Version: 1.0.0  
Framework: Ultralytics YOLOv8-OBB  
UI: Gradio v6.0

---

**Need Help?**  
Check the code documentation in each module or contact the development team.
