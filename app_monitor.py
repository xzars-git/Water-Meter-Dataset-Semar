"""
Real-time Training Monitor App
Web-based dashboard for monitoring YOLOv8 training

Usage:
    python app_monitor.py
    
Then open: http://localhost:7860
"""
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import yaml
import time
from typing import List, Dict, Tuple
import psutil
import GPUtil

# Global state
training_running = False
current_epoch = 0
metrics_history = []


def get_system_stats() -> Dict:
    """Get real-time system statistics"""
    try:
        # GPU stats
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_util = f"{gpu.load * 100:.1f}%"
            gpu_mem = f"{gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil * 100:.1f}%)"
            gpu_temp = f"{gpu.temperature}°C"
        else:
            gpu_util = "N/A"
            gpu_mem = "N/A"
            gpu_temp = "N/A"
        
        # RAM stats
        ram = psutil.virtual_memory()
        ram_used = f"{ram.used / (1024**3):.1f} GB"
        ram_total = f"{ram.total / (1024**3):.1f} GB"
        ram_percent = f"{ram.percent:.1f}%"
        
        return {
            "GPU Utilization": gpu_util,
            "GPU Memory": gpu_mem,
            "GPU Temperature": gpu_temp,
            "RAM Used": f"{ram_used} / {ram_total}",
            "RAM Usage": ram_percent,
        }
    except Exception as e:
        return {"Error": str(e)}


def load_training_results(project_path: str = "water-meter-ai") -> Tuple[pd.DataFrame, str]:
    """Load training results from runs directory"""
    try:
        # Find latest experiment
        runs_dir = Path(project_path)
        if not runs_dir.exists():
            return pd.DataFrame(), "No training runs found"
        
        # Find most recent experiment
        exp_dirs = sorted(runs_dir.glob("exp*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not exp_dirs:
            return pd.DataFrame(), "No experiments found"
        
        latest_exp = exp_dirs[0]
        results_csv = latest_exp / "results.csv"
        
        if not results_csv.exists():
            return pd.DataFrame(), f"No results.csv in {latest_exp.name}"
        
        # Load results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        status = f"✅ Loaded: {latest_exp.name} | Epochs: {len(df)}"
        return df, status
        
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {str(e)}"


def create_metrics_plot(df: pd.DataFrame) -> go.Figure:
    """Create interactive metrics plot"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font_size=20)
        return fig
    
    fig = go.Figure()
    
    # Loss curves
    if 'train/box_loss' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['train/box_loss'],
            name='Train Box Loss', mode='lines+markers'
        ))
    
    if 'val/box_loss' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['epoch'], y=df['val/box_loss'],
            name='Val Box Loss', mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Training Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig


def create_map_plot(df: pd.DataFrame) -> go.Figure:
    """Create mAP metrics plot"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font_size=20)
        return fig
    
    fig = go.Figure()
    
    # mAP metrics
    map_cols = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    colors = ['#00ff00', '#ffaa00']
    
    for col, color in zip(map_cols, colors):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['epoch'], y=df[col],
                name=col.split('/')[-1].replace('(B)', ''),
                mode='lines+markers',
                line=dict(color=color, width=3)
            ))
    
    fig.update_layout(
        title="mAP Metrics (Target: mAP50 > 95%)",
        xaxis_title="Epoch",
        yaxis_title="mAP Score",
        yaxis_range=[0, 1],
        hovermode='x unified',
        template='plotly_dark'
    )
    
    # Add target line
    fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                  annotation_text="Target: 95%")
    
    return fig


def get_current_metrics(df: pd.DataFrame) -> str:
    """Get current training metrics as formatted text"""
    if df.empty:
        return "No metrics available"
    
    latest = df.iloc[-1]
    
    metrics_text = f"""
## 📊 Current Metrics (Epoch {int(latest['epoch'])})

### Detection Performance
- **mAP50:** {latest.get('metrics/mAP50(B)', 0):.4f} {'✅' if latest.get('metrics/mAP50(B)', 0) > 0.90 else '⚠️'}
- **mAP50-95:** {latest.get('metrics/mAP50-95(B)', 0):.4f}
- **Precision:** {latest.get('metrics/precision(B)', 0):.4f}
- **Recall:** {latest.get('metrics/recall(B)', 0):.4f}

### Loss Values
- **Box Loss:** {latest.get('train/box_loss', 0):.4f}
- **Class Loss:** {latest.get('train/cls_loss', 0):.4f}
- **DFL Loss:** {latest.get('train/dfl_loss', 0):.4f}

### Status
{'🎯 **Target Achieved!** Model ready for deployment.' if latest.get('metrics/mAP50(B)', 0) > 0.95 else '⏳ Still training... Target: mAP50 > 95%'}
    """
    
    return metrics_text


def refresh_dashboard():
    """Refresh all dashboard components"""
    df, status = load_training_results()
    loss_plot = create_metrics_plot(df)
    map_plot = create_map_plot(df)
    metrics_text = get_current_metrics(df)
    system_stats = get_system_stats()
    
    # Format system stats
    stats_text = "\n".join([f"**{k}:** {v}" for k, v in system_stats.items()])
    
    return loss_plot, map_plot, metrics_text, stats_text, status


# Create Gradio Interface
with gr.Blocks(title="Water Meter AI - Training Monitor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🚀 Water Meter AI - Real-time Training Monitor
    **RTX 3080 Ti + 32GB RAM | Dataset: 23,986 images**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            status_text = gr.Markdown("Status: Ready")
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh", variant="primary")
    
    with gr.Tabs():
        with gr.Tab("📈 Metrics"):
            with gr.Row():
                with gr.Column():
                    loss_plot = gr.Plot(label="Loss Curves")
                with gr.Column():
                    map_plot = gr.Plot(label="mAP Metrics")
            
            current_metrics = gr.Markdown("No metrics yet")
        
        with gr.Tab("💻 System Monitor"):
            system_stats = gr.Markdown("Loading...")
            
            gr.Markdown("""
            ### GPU Configuration
            - **Model:** RTX 3080 Ti (12GB VRAM)
            - **Batch Size:** 32 (optimized for your GPU)
            - **AMP:** Enabled (faster training)
            - **Cache:** Enabled (32GB RAM sufficient)
            
            ### Dataset Info
            - **Train:** 22,512 images (94%)
            - **Valid:** 795 images (3%)
            - **Test:** 679 images (3%)
            - **Total:** 23,986 images
            
            ### Augmentation (3x outputs per image)
            - Rotation: ±15°
            - Shear: ±10° H/V
            - Brightness: ±25%
            - Exposure: ±20%
            - Blur: up to 2.5px
            - Noise: up to 1.96%
            """)
        
        with gr.Tab("📋 Training Config"):
            gr.Markdown("""
            ### Current Configuration
            
            ```yaml
            model: yolov8n-obb
            epochs: 100
            batch_size: 32  # RTX 3080 Ti optimized
            image_size: 512x512
            cache: true     # 32GB RAM
            workers: 12
            
            optimizer: AdamW
            lr0: 0.01
            patience: 20
            
            device: cuda:0  # RTX 3080 Ti
            amp: true       # Mixed precision
            ```
            
            ### Performance Targets
            - ✅ mAP50: > 95%
            - ✅ Model Size: < 10MB (TFLite INT8)
            - ✅ Inference: < 100ms (mobile)
            
            ### Commands
            ```powershell
            # Start training
            python scripts/train.py --config configs/train_config.yaml
            
            # Export to TFLite
            python scripts/export.py --weights runs/train/exp/weights/best.pt
            ```
            """)
    
    # Auto-refresh functionality
    refresh_btn.click(
        fn=refresh_dashboard,
        outputs=[loss_plot, map_plot, current_metrics, system_stats, status_text]
    )
    
    # Initial load
    app.load(
        fn=refresh_dashboard,
        outputs=[loss_plot, map_plot, current_metrics, system_stats, status_text]
    )


if __name__ == "__main__":
    print("="*60)
    print("🚀 Water Meter AI - Training Monitor")
    print("="*60)
    print("RTX 3080 Ti + 32GB RAM | 23,986 images")
    print("="*60)
    print("\n📊 Starting web dashboard...")
    print("🌐 Open: http://localhost:7860")
    print("\n💡 Tip: Refresh every 30 seconds to see training progress")
    print("="*60)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True to create public link
        inbrowser=True
    )
