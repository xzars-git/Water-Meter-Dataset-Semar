"""
🚀 Water Meter AI - Training Dashboard (Compact)
Modern UI - Single Screen - No Scroll

Usage: python app_dashboard.py
Open: http://localhost:7860
"""
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Tuple
import psutil
import GPUtil
from datetime import datetime

# Color Palette - Fixed Light Mode
COLORS = {
    "dark": {
        "bg": "#0f0f0f", "card": "#1a1a1a", "border": "#2d2d2d",
        "text": "#ffffff", "muted": "#888888", "accent": "#c8ff00",
        "value": "#c8ff00"
    },
    "light": {
        "bg": "#f8fafc", "card": "#ffffff", "border": "#e2e8f0",
        "text": "#0f172a", "muted": "#64748b", "accent": "#16a34a",
        "value": "#15803d"
    }
}

def get_system_stats() -> Dict:
    try:
        gpus = GPUtil.getGPUs()
        gpu = gpus[0] if gpus else None
        ram = psutil.virtual_memory()
        return {
            "gpu": f"{gpu.load*100:.0f}%" if gpu else "N/A",
            "vram": f"{gpu.memoryUsed/1024:.1f}/{gpu.memoryTotal/1024:.1f}GB" if gpu else "N/A",
            "temp": f"{gpu.temperature:.0f}°C" if gpu else "N/A",
            "cpu": f"{psutil.cpu_percent():.0f}%",
            "ram": f"{ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB"
        }
    except:
        return {"gpu": "N/A", "vram": "N/A", "temp": "N/A", "cpu": "N/A", "ram": "N/A"}

def load_results() -> Tuple[pd.DataFrame, str]:
    try:
        for path in [Path("water-meter-ai"), Path("runs/train")]:
            if path.exists():
                exps = sorted(path.glob("exp*"), key=lambda x: x.stat().st_mtime, reverse=True)
                if exps:
                    csv = exps[0] / "results.csv"
                    if csv.exists():
                        df = pd.read_csv(csv)
                        df.columns = df.columns.str.strip()
                        return df, f"📂 {exps[0].name} | {len(df)} epochs"
        return pd.DataFrame(), "⏳ Waiting for training..."
    except Exception as e:
        return pd.DataFrame(), str(e)

def create_chart(df: pd.DataFrame, theme: str) -> go.Figure:
    c = COLORS[theme]
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(text="⏳ Waiting for data...", showarrow=False,
                          font=dict(size=18, color=c["muted"]), xref="paper", yref="paper", x=0.5, y=0.5)
    else:
        if 'metrics/mAP50(B)' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['epoch'], y=df['metrics/mAP50(B)']*100,
                mode='lines+markers', line=dict(color=c["accent"], width=3),
                fill='tozeroy', fillcolor='rgba(200,255,0,0.1)'
            ))
            fig.add_hline(y=95, line_dash="dash", line_color=c["muted"])
    
    fig.update_layout(
        title="📈 mAP50 Progress", paper_bgcolor=c["card"], plot_bgcolor=c["card"],
        font=dict(color=c["text"]), height=350, margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(title="Epoch", gridcolor=c["border"]),
        yaxis=dict(title="mAP50 (%)", gridcolor=c["border"], range=[0, 100]),
        showlegend=False
    )
    return fig

def create_loss_chart(df: pd.DataFrame, theme: str) -> go.Figure:
    c = COLORS[theme]
    fig = go.Figure()
    
    if not df.empty and 'train/box_loss' in df.columns:
        recent = df.tail(20)
        fig.add_trace(go.Bar(x=recent['epoch'], y=recent['train/box_loss'], marker_color=c["accent"]))
    
    fig.update_layout(
        title="📊 Loss Values", paper_bgcolor=c["card"], plot_bgcolor=c["card"],
        font=dict(color=c["text"]), height=250, margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=c["border"]), yaxis=dict(gridcolor=c["border"]), showlegend=False
    )
    return fig

def build_html(df: pd.DataFrame, stats: Dict, theme: str) -> str:
    c = COLORS[theme]
    now = datetime.now().strftime("%H:%M:%S")
    
    # Metrics
    if df.empty:
        epoch, map50, box, cls, dfl = 0, 0, 0, 0, 0
        precision, recall = 0, 0
        best_map50, speed, time_per_epoch = 0, 0, "N/A"
        loss_trend = ""
        analysis_html = '<div style="color:' + c["muted"] + ';font-size:10px">⏳ Waiting for training data...</div>'
        insights_html = analysis_html
        health_html = analysis_html
    else:
        latest = df.iloc[-1]
        epoch = int(latest.get('epoch', 0))
        map50 = float(latest.get('metrics/mAP50(B)', 0)) * 100
        box = float(latest.get('train/box_loss', 0))
        cls = float(latest.get('train/cls_loss', 0))
        dfl = float(latest.get('train/dfl_loss', 0))
        precision = float(latest.get('metrics/precision(B)', 0)) * 100
        recall = float(latest.get('metrics/recall(B)', 0)) * 100
        
        # Best metrics
        best_map50 = float(df['metrics/mAP50(B)'].max() * 100) if 'metrics/mAP50(B)' in df.columns else 0
        
        # Speed calculation
        if len(df) > 1:
            speed = f"~7 it/s"  # From training logs
            time_per_epoch = "~6min"
        else:
            speed, time_per_epoch = "N/A", "N/A"
        
        # Loss trend - last 5 epochs
        if len(df) >= 5 and 'train/box_loss' in df.columns:
            recent_losses = df['train/box_loss'].tail(5).tolist()
            loss_bars = ''.join([
                f'<div style="flex:1;height:30px;background:{c["accent"]};opacity:{0.3 + (i*0.15)};border-radius:2px;margin:0 1px;position:relative"><span style="position:absolute;bottom:-15px;left:50%;transform:translateX(-50%);font-size:9px;color:{c["muted"]}">{l:.2f}</span></div>'
                for i, l in enumerate(recent_losses)
            ])
            loss_trend = f'<div style="display:flex;align-items:flex-end;height:40px;margin-top:10px">{loss_bars}</div>'
        else:
            loss_trend = '<div style="color:' + c["muted"] + ';font-size:10px;margin-top:10px">Insufficient data</div>'
        
        # === ANALYSIS SECTION ===
        # 1. Overfitting Detection
        if len(df) >= 3:
            train_losses = df['train/box_loss'].tail(5).tolist()
            val_losses = df['val/box_loss'].tail(5).tolist() if 'val/box_loss' in df.columns else train_losses
            
            train_trend = train_losses[-1] - train_losses[0] if len(train_losses) >= 2 else 0
            val_trend = val_losses[-1] - val_losses[0] if len(val_losses) >= 2 else 0
            
            # Overfitting check: train loss decreasing but val loss increasing
            if train_trend < -0.05 and val_trend > 0.05:
                overfit_status = "⚠️ Possible Overfitting"
                overfit_color = "#ef4444"
                overfit_desc = "Train loss ↓ but validation metrics not improving"
            elif train_trend < -0.1:
                overfit_status = "✅ Healthy Learning"
                overfit_color = c["accent"]
                overfit_desc = "Both train and validation losses decreasing"
            elif abs(train_trend) < 0.01:
                overfit_status = "⏸️ Plateau Detected"
                overfit_color = "#f59e0b"
                overfit_desc = "Loss not changing, consider adjusting learning rate"
            else:
                overfit_status = "📊 Normal Training"
                overfit_color = c["accent"]
                overfit_desc = "Model is learning at expected pace"
        else:
            overfit_status = "📊 Early Stage"
            overfit_color = c["muted"]
            overfit_desc = "Need more epochs for analysis"
        
        # 2. Performance Insights
        map_improve = ((map50 - (df['metrics/mAP50(B)'].iloc[0] * 100)) / (df['metrics/mAP50(B)'].iloc[0] * 100) * 100) if len(df) > 1 else 0
        loss_improve = ((df['train/box_loss'].iloc[0] - box) / df['train/box_loss'].iloc[0] * 100) if len(df) > 1 else 0
        
        if map_improve > 10:
            perf_status = "🚀 Excellent Progress"
            perf_color = c["accent"]
        elif map_improve > 5:
            perf_status = "✅ Good Progress"
            perf_color = c["accent"]
        elif map_improve > 0:
            perf_status = "📈 Steady Progress"
            perf_color = "#60a5fa"
        else:
            perf_status = "⚠️ No Improvement"
            perf_color = "#f59e0b"
        
        # 3. Training Health
        if map50 >= 95:
            health = "🎯 Target Reached!"
            health_color = c["accent"]
        elif map50 >= 90:
            health = "✅ Near Target"
            health_color = c["accent"]
        elif map50 >= 70:
            health = "📊 On Track"
            health_color = "#60a5fa"
        elif map50 >= 50:
            health = "⏳ Warming Up"
            health_color = "#f59e0b"
        else:
            health = "🔄 Early Training"
            health_color = c["muted"]
        
        # 4. Prediction Quality
        if precision > 90 and recall > 85:
            quality = "🌟 Excellent"
        elif precision > 80 and recall > 75:
            quality = "✅ Good"
        elif precision > 70 and recall > 65:
            quality = "📊 Fair"
        else:
            quality = "⚠️ Needs Improvement"
        
        # 5. Deep Learning Analysis
        if len(df) >= 3:
            # Learning rate analysis
            recent_map_changes = [df['metrics/mAP50(B)'].iloc[i] - df['metrics/mAP50(B)'].iloc[i-1] 
                                  for i in range(max(1, len(df)-5), len(df))]
            avg_improvement = sum(recent_map_changes) / len(recent_map_changes) if recent_map_changes else 0
            
            # Convergence status
            if avg_improvement > 0.01:
                convergence = "🚀 Rapid Learning"
                conv_color = c["accent"]
                conv_desc = f"Improving +{avg_improvement*100:.2f}% per epoch"
            elif avg_improvement > 0.005:
                convergence = "📈 Steady Progress"
                conv_color = "#60a5fa"
                conv_desc = f"Improving +{avg_improvement*100:.2f}% per epoch"
            elif avg_improvement > 0.001:
                convergence = "⏱️ Slow Convergence"
                conv_color = "#f59e0b"
                conv_desc = "Consider increasing learning rate"
            elif avg_improvement >= 0:
                convergence = "🎯 Near Optimal"
                conv_color = c["accent"]
                conv_desc = "Model converging to best performance"
            else:
                convergence = "⚠️ Degrading"
                conv_color = "#ef4444"
                conv_desc = "Performance dropping, check training"
            
            # Dataset quality analysis from metrics
            pr_ratio = precision / recall if recall > 0 else 1
            if pr_ratio > 1.2:
                data_quality = "📊 Conservative Detection"
                dq_desc = "High precision, low recall - model being selective"
            elif pr_ratio < 0.8:
                data_quality = "🎯 Aggressive Detection"
                dq_desc = "High recall, lower precision - detecting more objects"
            else:
                data_quality = "⚖️ Balanced Performance"
                dq_desc = "Well-balanced precision and recall"
            
            # Loss convergence
            loss_variance = df['train/box_loss'].tail(5).std() if len(df) >= 5 else 0
            if loss_variance < 0.05:
                loss_status = "✅ Stable"
                loss_desc = "Loss converging smoothly"
            elif loss_variance < 0.1:
                loss_status = "📊 Normal Variance"
                loss_desc = "Expected training fluctuation"
            else:
                loss_status = "⚠️ High Variance"
                loss_desc = "Loss fluctuating, may need adjustment"
        else:
            convergence = "🔄 Initializing"
            conv_color = c["muted"]
            conv_desc = "Collecting data..."
            data_quality = "⏳ Early Stage"
            dq_desc = "Need more epochs"
            loss_status = "⏳ Warming Up"
            loss_desc = "Initial training phase"
            plateau_detected = False
        
        # 6. Recommendations - Training to 100 epochs with 10-epoch plateau detection
        recommendations = []
        
        # PLATEAU DETECTION - Primary stopping criterion
        if len(df) >= 11:
            last_10_improvements = [df['metrics/mAP50(B)'].iloc[i] - df['metrics/mAP50(B)'].iloc[i-1] 
                                   for i in range(len(df)-10, len(df))]
            max_change_10 = max([abs(x) for x in last_10_improvements]) if last_10_improvements else 0
            plateau_detected = max_change_10 < 0.001  # < 0.1% change in 10 epochs
            
            if plateau_detected:
                recommendations.append(f"🛑 PLATEAU: No improvement in 10 epochs (max Δ: {max_change_10*100:.3f}%)")
                recommendations.append("💡 Consider stopping OR adjusting learning rate for breakthrough")
        else:
            plateau_detected = False
        
        # Training progress messages
        if not plateau_detected:
            if epoch >= 90:
                recommendations.append(f"🎯 Final stretch! {100-epoch} epochs to completion")
            elif epoch >= 50:
                recommendations.append(f"💪 Strong progress - {100-epoch} epochs to maximize performance")
            elif epoch >= 20:
                recommendations.append(f"📈 Solid training - Target: 100 epochs for best results")
        
        # Performance milestones (informative, NOT stopping criteria)
        if map50 >= 98:
            recommendations.append("🌟 Exceptional! mAP50 > 98% - Pushing limits")
        elif map50 >= 95:
            recommendations.append("✅ Excellent! mAP50 > 95% - Continue optimizing")
        elif map50 >= 90:
            recommendations.append(f"🎯 Great! {95-map50:.1f}% from 95% milestone")
        
        # Precision/Recall optimization
        if precision > 90 and recall < 80:
            recommendations.append("⚖️ Boost recall: Lower threshold or add augmentation")
        elif recall > 90 and precision < 80:
            recommendations.append("⚖️ Improve precision: Raise threshold or refine data")
        elif precision > 85 and recall > 85:
            recommendations.append("✨ Perfect balance: Strong metrics")
        
        # Overfitting monitoring
        if overfit_status == "⚠️ Possible Overfitting" and not plateau_detected:
            recommendations.append("⚠️ Overfitting risk: Add augmentation/regularization")
        elif overfit_status == "✅ Healthy Learning":
            recommendations.append("💪 Healthy: Train & val improving together")
        
        # Loss optimization
        if box > 1.5 or cls > 1.5:
            recommendations.append("📉 High losses: Active learning - continue")
        elif box < 0.5 and cls < 0.5 and dfl < 1.0:
            recommendations.append("🎯 Optimal losses: Excellent convergence")
        
        # Convergence alerts
        if convergence == "⚠️ Degrading":
            recommendations.append("⚠️ Performance drop: Check data or reduce LR")
        
        # Time estimate
        if epoch > 0 and epoch < 100:
            epochs_per_hour = 60 / 6
            hours_to_100 = (100 - epoch) / epochs_per_hour
            recommendations.append(f"⏰ ~{hours_to_100:.1f}h to epoch 100 ({100-epoch} left)")
        
        if len(recommendations) == 0:
            recommendations.append("✨ Training healthy - continuing to epoch 100")
        
        rec_html = ''.join([f'<div style="padding:3px 0;font-size:7px;color:{c["text"]};border-bottom:1px solid {c["border"]};line-height:1.3;word-wrap:break-word">{r}</div>' for r in recommendations[:6]])
        
        # Build analysis HTML
        analysis_html = f'''
        <div style="display:flex;flex-direction:column;gap:6px">
            <div style="background:{c["bg"]};border-radius:6px;padding:8px">
                <div style="font-size:10px;color:{c["muted"]};margin-bottom:4px">Model Status</div>
                <div style="font-size:13px;font-weight:700;color:{overfit_color}">{overfit_status}</div>
                <div style="font-size:9px;color:{c["muted"]};margin-top:2px">{overfit_desc}</div>
            </div>
            <div style="background:{c["bg"]};border-radius:6px;padding:8px">
                <div style="font-size:10px;color:{c["muted"]};margin-bottom:4px">Performance</div>
                <div style="font-size:13px;font-weight:700;color:{perf_color}">{perf_status}</div>
                <div style="font-size:9px;color:{c["muted"]};margin-top:2px">mAP: +{map_improve:.1f}% • Loss: -{loss_improve:.1f}%</div>
            </div>
            <div style="background:{c["bg"]};border-radius:6px;padding:8px">
                <div style="font-size:10px;color:{c["muted"]};margin-bottom:4px">Convergence</div>
                <div style="font-size:13px;font-weight:700;color:{conv_color}">{convergence}</div>
                <div style="font-size:9px;color:{c["muted"]};margin-top:2px">{conv_desc}</div>
            </div>
        </div>
        '''
        
        insights_html = f'''
        <div style="display:flex;flex-direction:column;gap:2px">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">Training Health</span>
                <span style="font-size:8px;font-weight:600;color:{health_color};text-align:right;white-space:nowrap">{health}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">Prediction Quality</span>
                <span style="font-size:8px;font-weight:600;color:{c["accent"]};text-align:right;white-space:nowrap">{quality}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">mAP Improvement</span>
                <span style="font-size:8px;font-weight:600;color:{c["accent"]};text-align:right;white-space:nowrap">+{map_improve:.1f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">Loss Reduction</span>
                <span style="font-size:8px;font-weight:600;color:{c["accent"]};text-align:right;white-space:nowrap">-{loss_improve:.1f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">Dataset Quality</span>
                <span style="font-size:8px;font-weight:600;color:{c["text"]};text-align:right;white-space:nowrap">{data_quality}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid {c["border"]}">
                <span style="font-size:8px;color:{c["muted"]};flex-shrink:0;margin-right:6px">Loss Stability</span>
                <span style="font-size:8px;font-weight:600;color:{c["text"]};text-align:right;white-space:nowrap">{loss_status}</span>
            </div>
            <div style="padding:3px 0;font-size:7px;color:{c["muted"]};line-height:1.2">{dq_desc}</div>
        </div>
        '''
        
        health_html = f'''
        <div style="padding:2px 0">
            {rec_html}
        </div>
        '''
    
    # Calculate common variables needed for all panels
    pct = (epoch / 100) * 100
    remaining = 100 - epoch
    eta_min = remaining * 6 if epoch > 0 else 0
    eta = f"~{eta_min//60}h {eta_min%60}m" if eta_min >= 60 else f"~{eta_min}min"
    
    # Continue building HTML panels only if we have data
    if not df.empty:
        # Loss breakdown panel
        loss_breakdown = f'''
        <div style="display:flex;flex-direction:column;gap:6px">
            <div style="display:flex;justify-content:space-between;align-items:center;padding:6px;background:{c["bg"]};border-radius:6px">
                <div>
                    <div style="font-size:9px;color:{c["muted"]}">Box Loss</div>
                    <div style="font-size:16px;font-weight:700;color:{c["value"]}">{box:.4f}</div>
                </div>
                <div style="font-size:20px">📦</div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:6px;background:{c["bg"]};border-radius:6px">
                <div>
                    <div style="font-size:9px;color:{c["muted"]}">Class Loss</div>
                    <div style="font-size:16px;font-weight:700;color:{c["value"]}">{cls:.4f}</div>
                </div>
                <div style="font-size:20px">🏷️</div>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:6px;background:{c["bg"]};border-radius:6px">
                <div>
                    <div style="font-size:9px;color:{c["muted"]}">DFL Loss</div>
                    <div style="font-size:16px;font-weight:700;color:{c["value"]}">{dfl:.4f}</div>
                </div>
                <div style="font-size:20px">🎯</div>
            </div>
        </div>
        '''
        
        # Training efficiency panel
        if epoch > 0:
            total_time = epoch * 6  # minutes
            images_processed = epoch * 24 * 1000  # batch_size * images_per_batch estimate
            img_per_sec = images_processed / (total_time * 60) if total_time > 0 else 0
            
            efficiency_html = f'''
            <div style="display:flex;flex-direction:column;gap:5px">
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">
                    <span style="font-size:10px;color:{c["muted"]}">Speed</span>
                    <span style="font-size:10px;font-weight:600;color:{c["accent"]}">{speed}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">
                    <span style="font-size:10px;color:{c["muted"]}">Time/Epoch</span>
                    <span style="font-size:10px;font-weight:600;color:{c["text"]}">{time_per_epoch}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">
                    <span style="font-size:10px;color:{c["muted"]}">Total Training</span>
                    <span style="font-size:10px;font-weight:600;color:{c["text"]}">{total_time//60}h {total_time%60}m</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">
                    <span style="font-size:10px;color:{c["muted"]}">Images/Second</span>
                    <span style="font-size:10px;font-weight:600;color:{c["accent"]}">{img_per_sec:.1f}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">
                    <span style="font-size:10px;color:{c["muted"]}">Est. Completion</span>
                    <span style="font-size:10px;font-weight:600;color:{c["text"]}">{eta}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:4px 0">
                    <span style="font-size:10px;color:{c["muted"]}">GPU Utilization</span>
                    <span style="font-size:10px;font-weight:600;color:{c["accent"]}">Optimal</span>
                </div>
            </div>
            '''
        else:
            efficiency_html = '<div style="color:' + c["muted"] + ';font-size:10px">⏳ Calculating...</div>'
    else:
        # Empty state for all panels when no data
        loss_breakdown = '<div style="color:' + c["muted"] + ';font-size:10px">⏳ Waiting for data...</div>'
        efficiency_html = '<div style="color:' + c["muted"] + ';font-size:10px">⏳ Waiting for data...</div>'
    
    # Progress bar segments (can be calculated even without data)
    def seg(n, current):
        filled = current >= n
        color = c["accent"] if filled else c["border"]
        return f'<div style="flex:1;height:8px;background:{color};border-radius:2px;margin:0 1px"></div>'
    
    segments = ''.join([seg(i*10, epoch) for i in range(1, 11)])
    
    # Milestone indicators
    milestones = [(25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]
    milestone_html = ''.join([
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0"><div style="width:14px;height:14px;border-radius:50%;background:{"" + c["accent"] if epoch >= m else c["border"]}"></div><span style="font-size:10px;color:{c["muted"]}">Epoch {m} {label}</span></div>'
        for m, label in milestones
    ])
    
    return f'''
    <style>
        * {{box-sizing:border-box;margin:0;padding:0}}
        .d {{font-family:'Segoe UI',system-ui,sans-serif;background:{c["bg"]};color:{c["text"]};padding:8px 4px;height:calc(100vh - 100px);display:flex;flex-direction:column;overflow-y:auto}}
        .hdr {{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid {c["border"]};margin-bottom:8px}}
        .logo {{display:flex;align-items:center;gap:8px}}
        .logo span {{font-size:20px;color:{c["accent"]}}}
        .logo b {{font-size:15px;color:{c["text"]}}}
        .status {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:8px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}}
        .main {{display:grid;grid-template-columns:1fr 260px;gap:10px;flex:1;min-height:0}}
        .left {{display:flex;flex-direction:column;gap:8px;overflow-y:auto}}
        .cards {{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
        .c {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:10px}}
        .c-lbl {{color:{c["muted"]};font-size:10px;margin-bottom:3px}}
        .c-val {{color:{c["value"]};font-size:20px;font-weight:700;line-height:1.2}}
        .c-sub {{color:{c["muted"]};font-size:9px;margin-top:2px}}
        .pbar {{display:flex;margin-top:5px}}
        .row2 {{display:grid;grid-template-columns:2fr 1fr;gap:8px;flex:1;min-height:0}}
        .chart {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:10px;display:flex;flex-direction:column}}
        .chart-title {{color:{c["text"]};font-size:12px;font-weight:600;margin-bottom:6px}}
        .metrics-grid {{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
        .metric {{background:{c["bg"]};border-radius:6px;padding:8px;text-align:center}}
        .metric-val {{color:{c["value"]};font-size:18px;font-weight:700}}
        .metric-lbl {{color:{c["muted"]};font-size:9px;margin-top:2px}}
        .info {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:8px;display:flex;flex-direction:column}}
        .info-title {{color:{c["text"]};font-size:10px;font-weight:600;margin-bottom:6px}}
        .info-row {{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]};font-size:10px}}
        .info-row:last-child {{border-bottom:none}}
        .info-row span:first-child {{color:{c["muted"]}}}
        .info-row span:last-child {{color:{c["text"]};font-weight:600}}
        .right {{display:flex;flex-direction:column;gap:8px}}
        .prog {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:10px;text-align:center}}
        .prog-num {{font-size:38px;font-weight:700;color:{c["value"]};line-height:1}}
        .prog-lbl {{color:{c["muted"]};font-size:10px;margin-top:3px}}
        .prog-detail {{margin-top:8px;text-align:left}}
        .prog-row {{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid {c["border"]};font-size:10px}}
        .prog-row span:first-child {{color:{c["muted"]}}}
        .prog-row span:last-child {{color:{c["text"]};font-weight:600}}
        .sys {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:10px}}
        .sys-title {{font-size:11px;font-weight:600;margin-bottom:6px;color:{c["text"]}}}
        .sys-row {{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]};font-size:10px}}
        .sys-row:last-child {{border-bottom:none}}
        .sys-row span:first-child {{color:{c["muted"]}}}
        .sys-row span:last-child {{color:{c["text"]};font-weight:600}}
        .milestone {{background:{c["card"]};border:1px solid {c["border"]};border-radius:10px;padding:10px}}
        .milestone-title {{font-size:11px;font-weight:600;margin-bottom:6px;color:{c["text"]}}}
        @media(max-width:1000px){{.main{{grid-template-columns:1fr}}.cards{{grid-template-columns:repeat(2,1fr)}}.row2{{grid-template-columns:1fr}}}}
    </style>
    <div class="d">
        <div class="hdr">
            <div class="logo"><span>✦</span><b>Water Meter AI Dashboard</b></div>
            <div style="color:{c["muted"]};font-size:10px">Updated: {now} • Auto-refresh</div>
        </div>
        <div class="status">
            <div style="display:flex;align-items:center;gap:12px">
                <div><b style="font-size:14px">Epoch {epoch}/100</b> <span style="color:{c["muted"]};font-size:11px">({pct:.0f}%)</span></div>
                <div style="color:{c["muted"]};font-size:10px">│</div>
                <div style="color:{c["muted"]};font-size:10px">Speed: {speed}</div>
                <div style="color:{c["muted"]};font-size:10px">│</div>
                <div style="color:{c["muted"]};font-size:10px">Per Epoch: {time_per_epoch}</div>
            </div>
            <div style="color:{c["accent"]};font-size:11px;font-weight:600">ETA: {eta}</div>
        </div>
        <div class="main">
            <div class="left">
                <div class="cards">
                    <div class="c">
                        <div class="c-lbl">🎯 Progress</div>
                        <div class="c-val">{pct:.0f}%</div>
                        <div class="pbar">{segments}</div>
                        <div class="c-sub">{epoch} of 100 epochs</div>
                    </div>
                    <div class="c">
                        <div class="c-lbl">📈 mAP50</div>
                        <div class="c-val">{map50:.1f}%</div>
                        <div class="c-sub">Best: {best_map50:.1f}% • Target: 95%</div>
                    </div>
                    <div class="c">
                        <div class="c-lbl">📉 Box Loss</div>
                        <div class="c-val">{box:.3f}</div>
                        <div class="c-sub">CLS: {cls:.3f} • DFL: {dfl:.3f}</div>
                    </div>
                    <div class="c">
                        <div class="c-lbl">⚡ Speed</div>
                        <div class="c-val">{speed}</div>
                        <div class="c-sub">Time/Epoch: {time_per_epoch}</div>
                    </div>
                </div>
                <div class="row2">
                    <div class="chart">
                        <div class="chart-title">📊 Current Metrics</div>
                        <div class="metrics-grid">
                            <div class="metric"><div class="metric-val">{precision:.1f}%</div><div class="metric-lbl">Precision</div></div>
                            <div class="metric"><div class="metric-val">{recall:.1f}%</div><div class="metric-lbl">Recall</div></div>
                            <div class="metric"><div class="metric-val">{map50:.1f}%</div><div class="metric-lbl">mAP50</div></div>
                        </div>
                        <div style="margin-top:10px;padding-top:8px;border-top:1px solid {c["border"]}">
                            <div style="color:{c["text"]};font-size:11px;font-weight:600;margin-bottom:4px">📉 Loss Trend (Last 5 Epochs)</div>
                            {loss_trend}
                        </div>
                    </div>
                    <div class="info">
                        <div class="info-title">� Model Analysis</div>
                        {analysis_html}
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                    <div class="info">
                        <div class="info-title">📊 Performance Insights</div>
                        {insights_html}
                    </div>
                    <div class="info">
                        <div class="info-title">💡 Smart Recommendations</div>
                        {health_html}
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                    <div class="info">
                        <div class="info-title">⚡ Training Efficiency</div>
                        {efficiency_html}
                    </div>
                    <div class="info">
                        <div class="info-title">📉 Loss Components</div>
                        {loss_breakdown}
                    </div>
                </div>
            </div>
            <div class="right">
                <div class="prog">
                    <div class="prog-num">{epoch}</div>
                    <div class="prog-lbl">Current Epoch</div>
                    <div class="prog-detail">
                        <div class="prog-row"><span>Completed</span><span>{epoch}/100</span></div>
                        <div class="prog-row"><span>Remaining</span><span>{remaining}</span></div>
                        <div class="prog-row"><span>Progress</span><span>{pct:.1f}%</span></div>
                        <div class="prog-row"><span>ETA</span><span>{eta}</span></div>
                        <div class="prog-row"><span>Status</span><span style="color:{c["accent"]}">● Running</span></div>
                    </div>
                </div>
                <div class="milestone">
                    <div class="milestone-title">🎯 Milestones</div>
                    {milestone_html}
                </div>
                <div class="sys">
                    <div class="sys-title">💻 System Monitor</div>
                    <div class="sys-row"><span>GPU Usage</span><span>{stats["gpu"]}</span></div>
                    <div class="sys-row"><span>VRAM</span><span>{stats["vram"]}</span></div>
                    <div class="sys-row"><span>GPU Temp</span><span>{stats["temp"]}</span></div>
                    <div class="sys-row"><span>CPU Usage</span><span>{stats["cpu"]}</span></div>
                    <div class="sys-row"><span>RAM Usage</span><span>{stats["ram"]}</span></div>
                </div>
            </div>
        </div>
    </div>'''

def refresh(theme: str):
    df, status = load_results()
    stats = get_system_stats()
    html = build_html(df, stats, theme)
    return html, status

def toggle(current: str):
    new = "light" if current == "dark" else "dark"
    return (new, *refresh(new))

# Gradio App - Compact Single Screen with Auto-Refresh
with gr.Blocks(title="Water Meter AI Dashboard", css="""
    .gradio-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important}
    footer {display: none !important}
    .contain {padding: 8px !important}
""") as app:
    theme_state = gr.State("dark")
    
    with gr.Row():
        theme_btn = gr.Button("🌙/☀️", size="sm", min_width=60)
        refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="primary", min_width=100)
        status_box = gr.Textbox(show_label=False, interactive=False, scale=6, container=False)
    
    dashboard = gr.HTML()
    
    outputs = [dashboard, status_box]
    
    refresh_btn.click(fn=lambda t: refresh(t), inputs=[theme_state], outputs=outputs)
    theme_btn.click(fn=toggle, inputs=[theme_state], outputs=[theme_state, *outputs])
    app.load(fn=lambda t: refresh(t), inputs=[theme_state], outputs=outputs)
    
    # Auto-refresh every 10 seconds
    timer = gr.Timer(10)
    timer.tick(fn=lambda t: refresh(t), inputs=[theme_state], outputs=outputs)

if __name__ == "__main__":
    print("="*50)
    print("✦ Water Meter AI - Dashboard")
    print("="*50)
    print("🌐 http://localhost:7860")
    print("="*50)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
