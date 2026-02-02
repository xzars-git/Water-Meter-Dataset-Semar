"""
Water Meter AI Detection System - Demo Entry Point
===================================================
Standalone demo application with all dependencies included.

Usage:
    cd app_demo
    python main.py

The application will be available at http://localhost:7860
"""

import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add app_demo to path
app_demo_path = os.path.dirname(os.path.abspath(__file__))
if app_demo_path not in sys.path:
    sys.path.insert(0, app_demo_path)

# Import the application
from app_demo.src.app import demo, CUSTOM_CSS, system
from app_demo.src.theme import get_seaside_theme


def main():
    """Launch the Water Meter AI Detection System"""
    print("\n" + "=" * 70)
    print("🚀 Launching Water Meter AI Detection System - DEMO")
    print("=" * 70)
    print("\n📊 System Information:")
    print(f"   Version: 1.0.0 (Demo)")
    print(f"   Theme: Modern Dark with Roboto Font")
    print(f"   Model: {system.model_path}")
    print(f"   Mode: {'MOCK (Development)' if system.is_mock_mode else 'PRODUCTION'}")
    print(f"   CSS: Loaded from assets/styles/custom.css")
    print("\n💡 Access the application at: http://localhost:7860")
    print("   (or the URL shown below)\n")
    print("=" * 70 + "\n")
    
    # Launch with all configurations
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        theme=get_seaside_theme(),
        css=CUSTOM_CSS
    )


if __name__ == "__main__":
    main()
