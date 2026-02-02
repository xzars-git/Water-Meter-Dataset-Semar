"""
Water Meter AI Detection System - Demo Application Package
===========================================================
Standalone demo application with all components included.

Modules:
- app: Gradio web interface
- core: Business logic and ML inference  
- theme: UI design system
"""

__version__ = "1.0.0"
__author__ = "Bapenda Water Meter Detection Project"

# Package-level imports
from app_demo.src.core import WaterMeterSystem
from app_demo.src.theme import get_seaside_theme
from app_demo.src.app import demo

__all__ = [
    "WaterMeterSystem",
    "get_seaside_theme",
    "demo"
]
