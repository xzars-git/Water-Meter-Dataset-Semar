"""
Inference Module - Detection Parsing & Sorting Logic
====================================================
Handles YOLO result parsing, OBB/Box extraction, and left-to-right sorting.
"""

from .parser import DetectionParser

__all__ = ['DetectionParser']
