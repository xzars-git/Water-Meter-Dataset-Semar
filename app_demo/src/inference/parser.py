"""
Detection Parser - Extract & Sort YOLO Detections
=================================================
Handles both OBB (Oriented Bounding Box) and regular bounding box formats.
Implements left-to-right sorting for sequential digit reading.
"""

import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO


class DetectionParser:
    """
    Parse YOLO detection results and sort detections spatially.
    
    Supports:
    - OBB (Oriented Bounding Box) format: xyxyxyxy (4 corners)
    - Regular bounding box format: xyxy
    - Left-to-right sorting based on x_center
    """
    
    @staticmethod
    def parse_results(results, model: YOLO) -> List[Dict[str, Any]]:
        """
        Parse YOLO results and extract detections.
        
        Args:
            results: YOLO prediction results
            model: YOLO model instance (for class names)
        
        Returns:
            List of detection dictionaries with keys:
                - class: Class name
                - x_center: Center X coordinate (for sorting)
                - bbox: [x1, y1, x2, y2]
                - conf: Confidence score
        """
        detections = []
        
        if len(results) == 0:
            return detections
        
        # Try OBB first (for rotated bounding boxes)
        if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
            print("⚠️ Detected OBB model - using OBB format")
            detections = DetectionParser._parse_obb(results[0].obb, model)
        
        # Fallback to regular boxes
        elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
            print("✅ Using regular bounding boxes")
            detections = DetectionParser._parse_boxes(results[0].boxes, model)
        
        return detections
    
    @staticmethod
    def _parse_obb(obb_results, model: YOLO) -> List[Dict[str, Any]]:
        """Parse OBB (Oriented Bounding Box) results."""
        detections = []
        
        for obb in obb_results:
            # Get rotated box coordinates (4 corners)
            xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy()  # Shape: (4, 2) - 4 corners
            
            # Calculate axis-aligned bounding box from 4 corners
            x_coords = xyxyxyxy[:, 0]
            y_coords = xyxyxyxy[:, 1]
            x1, y1 = float(x_coords.min()), float(y_coords.min())
            x2, y2 = float(x_coords.max()), float(y_coords.max())
            
            # Extract class id and name
            cls_id = int(obb.cls[0])
            class_name = model.names[cls_id]
            
            # Calculate x_center for sorting
            x_center = (x1 + x2) / 2
            
            detections.append({
                "class": class_name,
                "x_center": x_center,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(obb.conf[0])
            })
        
        return detections
    
    @staticmethod
    def _parse_boxes(box_results, model: YOLO) -> List[Dict[str, Any]]:
        """Parse regular bounding box results."""
        detections = []
        
        for box in box_results:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Extract class id and name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Calculate x_center for sorting (key for left-to-right ordering)
            x_center = (x1 + x2) / 2
            
            # Store detection data
            detections.append({
                "class": class_name,
                "x_center": x_center,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(box.conf[0])
            })
        
        return detections
    
    @staticmethod
    def sort_left_to_right(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort detections from left to right based on x_center.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Sorted list of detections (left to right)
        """
        return sorted(detections, key=lambda x: x['x_center'])
    
    @staticmethod
    def extract_reading(detections: List[Dict[str, Any]]) -> str:
        """
        Extract final reading by concatenating digit classes.
        
        Args:
            detections: List of detection dictionaries (should be sorted)
        
        Returns:
            Final reading string (e.g., "03822")
        """
        digit_readings = []
        
        for d in detections:
            class_name = d['class']
            # Only process digit classes, skip border or other classes
            if class_name.startswith('digit_'):
                # Extract digit number from 'digit_X' format
                digit = class_name.replace('digit_', '')
                digit_readings.append(digit)
        
        return "".join(digit_readings)
    
    @staticmethod
    def calculate_avg_confidence(detections: List[Dict[str, Any]]) -> float:
        """
        Calculate average confidence across all detections.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Average confidence score (0.0 if no detections)
        """
        if not detections:
            return 0.0
        
        confidences = [d['conf'] for d in detections]
        return float(np.mean(confidences))
