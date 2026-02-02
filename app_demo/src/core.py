"""
Water Meter AI Detection System - Core Processing Module
=========================================================
Backend processing logic for water meter detection and reading extraction.

This module contains the WaterMeterSystem class that handles:
- Single image inference
- Batch image processing
- Video processing
- Real-time webcam detection

Status: PRODUCTION MODE with Left-to-Right Digit Sorting
"""

import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


class WaterMeterSystem:
    """
    Water Meter AI Detection System - Core Processing Class
    
    Handles all AI inference operations for water meter detection and reading extraction.
    Implements left-to-right digit sorting for accurate reading extraction.
    
    Attributes:
        model_path (str): Path to the YOLO model weights
        model (YOLO): The loaded YOLO model
        is_mock_mode (bool): Flag indicating if running in mock mode
    """
    
    def __init__(self, model_path: str) -> None:
        """
        Initialize the Water Meter System with a model.
        
        Args:
            model_path (str): Path to the YOLOv8 model weights file
        """
        print(f"Loading model from {model_path}...")
        self.model_path = model_path
        self.is_mock_mode = False
        
        try:
            # Load the YOLO model
            self.model = YOLO(model_path)
            print("Model loaded successfully (Production Mode).")
            print("✅ Ready for inference with left-to-right digit sorting!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to MOCK MODE")
            self.model = None
            self.is_mock_mode = True
    
    def predict_single(
        self, 
        image: Image.Image
    ) -> Tuple[Image.Image, Union[Dict[str, Any], str]]:
        """
        Perform single image inference for water meter detection with left-to-right digit sorting.
        
        Args:
            image (Image.Image): Input PIL Image containing a water meter
        
        Returns:
            Tuple containing:
                - annotated_image (Image.Image): Image with detection bounding boxes and reading banner
                - json_result (Dict or str): Detection results in JSON format
        """
        start_time = time.time()
        
        # Step A: Inference
        # Convert PIL Image to numpy array (RGB)
        img_array = np.array(image)
        
        # Run YOLO inference with lower confidence threshold for better detection
        # Try 0.05 for very sensitive detection
        results = self.model.predict(source=img_array, conf=0.05, verbose=False)
        
        # Step B: Extraction & Sorting (The Fix)
        detections = []
        
        # Check if this is OBB model or regular detection model
        if len(results) > 0:
            # Try OBB first (for rotated bounding boxes)
            if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
                print("⚠️ Detected OBB model - using OBB format")
                for obb in results[0].obb:
                    # Get rotated box coordinates (4 corners)
                    xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy()  # Shape: (4, 2) - 4 corners
                    
                    # Calculate bounding box from 4 corners
                    x_coords = xyxyxyxy[:, 0]
                    y_coords = xyxyxyxy[:, 1]
                    x1, y1 = float(x_coords.min()), float(y_coords.min())
                    x2, y2 = float(x_coords.max()), float(y_coords.max())
                    
                    # Extract class id and name
                    cls_id = int(obb.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    # Calculate x_center for sorting
                    x_center = (x1 + x2) / 2
                    
                    detections.append({
                        "class": class_name,
                        "x_center": x_center,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(obb.conf[0])
                    })
            
            # Fallback to regular boxes
            elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                print("✅ Using regular bounding boxes")
                for box in results[0].boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Extract class id and name
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    # Calculate x_center for sorting (key for left-to-right ordering)
                    x_center = (x1 + x2) / 2
                    
                    # Store detection data
                    detections.append({
                        "class": class_name,
                        "x_center": x_center,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(box.conf[0])
                    })
        
        # Sort detections by x_center (left-to-right)
        detections.sort(key=lambda x: x['x_center'])
        
        # Concatenate class names to form final reading
        # Filter only digit classes and extract the digit number
        digit_readings = []
        for d in detections:
            class_name = d['class']
            # Only process digit classes, skip border
            if class_name.startswith('digit_'):
                # Extract digit number from 'digit_X' format
                digit = class_name.replace('digit_', '')
                digit_readings.append(digit)
        
        final_reading = "".join(digit_readings)
        
        # Step C: Visualization (License Plate Style)
        # Convert PIL Image to OpenCV format (BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = img_bgr.shape[:2]
        
        # Draw bounding boxes for each digit
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_name = det['class']
            
            # Draw rectangle in Navy Blue (#112D4E -> BGR: 78, 45, 17)
            cv2.rectangle(
                img_bgr,
                (x1, y1),
                (x2, y2),
                color=(78, 45, 17),  # Navy Blue in BGR
                thickness=2
            )
            
            # Add confidence label below each box
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            # Draw label background
            cv2.rectangle(
                img_bgr,
                (x1, y2),
                (x1 + text_width + 6, y2 + text_height + 6),
                color=(78, 45, 17),
                thickness=-1
            )
            
            # Draw label text
            cv2.putText(
                img_bgr,
                label,
                (x1 + 3, y2 + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        # Draw Result Banner (License Plate Style)
        if final_reading:
            # Banner dimensions
            banner_text = f"Reading: {final_reading}"
            font_scale = 1.2
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                banner_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness
            )
            
            # Create filled rectangle at top-left
            banner_height = text_height + 30
            banner_width = text_width + 40
            
            cv2.rectangle(
                img_bgr,
                (10, 10),
                (10 + banner_width, 10 + banner_height),
                color=(78, 45, 17),  # Navy Blue background
                thickness=-1
            )
            
            # Draw border
            cv2.rectangle(
                img_bgr,
                (10, 10),
                (10 + banner_width, 10 + banner_height),
                color=(175, 114, 63),  # Medium Blue border (#3F72AF in BGR)
                thickness=2
            )
            
            # Draw text (White)
            cv2.putText(
                img_bgr,
                banner_text,
                (30, 30 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
        
        # Step D: Return
        # Convert back to PIL Image (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil_image = Image.fromarray(img_rgb)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create JSON result
        json_result = {
            "status": "success",
            "mode": "production",
            "reading": final_reading,
            "detections": [
                {
                    "class": d['class'],
                    "confidence": d['conf'],
                    "bbox": {
                        "x1": d['bbox'][0],
                        "y1": d['bbox'][1],
                        "x2": d['bbox'][2],
                        "y2": d['bbox'][3]
                    },
                    "x_center": d['x_center']
                }
                for d in detections
            ],
            "image_size": {
                "width": width,
                "height": height
            },
            "processing_time": f"{processing_time:.3f}s",
            "num_detections": len(detections)
        }
        
        # Convert to formatted JSON string
        json_str = json.dumps(json_result, indent=2)
        
        return annotated_pil_image, json_str
    
    def predict_batch(
        self, 
        files: List[str]
    ) -> Tuple[List[Image.Image], pd.DataFrame]:
        """
        Perform batch inference on multiple images with left-to-right digit sorting.
        
        Args:
            files (List[str]): List of file paths to process
        
        Returns:
            Tuple containing:
                - gallery_images (List[Image.Image]): List of annotated images
                - dataframe (pd.DataFrame): Results table with columns:
                    Filename, Reading, Confidence (Average)
        """
        gallery_images = []
        results_data = []
        
        for idx, file_path in enumerate(files):
            # Load image
            try:
                image = Image.open(file_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            # Convert PIL Image to numpy array (RGB)
            img_array = np.array(image)
            
            # Run YOLO inference with lower confidence threshold
            results = self.model.predict(source=img_array, conf=0.10, verbose=False)
            
            # Extract and sort detections (left-to-right)
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    x_center = (x1 + x2) / 2
                    
                    detections.append({
                        "class": class_name,
                        "x_center": x_center,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": float(box.conf[0])
                    })
            
            # Sort by x_center (left-to-right)
            detections.sort(key=lambda x: x['x_center'])
            
            # Concatenate to form final reading
            final_reading = "".join([d['class'] for d in detections])
            
            # Calculate average confidence
            avg_confidence = np.mean([d['conf'] for d in detections]) if detections else 0.0
            
            # Visualize
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(
                    img_bgr,
                    (x1, y1),
                    (x2, y2),
                    color=(78, 45, 17),  # Navy Blue
                    thickness=2
                )
            
            # Draw result banner
            if final_reading:
                banner_text = f"{final_reading}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    banner_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    2
                )
                
                cv2.rectangle(
                    img_bgr,
                    (5, 5),
                    (15 + text_width, 15 + text_height),
                    color=(78, 45, 17),
                    thickness=-1
                )
                
                cv2.putText(
                    img_bgr,
                    banner_text,
                    (10, 10 + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Convert back to PIL
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil_image = Image.fromarray(img_rgb)
            
            # Add to gallery
            gallery_images.append(annotated_pil_image)
            
            # Add to results data
            filename = Path(file_path).name
            results_data.append({
                "Filename": filename,
                "Reading": final_reading if final_reading else "No Detection",
                "Confidence": f"{avg_confidence:.2%}"
            })
        
        # Create DataFrame
        df = pd.DataFrame(results_data)
        
        return gallery_images, df
    
    def process_video(
        self, 
        video_path: str
    ) -> str:
        """
        Process video file for water meter detection and tracking.
        
        Args:
            video_path (str): Path to input video file
        
        Returns:
            str: Path to processed video (currently returns input path in mock mode)
        
        TODO: Implement YOLOv8 Tracking logic here later
        - Load video with cv2.VideoCapture
        - Run YOLO inference on each frame
        - Apply tracking algorithm
        - Draw bounding boxes and IDs
        - Save processed video
        - Return path to processed video
        """
        print(f"Processing video: {video_path}")
        print("⚠️  MOCK MODE: Returning original video without processing")
        
        # Simulate processing delay
        time.sleep(1.0)
        
        # TODO: Implement YOLOv8 Tracking logic here later
        # For now, just return the input video path (pass-through)
        # This allows testing the UI video player
        
        return video_path
    
    def predict_webcam(
        self, 
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Perform real-time inference on webcam frames with left-to-right digit sorting.
        
        Args:
            frame (np.ndarray): Input frame from webcam (BGR format from OpenCV)
        
        Returns:
            np.ndarray: Annotated frame with detection boxes and sorted reading
        """
        if frame is None:
            return frame
        
        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference with lower confidence threshold
        results = self.model.predict(source=frame_rgb, conf=0.10, verbose=False)
        
        # Extract and sort detections (left-to-right)
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                x_center = (x1 + x2) / 2
                
                detections.append({
                    "class": class_name,
                    "x_center": x_center,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(box.conf[0])
                })
        
        # Sort by x_center (left-to-right)
        detections.sort(key=lambda x: x['x_center'])
        
        # Concatenate to form final reading
        final_reading = "".join([d['class'] for d in detections])
        
        # Create annotated frame
        annotated_frame = frame.copy()
        height, width = annotated_frame.shape[:2]
        
        # Draw bounding boxes for each digit
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # Draw rectangle in Navy Blue
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color=(78, 45, 17),  # Navy Blue in BGR
                thickness=2
            )
            
            # Add confidence badge
            cv2.putText(
                annotated_frame,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (175, 114, 63),  # Medium Blue
                2,
                cv2.LINE_AA
            )
        
        # Draw result banner with sorted reading
        if final_reading:
            banner_text = f"Reading: {final_reading}"
            font_scale = 1.0
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(
                banner_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness
            )
            
            # Draw filled rectangle (banner background)
            banner_height = text_height + 20
            banner_width = text_width + 30
            
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (10 + banner_width, 10 + banner_height),
                color=(78, 45, 17),  # Navy Blue
                thickness=-1
            )
            
            # Draw border
            cv2.rectangle(
                annotated_frame,
                (10, 10),
                (10 + banner_width, 10 + banner_height),
                color=(175, 114, 63),  # Medium Blue border
                thickness=2
            )
            
            # Draw text
            cv2.putText(
                annotated_frame,
                banner_text,
                (25, 25 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
        else:
            # No detections
            cv2.putText(
                annotated_frame,
                "No digits detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red text
                2,
                cv2.LINE_AA
            )
        
        return annotated_frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        if self.model is not None:
            return {
                "model_path": self.model_path,
                "is_mock_mode": self.is_mock_mode,
                "status": "Production Mode - Real Inference",
                "framework": "YOLOv8/YOLO11 (Ultralytics)",
                "classes": list(self.model.names.values()),
                "num_classes": len(self.model.names),
                "version": "1.0.0-production"
            }
        else:
            return {
                "model_path": self.model_path,
                "is_mock_mode": True,
                "status": "Mock Mode - Model Not Loaded",
                "framework": "N/A",
                "classes": [],
                "num_classes": 0,
                "version": "1.0.0-mock"
            }
    
    def __repr__(self) -> str:
        """String representation of the WaterMeterSystem instance."""
        return (
            f"WaterMeterSystem(model_path='{self.model_path}', "
            f"mock_mode={self.is_mock_mode})"
        )


# Utility functions for testing
def create_dummy_system(model_path: str = "yolo11n.pt") -> WaterMeterSystem:
    """
    Create a WaterMeterSystem instance with dummy model.
    
    Args:
        model_path (str): Path to model file (default: yolo11n.pt)
    
    Returns:
        WaterMeterSystem: Initialized system instance
    """
    return WaterMeterSystem(model_path)


if __name__ == "__main__":
    # Test the system
    print("=" * 60)
    print("Water Meter AI Detection System - Core Module Test")
    print("=" * 60)
    
    # Initialize system with actual model
    model_path = "../../assets/models/water_meter_model.pt"  # Adjust path as needed
    system = create_dummy_system(model_path)
    print("\n" + str(system))
    
    # Get model info
    print("\n📊 Model Information:")
    info = system.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Core module initialized successfully!")
    print("Ready for frontend integration with LEFT-TO-RIGHT sorting.")
    print("=" * 60)
