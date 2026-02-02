"""
Water Meter AI Detection System - Core Processing Module
=========================================================
Backend processing logic for water meter detection and reading extraction.

This module contains the WaterMeterSystem class that handles:
- Single image inference
- Batch image processing
- Video processing
- Real-time webcam detection

Current Status: MOCK MODE (Dummy Logic for Frontend Development)
TODO: Replace with actual YOLOv8 inference implementation
"""

import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont


class WaterMeterSystem:
    """
    Water Meter AI Detection System - Core Processing Class
    
    Handles all AI inference operations for water meter detection and reading extraction.
    Currently running in MOCK MODE with dummy predictions for frontend development.
    
    Attributes:
        model_path (str): Path to the YOLO model weights
        model (Any): The loaded YOLO model (currently None in mock mode)
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
        self.model = None  # Will be replaced with actual YOLO model
        self.is_mock_mode = True
        
        # Simulate model loading delay
        time.sleep(0.3)
        
        print("Model loaded successfully (Mock Mode).")
        print("⚠️  WARNING: Running in MOCK MODE - predictions are dummy data!")
    
    def predict_single(
        self, 
        image: Image.Image
    ) -> Tuple[Image.Image, Union[Dict[str, Any], str]]:
        """
        Perform single image inference for water meter detection.
        
        Args:
            image (Image.Image): Input PIL Image containing a water meter
        
        Returns:
            Tuple containing:
                - annotated_image (Image.Image): Image with detection bounding boxes
                - json_result (Dict or str): Detection results in JSON format
        """
        # Simulate processing delay
        time.sleep(0.5)
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Get image dimensions
        width, height = annotated_image.size
        
        # Draw mock bounding box (centered, 60% of image size)
        box_width = int(width * 0.6)
        box_height = int(height * 0.4)
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        # Draw rectangle with Navy Blue color (#112D4E)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="#112D4E",
            width=4
        )
        
        # Add label text
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        label = "Water Meter: 12345.67 m³"
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 35), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
            fill="#112D4E"
        )
        
        # Draw label text
        draw.text((x1, y1 - 35), label, fill="white", font=font)
        
        # Create mock JSON result
        json_result = {
            "status": "success",
            "mode": "mock",
            "detections": [
                {
                    "class": "water_meter",
                    "confidence": 0.95,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "reading": {
                        "value": 12345.67,
                        "unit": "m³",
                        "digits": "12345.67"
                    }
                }
            ],
            "image_size": {
                "width": width,
                "height": height
            },
            "processing_time": "0.5s"
        }
        
        # Convert to formatted JSON string
        json_str = json.dumps(json_result, indent=2)
        
        return annotated_image, json_str
    
    def predict_batch(
        self, 
        files: List[str]
    ) -> Tuple[List[Image.Image], pd.DataFrame]:
        """
        Perform batch inference on multiple images.
        
        Args:
            files (List[str]): List of file paths to process
        
        Returns:
            Tuple containing:
                - gallery_images (List[Image.Image]): List of annotated images
                - dataframe (pd.DataFrame): Results table with columns:
                    Filename, Reading, Confidence
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
            
            # Simulate processing delay
            time.sleep(0.2)
            
            # Create annotated image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Get image dimensions
            width, height = annotated_image.size
            
            # Draw mock bounding box
            box_width = int(width * 0.5)
            box_height = int(height * 0.3)
            x1 = (width - box_width) // 2
            y1 = (height - box_height) // 2
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # Draw rectangle with Medium Blue color (#3F72AF)
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="#3F72AF",
                width=3
            )
            
            # Generate mock reading and confidence
            mock_reading = 10000.0 + (idx * 123.45)
            mock_confidence = 0.85 + (idx % 10) * 0.01
            
            # Add to gallery
            gallery_images.append(annotated_image)
            
            # Add to results data
            filename = Path(file_path).name
            results_data.append({
                "Filename": filename,
                "Reading": f"{mock_reading:.2f} m³",
                "Confidence": f"{mock_confidence:.2%}"
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
        Perform real-time inference on webcam frames.
        
        Args:
            frame (np.ndarray): Input frame from webcam (BGR format from OpenCV)
        
        Returns:
            np.ndarray: Annotated frame with detection boxes
        """
        if frame is None:
            return frame
        
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Get frame dimensions
        height, width = annotated_frame.shape[:2]
        
        # Draw mock bounding box (centered)
        box_width = int(width * 0.5)
        box_height = int(height * 0.4)
        x1 = (width - box_width) // 2
        y1 = (height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height
        
        # Draw rectangle in Navy Blue (BGR format: #112D4E -> RGB(17, 45, 78) -> BGR(78, 45, 17))
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color=(78, 45, 17),  # Navy Blue in BGR
            thickness=3
        )
        
        # Add label
        label = "Water Meter: 12345.67"
        confidence = "95.0%"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            2
        )
        
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 15),
            (x1 + text_width + 10, y1),
            color=(78, 45, 17),  # Navy Blue background
            thickness=-1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),  # White text
            2,
            cv2.LINE_AA
        )
        
        # Add confidence badge
        cv2.putText(
            annotated_frame,
            confidence,
            (x2 - 80, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (175, 114, 63),  # Medium Blue in BGR (#3F72AF)
            2,
            cv2.LINE_AA
        )
        
        # Add "MOCK MODE" watermark
        cv2.putText(
            annotated_frame,
            "MOCK MODE",
            (10, height - 10),
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
        return {
            "model_path": self.model_path,
            "is_mock_mode": self.is_mock_mode,
            "status": "Mock Mode - Dummy Predictions",
            "framework": "YOLOv8 (Not Loaded)",
            "classes": ["water_meter", "digit_0-9"],
            "input_size": "640x640",
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
    # Test the mock system
    print("=" * 60)
    print("Water Meter AI Detection System - Core Module Test")
    print("=" * 60)
    
    # Initialize system
    system = create_dummy_system()
    print("\n" + str(system))
    
    # Get model info
    print("\n📊 Model Information:")
    info = system.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Core module initialized successfully!")
    print("Ready for frontend integration.")
    print("=" * 60)
