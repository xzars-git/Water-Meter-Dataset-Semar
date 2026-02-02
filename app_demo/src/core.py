"""
Water Meter AI Detection System - Core Processing Module (Refactored)
=====================================================================
Backend processing logic for water meter detection and reading extraction.

This module contains the WaterMeterSystem class that orchestrates:
- Single image inference
- Batch image processing
- Video processing
- Real-time webcam detection

Architecture:
- Uses modular components from inference/, visualization/, and utils/
- Clean separation of concerns for better maintenance and testing

Status: PRODUCTION MODE with Modular Architecture
"""

import time
import json
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from ultralytics import YOLO

# Import custom modules
from .inference import DetectionParser
from .visualization import FrameAnnotator
from .utils import VideoProcessor


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
        results = self.model.predict(source=img_array, conf=0.05, verbose=False)
        
        # Step B: Extraction & Sorting (Using DetectionParser module)
        detections = DetectionParser.parse_results(results, self.model)
        detections = DetectionParser.sort_left_to_right(detections)
        final_reading = DetectionParser.extract_reading(detections)
        
        # Step C: Visualization (Using FrameAnnotator module)
        # Convert PIL Image to OpenCV format (BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = img_bgr.shape[:2]
        
        # Annotate image with boxes and banner
        img_bgr = FrameAnnotator.annotate_full(
            img_bgr,
            detections,
            final_reading,
            show_labels=True,
            banner_style="full"
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
            
            # Run YOLO inference
            results = self.model.predict(source=img_array, conf=0.05, verbose=False)
            
            # Extract and sort detections (Using DetectionParser module)
            detections = DetectionParser.parse_results(results, self.model)
            detections = DetectionParser.sort_left_to_right(detections)
            final_reading = DetectionParser.extract_reading(detections)
            avg_confidence = DetectionParser.calculate_avg_confidence(detections)
            
            # Visualize (Using FrameAnnotator module)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Draw bounding boxes
            img_bgr = FrameAnnotator.draw_bounding_boxes(img_bgr, detections, show_labels=False)
            
            # Draw simple banner
            if final_reading:
                img_bgr = FrameAnnotator.draw_simple_banner(img_bgr, final_reading)
            
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
        Process video file for water meter detection with left-to-right digit sorting.
        
        Args:
            video_path (str): Path to input video file
        
        Returns:
            str: Path to processed video file
        """
        # Use VideoProcessor module for video handling
        output_path = VideoProcessor.process_video_frames(
            video_path,
            frame_callback=self.predict_webcam,
            output_path=None,
            progress_callback=None
        )
        
        return output_path
    
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
        
        # Run YOLO inference
        results = self.model.predict(source=frame_rgb, conf=0.05, verbose=False)
        
        # Extract and sort detections (Using DetectionParser module)
        detections = DetectionParser.parse_results(results, self.model)
        detections = DetectionParser.sort_left_to_right(detections)
        final_reading = DetectionParser.extract_reading(detections)
        
        # Annotate frame (Using FrameAnnotator module)
        if final_reading:
            annotated = FrameAnnotator.draw_bounding_boxes(frame, detections, show_labels=False)
            annotated = FrameAnnotator.draw_confidence_badge(annotated, detections)
            annotated = FrameAnnotator.draw_result_banner(annotated, final_reading, font_scale=1.0)
        else:
            annotated = FrameAnnotator.draw_no_detection_message(frame)
        
        return annotated
    
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
                "status": "Production Mode - Modular Architecture",
                "framework": "YOLOv8/YOLO11 (Ultralytics)",
                "classes": list(self.model.names.values()),
                "num_classes": len(self.model.names),
                "version": "2.0.0-modular",
                "modules": {
                    "inference": "DetectionParser (OBB + Box parsing)",
                    "visualization": "FrameAnnotator (Navy Blue theme)",
                    "utils": "VideoProcessor (FFmpeg H.264 conversion)"
                }
            }
        else:
            return {
                "model_path": self.model_path,
                "is_mock_mode": True,
                "status": "Mock Mode - Model Not Loaded",
                "framework": "N/A",
                "classes": [],
                "num_classes": 0,
                "version": "2.0.0-mock"
            }
    
    def __repr__(self) -> str:
        """String representation of the WaterMeterSystem instance."""
        return (
            f"WaterMeterSystem(model_path='{self.model_path}', "
            f"mock_mode={self.is_mock_mode}, "
            f"architecture='modular')"
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
    print("=" * 70)
    print("Water Meter AI Detection System - Modular Architecture Test")
    print("=" * 70)
    
    # Initialize system with actual model
    model_path = "../../assets/models/water_meter_model.pt"  # Adjust path as needed
    system = create_dummy_system(model_path)
    print("\n" + str(system))
    
    # Get model info
    print("\n📊 Model Information:")
    info = system.get_model_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    - {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ Core module initialized successfully!")
    print("📦 Modular architecture with separated concerns:")
    print("   - inference/: Detection parsing & sorting")
    print("   - visualization/: Annotation & drawing")
    print("   - utils/: Video processing & helpers")
    print("   - preprocessing/: Ready for enhancement pipeline")
    print("=" * 70)
