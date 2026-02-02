"""
Frame Annotator - Draw Bounding Boxes and Result Banners
========================================================
Implements license plate style visualization with Navy Blue theme.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional


class FrameAnnotator:
    """
    Annotate frames/images with detection results.
    
    Features:
    - Navy Blue bounding boxes (#112D4E)
    - Result banner with reading
    - Confidence labels
    - License plate inspired design
    """
    
    # Color scheme (BGR format for OpenCV)
    NAVY_BLUE = (78, 45, 17)        # #112D4E
    MEDIUM_BLUE = (175, 114, 63)    # #3F72AF
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    
    @staticmethod
    def draw_bounding_boxes(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
            show_labels: Whether to show confidence labels
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_name = det['class']
            
            # Draw rectangle in Navy Blue
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color=FrameAnnotator.NAVY_BLUE,
                thickness=2
            )
            
            # Add confidence label if requested
            if show_labels:
                label = f"{class_name}: {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x1, y2),
                    (x1 + text_width + 6, y2 + text_height + 6),
                    color=FrameAnnotator.NAVY_BLUE,
                    thickness=-1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, y2 + text_height + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    FrameAnnotator.WHITE,
                    1,
                    cv2.LINE_AA
                )
        
        return annotated
    
    @staticmethod
    def draw_result_banner(
        frame: np.ndarray,
        reading: str,
        position: str = "top-left",
        font_scale: float = 1.2
    ) -> np.ndarray:
        """
        Draw result banner with final reading.
        
        Args:
            frame: Input frame (BGR format)
            reading: Final reading string
            position: Banner position ("top-left", "top-center")
            font_scale: Font size multiplier
        
        Returns:
            Annotated frame with banner
        """
        if not reading:
            return frame
        
        annotated = frame.copy()
        banner_text = f"Reading: {reading}"
        font_thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(
            banner_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )
        
        # Calculate banner dimensions
        banner_height = text_height + 30
        banner_width = text_width + 40
        
        # Position calculation
        if position == "top-left":
            x_start, y_start = 10, 10
        elif position == "top-center":
            height, width = frame.shape[:2]
            x_start = (width - banner_width) // 2
            y_start = 10
        else:
            x_start, y_start = 10, 10
        
        # Draw filled rectangle (banner background)
        cv2.rectangle(
            annotated,
            (x_start, y_start),
            (x_start + banner_width, y_start + banner_height),
            color=FrameAnnotator.NAVY_BLUE,
            thickness=-1
        )
        
        # Draw border
        cv2.rectangle(
            annotated,
            (x_start, y_start),
            (x_start + banner_width, y_start + banner_height),
            color=FrameAnnotator.MEDIUM_BLUE,
            thickness=2
        )
        
        # Draw text
        cv2.putText(
            annotated,
            banner_text,
            (x_start + 20, y_start + 15 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            FrameAnnotator.WHITE,
            font_thickness,
            cv2.LINE_AA
        )
        
        return annotated
    
    @staticmethod
    def draw_simple_banner(
        frame: np.ndarray,
        reading: str,
        font_scale: float = 0.8
    ) -> np.ndarray:
        """
        Draw simple compact banner (for batch processing).
        
        Args:
            frame: Input frame (BGR format)
            reading: Final reading string
            font_scale: Font size multiplier
        
        Returns:
            Annotated frame with compact banner
        """
        if not reading:
            return frame
        
        annotated = frame.copy()
        (text_width, text_height), baseline = cv2.getTextSize(
            reading,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            2
        )
        
        # Draw background
        cv2.rectangle(
            annotated,
            (5, 5),
            (15 + text_width, 15 + text_height),
            color=FrameAnnotator.NAVY_BLUE,
            thickness=-1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            reading,
            (10, 10 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            FrameAnnotator.WHITE,
            2,
            cv2.LINE_AA
        )
        
        return annotated
    
    @staticmethod
    def draw_confidence_badge(
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw confidence badges above bounding boxes.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
        
        Returns:
            Annotated frame with confidence badges
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # Draw confidence text above box
            cv2.putText(
                annotated,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                FrameAnnotator.MEDIUM_BLUE,
                2,
                cv2.LINE_AA
            )
        
        return annotated
    
    @staticmethod
    def draw_no_detection_message(frame: np.ndarray) -> np.ndarray:
        """
        Draw "No digits detected" message.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Annotated frame with message
        """
        annotated = frame.copy()
        
        cv2.putText(
            annotated,
            "No digits detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            FrameAnnotator.RED,
            2,
            cv2.LINE_AA
        )
        
        return annotated
    
    @staticmethod
    def annotate_full(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        reading: str,
        show_labels: bool = True,
        banner_style: str = "full"
    ) -> np.ndarray:
        """
        Full annotation pipeline: boxes + banner + labels.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
            reading: Final reading string
            show_labels: Whether to show confidence labels
            banner_style: "full" or "simple"
        
        Returns:
            Fully annotated frame
        """
        # Draw bounding boxes
        annotated = FrameAnnotator.draw_bounding_boxes(
            frame, 
            detections, 
            show_labels=show_labels
        )
        
        # Draw result banner
        if reading:
            if banner_style == "full":
                annotated = FrameAnnotator.draw_result_banner(annotated, reading)
            else:
                annotated = FrameAnnotator.draw_simple_banner(annotated, reading)
        else:
            annotated = FrameAnnotator.draw_no_detection_message(annotated)
        
        return annotated
