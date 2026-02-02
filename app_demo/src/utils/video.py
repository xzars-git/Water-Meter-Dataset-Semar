"""
Video Processor - Video I/O and FFmpeg Conversion
=================================================
Handles video reading, writing, and H.264 conversion for browser compatibility.
"""

import cv2
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable
import numpy as np


class VideoProcessor:
    """
    Video processing utilities.
    
    Features:
    - Video reading with OpenCV
    - Video writing with codec selection
    - FFmpeg H.264 conversion for browser playback
    - Progress tracking
    """
    
    @staticmethod
    def get_video_properties(video_path: str) -> dict:
        """
        Get video properties.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with fps, width, height, total_frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        properties = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return properties
    
    @staticmethod
    def create_output_path(input_path: str, suffix: str = "processed") -> Path:
        """
        Create output video path in temp directory.
        
        Args:
            input_path: Input video file path
            suffix: Suffix to add to filename
        
        Returns:
            Output path in temp directory
        """
        input_path = Path(input_path)
        
        # Create temp directory for processed videos
        temp_dir = Path(tempfile.gettempdir()) / "water_meter_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_{suffix}_{timestamp}.mp4"
        
        return temp_dir / output_filename
    
    @staticmethod
    def process_video_frames(
        video_path: str,
        frame_callback: Callable[[np.ndarray], np.ndarray],
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Process video frame by frame with callback function.
        
        Args:
            video_path: Input video path
            frame_callback: Function to process each frame
            output_path: Output video path (auto-generated if None)
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Path to processed video
        """
        print(f"📹 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file: {video_path}")
            return video_path
        
        # Get video properties
        props = VideoProcessor.get_video_properties(video_path)
        fps = props['fps']
        width = props['width']
        height = props['height']
        total_frames = props['total_frames']
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create output path
        if output_path is None:
            output_path = VideoProcessor.create_output_path(video_path)
        else:
            output_path = Path(output_path)
        
        print(f"💾 Output will be saved to: {output_path}")
        
        # Use mp4v codec for initial write (most compatible with OpenCV)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = output_path.with_suffix('.temp.mp4')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Error: Could not initialize video writer")
            cap.release()
            return video_path
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with callback
                processed_frame = frame_callback(frame)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                
                # Progress tracking
                if frame_count % 30 == 0:  # Progress every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    if progress_callback:
                        progress_callback(frame_count, total_frames)
        
        finally:
            cap.release()
            out.release()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Video processing complete (temp): {temp_output}")
        print(f"⏱️  Processing time: {elapsed_time:.2f}s ({frame_count/elapsed_time:.1f} FPS)")
        
        # Convert to H.264 for browser compatibility
        final_path = VideoProcessor.convert_to_h264(temp_output, output_path)
        
        return str(final_path)
    
    @staticmethod
    def convert_to_h264(
        input_path: Path,
        output_path: Path
    ) -> Path:
        """
        Convert video to H.264 codec for browser playback.
        
        Args:
            input_path: Input video path (temp mp4v file)
            output_path: Output video path (H.264)
        
        Returns:
            Path to converted video
        """
        print(f"🔄 Converting to H.264 format for browser playback...")
        
        try:
            # Try to use imageio-ffmpeg (bundled FFmpeg)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"✅ Using bundled FFmpeg: {ffmpeg_exe}")
            except ImportError:
                # Fallback to system FFmpeg
                ffmpeg_exe = shutil.which('ffmpeg')
                if ffmpeg_exe:
                    print(f"✅ Using system FFmpeg: {ffmpeg_exe}")
                else:
                    print("⚠️ FFmpeg not found, using mp4v codec (may not play in browser)")
                    print("   Install with: pip install imageio-ffmpeg")
                    input_path.rename(output_path)
                    return output_path
            
            # Convert to H.264
            result = subprocess.run([
                ffmpeg_exe, '-y', '-i', str(input_path),
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',  # Enable streaming
                str(output_path)
            ], check=True, capture_output=True, text=True)
            
            # Remove temp file
            input_path.unlink()
            print(f"✅ H.264 conversion complete: {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ H.264 conversion failed: {e.stderr}")
            print("   Using original mp4v file (may not play in browser)")
            if input_path.exists():
                input_path.rename(output_path)
        except Exception as e:
            print(f"⚠️ Unexpected error during conversion: {e}")
            if input_path.exists():
                input_path.rename(output_path)
        
        return output_path
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """
        Check if FFmpeg is available (bundled or system).
        
        Returns:
            True if FFmpeg is available
        """
        try:
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
            return True
        except ImportError:
            return shutil.which('ffmpeg') is not None
