import os
import cv2
import time
import numpy as np
from PIL import Image

class FrameExtractor:
    """Base class for extracting frames from videos."""
    
    def extract_uniform_frames(self, video_path, frames_per_video=40):
        """Extract frames uniformly from a video.
        
        Args:
            video_path: Path to the video file
            frames_per_video: Number of frames to extract
            
        Returns:
            List of (timestamp, PIL Image) tuples
        """
        print(f"Extracting {frames_per_video} frames from: {video_path}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video stats: {fps:.2f} fps, {total_frames} frames, {duration:.2f} seconds")
        
        # Calculate frame interval
        frame_interval = total_frames // frames_per_video
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0 and len(frames) < frames_per_video:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Rotate image if needed (height > width)
                img_array = np.array(pil_img)
                if img_array.shape[0] > img_array.shape[1]:
                    pil_img = pil_img.rotate(90, expand=True)
                    print(f"Rotated frame {len(frames)} from {img_array.shape} to {np.array(pil_img).shape}")
                
                frames.append((frame_count/fps, pil_img))
                print(f"Extracted frame {len(frames)}/{frames_per_video} at {frame_count/fps:.2f}s")
                
            frame_count += 1
            
        cap.release()
        
        print(f"Extracted {len(frames)} frames in {time.time() - start_time:.2f} seconds")
        return frames
    
    def save_frames(self, video_id, frames, output_dir):
        """Save frames to disk.
        
        Args:
            video_id: ID/name of the video
            frames: List of (timestamp, PIL Image) tuples
            output_dir: Directory to save the frames
            
        Returns:
            Path to the frames directory
        """
        print(f"Saving {len(frames)} frames to disk...")
        start_time = time.time()
        
        frames_dir = os.path.join(output_dir, f"{video_id}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, (timestamp, img) in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"{video_id}_frame_{i:04d}_{timestamp:.3f}.jpg")
            img.save(frame_path)
            
        print(f"Saved frames to {frames_dir} in {time.time() - start_time:.2f} seconds")
        return frames_dir