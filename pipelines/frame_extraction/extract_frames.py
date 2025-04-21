import os
import cv2
import time
import numpy as np
from PIL import Image

class FrameExtractor:
    """Base class for extracting frames from videos."""
    
    def extract_uniform_frames(self, video_path, frames_per_video=40, specific_frames: list=[]):
        """Extract frames uniformly from a video or at specific timestamps.
        
        Args:
            video_path: Path to the video file
            frames_per_video: Number of frames to extract when using uniform sampling
            specific_frames: Optional list of timestamps (in seconds) to extract
            
        Returns:
            List of (timestamp, PIL Image) tuples
        """
        if not specific_frames:
            print(f"Extracting {len(specific_frames)} specific frames from: {video_path}")
        else:
            print(f"Extracting {frames_per_video} uniform frames from: {video_path}")
        
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video stats: {fps:.2f} fps, {total_frames} frames, {duration:.2f} seconds")
        
        # Check if this is a BDD dataset video
        is_bdd = 'bdd' in video_path.lower()
        if is_bdd:
            print("BDD dataset detected, will apply 90° rotation")
        
        frames = []
        
        if specific_frames:
            # Process specific timestamps
            for timestamp in specific_frames:
                # Convert timestamp to frame index
                frame_index = int(timestamp * fps)
                
                # Seek to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Apply rotation only for BDD dataset
                    if is_bdd:
                        pil_img = pil_img.rotate(90, expand=True)
                        print(f"Rotated BDD frame at {timestamp:.2f}s")
                    
                    frames.append((timestamp, pil_img))
                    print(f"Extracted frame at timestamp {timestamp:.2f}s (frame {frame_index})")
                else:
                    print(f"Failed to extract frame at timestamp {timestamp:.2f}s")
        else:
            # Original uniform sampling logic
            frame_interval = total_frames // frames_per_video
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0 and len(frames) < frames_per_video:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Apply rotation only for BDD dataset
                    if is_bdd:
                        pil_img = pil_img.rotate(90, expand=True)
                        print(f"Rotated BDD frame {len(frames)}")
                    
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