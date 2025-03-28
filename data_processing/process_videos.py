import os
import zipfile
import tempfile
from pathlib import Path
import torch
from torchvision import transforms
from config import Config
from .extract_frames import FrameExtractor
from models.captioning_embedding.captioning_pipeline import CaptioningPipeline
from utils.database import CaptionDatabase

class VideoProcessor:
    """Process videos: extract frames, generate captions, and store in database."""
    
    def __init__(self, output_dir='outputs/frames'):
        """Initialize the video processor.
        
        Args:
            output_dir: Directory to save extracted frames
            frames_per_video: Number of frames to extract per video
        """
        self.output_dir = output_dir
        self.frames_per_video = Config.frames_per_video
        self.extractor = FrameExtractor()
        self.db = CaptionDatabase()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single_video(self, video_path, video_id):
        """Process a single video file.
        
        Args:
            video_path: Path to the video file
            video_id: ID for the video
            
        Returns:
            Number of processed frames
        """
        # Initialize captioning pipeline
        captioner = CaptioningPipeline(video_id=video_id)
        # Extract frames
        frames = self.extractor.extract_uniform_frames(video_path, self.frames_per_video)
        # Save frames to disk (optional)
        self.extractor.save_frames(video_id, frames, self.output_dir)
        
        # Process each frame for captioning
        frame_data = []
        for i, (timestamp, pil_img) in enumerate(frames):
            # Convert PIL image to tensor
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(pil_img)
            
            # Get caption
            _, _, caption, _, _ = captioner.run_pipeline(
                data_stream=image_tensor, 
                video_id=video_id, 
                frame_id=timestamp
            )
            
            # Add to batch
            frame_data.append((video_id, timestamp, caption))
            
            # Insert when batch is full or at end of frames
            if len(frame_data) >= Config.batch_size or i == len(frames) - 1:
                self.db.insert_many_captions(frame_data)
                frame_data = []  # Clear the batch
                print(f"Processed and saved batch of frames up to frame {i+1}/{len(frames)}")
                
        return len(frames)
    
    def process_from_zip(self, zip_path):
        """Process the first video from a zip file.
        
        Args:
            zip_path: Path to the zip file containing videos
            
        Returns:
            ID of the processed video, or None if no video was found
        """
        # Create a temporary directory to extract the video
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temp directory: {temp_dir}")
            
            # Extract the first video from the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                video_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.mov', '.mp4', '.avi'))]
                
                if not video_files:
                    print("Error: No video files found in the zip")
                    return None
                
                print(f"Found {len(video_files)} video files")
                
                # Sort the video files to ensure consistent "first" video
                video_files.sort()
                first_video = video_files[0]
                video_id = Path(first_video).stem
                
                print(f"Processing video: {first_video}")
                print(f"Video ID: {video_id}")
                
                # Get file info before extraction
                file_info = zip_ref.getinfo(first_video)
                print(f"Video file size: {file_info.file_size / (1024*1024):.2f} MB")
                
                # Extract the video
                print("Extracting video to temp directory...")
                zip_ref.extract(first_video, temp_dir)
                video_path = os.path.join(temp_dir, first_video)
                
                # Process the extracted video
                self.process_single_video(video_path, video_id)
                
                return video_id
    
    def close(self):
        """Close database connection."""
        self.db.close()