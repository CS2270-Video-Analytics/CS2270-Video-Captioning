import os
import zipfile
import tempfile
from pathlib import Path
import torch
from torchvision import transforms
from config.config import Config
from .extract_frames import FrameExtractor
from pipelines.captioning_embedding.captioning_pipeline import CaptioningPipeline
from tqdm import tqdm
import asyncio
import pdb

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
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def process_single_video(self, video_path:str, video_id:str, captioning_pipeline, curr_vec_idx:int, new_attributes_dict: dict={}, specific_frames: list = [], reboot: bool = False):
        """Process a single video file.
        
        Args:
            video_path: Path to the video file
            video_id: ID for the video
            
        Returns:
            Number of processed frames
        """
        # Extract frames
        frames = self.extractor.extract_uniform_frames(video_path, self.frames_per_video, specific_frames=specific_frames)
        # Save frames to disk (optional)
        if Config.save_frames:
            self.extractor.save_frames(video_id, frames, self.output_dir)

        sql_batch = []
        vector_batch = []
        batch_size = Config.batch_size
        for batch_start in range(0, len(frames), batch_size):
            tasks = []
            batch_end = min(batch_start + batch_size, len(frames))
            for i in range(batch_start, batch_end):
                timestamp, pil_img = frames[i]
                to_tensor = transforms.ToTensor()
                image_tensor = to_tensor(pil_img)
                task = captioning_pipeline.run_pipeline(
                    data_stream=image_tensor,
                    video_id=video_id,
                    frame_id=timestamp,
                    new_attributes_dict=new_attributes_dict,
                    reboot=reboot
                )
                tasks.append(task)
                curr_vec_idx += 1

            # Wait for all tasks in the current batch to complete
            results = await asyncio.gather(*tasks)
            
            for result in results:
                video_id, frame_id, description, object_list, image_embedding, focused_description = result
                
                sql_batch.append((video_id, frame_id, description, curr_vec_idx, focused_description) if not reboot else (video_id, frame_id, focused_description))
                vector_batch.append(image_embedding)

            # Yield the batch results
            yield (sql_batch, vector_batch)
            sql_batch = []
            vector_batch = []

            print(f"Processed and saved batch of frames up to frame {batch_end}/{len(frames)}")
