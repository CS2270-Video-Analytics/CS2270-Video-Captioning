import os
from data_processing import VideoProcessor

def main():
    # # Define paths
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # zip_path = os.path.join('/oscar/data/shared/BDD/128.32.162.150/bdd100k/video_parts', 'bdd100k_videos_test_00.zip')
    # output_dir = os.path.join(project_root, 'outputs', 'frames')
    
    # # Initialize processor
    # processor = VideoProcessor(output_dir=output_dir)
    
    # try:
    #     # Process video from zip
    #     video_id = processor.process_from_zip(zip_path)
        
    #     if video_id:
    #         print(f"Successfully processed video {video_id}")
    #     else:
    #         print("No video was processed")
    
    # finally:
    #     # Ensure database is closed
    #     processor.close()
    # Initialize processor
    processor = VideoProcessor(output_dir='outputs/frames')
    
    try:
        # Direct path to your video
        video_path = 'datasets/bdd_videos/bdd100k/videos/test/cd31bcc0-07b8e93f.mov'
        video_id = 'cd31bcc0-07b8e93f'  # video filename without extension
        
        # Process the video directly using process_single_video
        num_frames = processor.process_single_video(video_path, video_id)
        print(f"Successfully processed {num_frames} frames from video {video_id}")
        
    finally:
        # Ensure database is closed
        processor.close()

if __name__ == "__main__":
    main()