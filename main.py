import os
from data_processing import VideoProcessor

def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join('/oscar/data/shared/BDD/128.32.162.150/bdd100k/video_parts', 'bdd100k_videos_test_00.zip')
    output_dir = os.path.join(project_root, 'outputs', 'frames')
    
    # Initialize processor
    processor = VideoProcessor(output_dir=output_dir)
    
    try:
        # Process video from zip
        video_id = processor.process_from_zip(zip_path)
        
        if video_id:
            print(f"Successfully processed video {video_id}")
        else:
            print("No video was processed")
    
    finally:
        # Ensure database is closed
        processor.close()

if __name__ == "__main__":
    main()