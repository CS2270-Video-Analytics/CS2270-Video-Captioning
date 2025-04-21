import os
import sys
import time
from pipelines.frame_extraction.extract_frames import FrameExtractor

def test_specific_timestamps():
    """Test extracting frames at specific timestamps."""
    # Path to a video file - update with an actual video path
    video_path = "datasets/bdd/00a2f5b6-d4217a96.mov"
    
    # Specific timestamps to extract (in seconds)
    timestamps = [1.0, 5.5, 10.0, 15.7]
    
    # Create an instance of FrameExtractor
    extractor = FrameExtractor()
    
    # Extract frames at specific timestamps
    print(f"Extracting frames at timestamps: {timestamps}")
    start_time = time.time()
    frames = extractor.extract_uniform_frames(
        video_path=video_path,
        specific_frames=timestamps
    )
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\nExtracted {len(frames)} frames in {elapsed:.2f} seconds")
    for i, (ts, img) in enumerate(frames):
        print(f"Frame {i+1}: timestamp={ts:.2f}s, size={img.size}")
    
    # Save the extracted frames
    output_dir = "extracted_test_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (ts, img) in enumerate(frames):
        output_path = os.path.join(output_dir, f"frame_{i+1}_at_{ts:.2f}s.jpg")
        img.save(output_path)
        print(f"Saved frame to {output_path}")
    
    print(f"\nAll frames saved to {output_dir}")

def test_different_datasets():
    """Test extracting frames from different datasets with appropriate rotation."""
    
    # List to store test videos from different datasets
    test_videos = [
        # {"path": "datasets/bdd/00a2f5b6-d4217a96.mov", "name": "BDD"},
        {"path": "datasets/charades/013SD.mp4", "name": "Charades"},
        # {"path": "datasets/ucf101/v_Archery_g08_c01.mp4", "name": "UCF101"}
    ]
    
    extractor = FrameExtractor()
    
    for video in test_videos:
        print(f"\nTesting {video['name']} dataset video: {video['path']}")
        
        # Extract only one frame for testing
        frames = extractor.extract_uniform_frames(
            video_path=video['path'],
            frames_per_video=10
        )
        
        # Save sample frame
        if frames:
            os.makedirs("dataset_tests", exist_ok=True)
            for i in range(len(frames)):
                frames[0][1].save(f"dataset_tests/{video['name']}_sample_{i}.jpg")
                print(f"Saved sample frame from {video['name']} to dataset_tests/{video['name']}_sample_{i}.jpg")

if __name__ == "__main__":
    # Choose which test to run
    # test_specific_timestamps()
    test_different_datasets()  # Uncomment to test multiple datasets 