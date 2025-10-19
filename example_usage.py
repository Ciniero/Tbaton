#!/usr/bin/env python3
"""
Example usage of the Video to Instructions Generator
"""

import os
import sys
from video_to_instructions import VideoAnalyzer

def create_sample_video():
    """Create a simple sample video for testing."""
    import cv2
    import numpy as np
    
    # Create a simple video with moving objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_video.mp4', fourcc, 30.0, (640, 480))
    
    for i in range(90):  # 3 seconds at 30 FPS
        # Create a frame with a moving rectangle (simulating a person)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(50)  # Dark background
        
        # Draw a moving rectangle
        x = int(50 + i * 2)
        y = 200
        cv2.rectangle(frame, (x, y), (x + 50, y + 100), (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("Sample video created: sample_video.mp4")

def main():
    print("Video to Instructions Generator - Example Usage")
    print("=" * 50)
    
    # Check if sample video exists, create if not
    if not os.path.exists('sample_video.mp4'):
        print("Creating sample video...")
        create_sample_video()
    
    # Initialize analyzer
    print("\nInitializing video analyzer...")
    analyzer = VideoAnalyzer()
    
    # Process the sample video
    print("Processing sample video...")
    try:
        result = analyzer.process_video('sample_video.mp4')
        
        print("\n" + "="*50)
        print("GENERATED INSTRUCTIONS")
        print("="*50)
        
        if result['instructions']:
            for instruction in result['instructions']:
                print(instruction)
        else:
            print("No specific actions detected in the sample video.")
            print("This is expected for the simple test video.")
        
        print(f"\nAnalysis Summary:")
        print(f"- Total frames analyzed: {result['analysis_frames']}")
        print(f"- Total frames in video: {result['total_frames']}")
        print(f"- Instructions generated: {len(result['instructions'])}")
        
        # Save results
        import json
        with open('sample_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nDetailed results saved to: sample_results.json")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())