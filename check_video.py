#!/usr/bin/env python3
"""
Video File Checker for Chronicon
Checks video file properties and prepares for inference
"""

import os
import cv2
from datetime import datetime

def check_video_file(video_path: str):
    """
    Check video file properties and display information.
    
    Args:
        video_path: Path to video file
    """
    
    print("🎥 Video File Checker")
    print("=" * 40)
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("\n💡 To use this script:")
        print("1. Place a video file named 'test_video.mp4' in the current directory")
        print("2. Or specify a different video path")
        return False
    
    # Get file information
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    last_modified = datetime.fromtimestamp(os.path.getmtime(video_path))
    
    print(f"✅ Video file found: {video_path}")
    print(f"📊 File size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
    print(f"📅 Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Open video and get properties
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        print(f"\n📹 Video Properties:")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - Total frames: {frame_count:,}")
        print(f"   - Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Calculate frame intervals for different frame counts
        print(f"\n🎬 Frame Sampling Options:")
        frame_counts = [4, 8, 12, 16]
        
        for count in frame_counts:
            if frame_count > 0:
                interval = frame_count / count
                print(f"   - {count} frames: Every {interval:.1f} frames ({interval/fps:.1f}s intervals)")
        
        # Estimate processing time
        print(f"\n⏱️ Estimated Processing:")
        print(f"   - Model load time: ~10-30 seconds")
        print(f"   - Frame extraction: ~1-5 seconds")
        print(f"   - Inference time: ~5-15 seconds per frame set")
        print(f"   - Total estimated time: 20-60 seconds")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if file_size_mb > 100:
            print(f"   - Large file ({file_size_mb:.1f} MB), consider using fewer frames (4-8)")
        elif file_size_mb > 50:
            print(f"   - Medium file ({file_size_mb:.1f} MB), 8-12 frames should work well")
        else:
            print(f"   - Small file ({file_size_mb:.1f} MB), 12-16 frames should be fine")
        
        if duration > 300:  # 5 minutes
            print(f"   - Long video ({duration/60:.1f} minutes), consider using 8-12 frames")
        elif duration > 60:  # 1 minute
            print(f"   - Medium video ({duration:.1f} seconds), 8-16 frames should work")
        else:
            print(f"   - Short video ({duration:.1f} seconds), 12-16 frames recommended")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading video properties: {str(e)}")
        return False

def main():
    """Main function to check video file."""
    
    # Check for test_video.mp4 first
    video_path = "test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ {video_path} not found in current directory")
        print("\n📁 Files in current directory:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                print(f"   - {file}")
        
        print(f"\n💡 To run inference:")
        print(f"1. Place a video file named 'test_video.mp4' in the current directory")
        print(f"2. Or run: python inference_script.py <your_video_file>")
        return
    
    # Check the video file
    if check_video_file(video_path):
        print(f"\n✅ Video file is ready for inference!")
        print(f"\n🚀 To run inference:")
        print(f"   python inference_script.py {video_path}")
        print(f"   python inference_script.py {video_path} --frames 8")
        print(f"   python inference_script.py {video_path} --frames 12 --prompt 'Describe the main events'")
    else:
        print(f"\n❌ Video file check failed!")

if __name__ == "__main__":
    main()
