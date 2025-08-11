#!/usr/bin/env python3
"""
Inference Script for Chronicon Video Analysis
Processes test_video.mp4 with configurable frame parameters (max 16 frames)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from video_analyzer import VideoAnalyzer

def inference_video(video_path: str, num_frames: int = 8, prompt: str = None, 
                   resize: int = 336, max_tokens: int = 100, save_results: bool = True, verbose: bool = False,
                   model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"):
    """
    Perform inference on video with specified parameters.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (max 16)
        prompt: Custom prompt for analysis
        resize: Frame resize dimension
        max_tokens: Maximum tokens for generation
        save_results: Whether to save results to file
        verbose: Enable verbose output
        model_id: Model to use for inference
        
    Returns:
        Analysis results dictionary
    """
    
    print("🎥 Chronicon Video Inference - Detailed Process")
    print("=" * 60)
    
    # Validate parameters
    print("📋 Step 1: Validating parameters...")
    if num_frames > 16:
        print("⚠️ Warning: num_frames exceeds 16, setting to 16 for efficiency")
        num_frames = 16
    
    if num_frames < 1:
        print("❌ Error: num_frames must be at least 1")
        return None
    
    print(f"✅ Parameters validated:")
    print(f"   - Frames: {num_frames}")
    print(f"   - Resize: {resize}x{resize}")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Verbose: {verbose}")
    print(f"   - Model: {model_id}")
    
    # Check if video file exists
    print(f"\n📁 Step 2: Checking video file...")
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ Video file found: {video_path}")
    print(f"   - File size: {file_size_mb:.1f} MB ({file_size:,} bytes)")
    print(f"   - Last modified: {datetime.fromtimestamp(os.path.getmtime(video_path)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize analyzer
    print(f"\n🤖 Step 3: Initializing VideoAnalyzer...")
    print("   - Loading model...")
    print("   - Setting up 4-bit quantization...")
    print("   - Configuring device (CUDA/CPU)...")
    
    start_time = time.time()
    
    try:
        analyzer = VideoAnalyzer(model_id=model_id, verbose=verbose)
        init_time = time.time() - start_time
        print(f"✅ Model loaded successfully!")
        print(f"   - Device: {analyzer.device}")
        print(f"   - Model: {analyzer.model_id}")
        print(f"   - Load time: {init_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        if verbose:
            import traceback
            print("Detailed error:")
            traceback.print_exc()
        return None
    
    # Perform inference
    print(f"\n🎬 Step 4: Starting video analysis...")
    print("   - Opening video file...")
    print("   - Extracting frame metadata...")
    
    inference_start = time.time()
    
    try:
        # Use custom prompt if provided, otherwise use default
        if prompt is None:
            prompt = "Describe the main events, actions, and content in this video. Focus on key moments and important details."
        
        print(f"   - Using prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"   - Sampling {num_frames} frames...")
        
        if verbose:
            print("   - Frame sampling process:")
            print("     * Calculating frame intervals...")
            print("     * Extracting frames from video...")
            print("     * Resizing frames to {resize}x{resize}...")
            print("     * Converting to RGB format...")
        
        result = analyzer.analyze_video(
            video_path=video_path,
            prompt=prompt,
            num_frames=num_frames,
            resize=resize,
            max_tokens=max_tokens
        )
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        print(f"✅ Video analysis completed!")
        print(f"   - Frames processed: {result.get('num_frames', num_frames)}")
        print(f"   - Analysis time: {inference_time:.2f} seconds")
        print(f"   - Total time: {total_time:.2f} seconds")
        
        # Add timing information
        result.update({
            "inference_parameters": {
                "num_frames": num_frames,
                "resize": resize,
                "max_tokens": max_tokens,
                "custom_prompt": prompt is not None
            },
            "timing": {
                "model_load_time": init_time,
                "inference_time": inference_time,
                "total_time": total_time
            },
            "video_info": {
                "file_size_mb": file_size_mb,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(video_path)).isoformat()
            }
        })
        
        # Display results
        print(f"\n📝 Step 5: Analysis Results")
        print(f"   - Video: {result['video_path']}")
        print(f"   - Frames analyzed: {result['num_frames']}")
        print(f"   - Model: {result['model']}")
        print(f"   - Analysis length: {len(result['analysis'])} characters")
        print(f"   - Analysis preview: {result['analysis'][:200]}...")
        
        if verbose:
            print(f"\n📋 Full Analysis:")
            print(f"{result['analysis']}")
        
        # Save results if requested
        if save_results:
            print(f"\n💾 Step 6: Saving results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_results_{timestamp}.json"
            
            analyzer.save_analysis(result, filename)
            print(f"✅ Results saved to: {filename}")
            
            if verbose:
                print(f"   - File size: {os.path.getsize(filename)} bytes")
                print(f"   - Timestamp: {timestamp}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        if verbose:
            import traceback
            print("Detailed error:")
            traceback.print_exc()
        return None

def main():
    """Main function to handle command line arguments and run inference."""
    
    parser = argparse.ArgumentParser(
        description="Chronicon Video Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_script.py test_video.mp4
  python inference_script.py test_video.mp4 --frames 12
  python inference_script.py test_video.mp4 --frames 16 --prompt "Describe the sports action in this video"
  python inference_script.py test_video.mp4 --frames 8 --resize 512 --max-tokens 150
  python inference_script.py test_video.mp4 --verbose
  python inference_script.py test_video.mp4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct
  python inference_script.py test_video.mp4 --model HuggingFaceTB/SmolVLM2-2.2B-Instruct --verbose
  python inference_script.py --list-models
        """
    )
    
    parser.add_argument(
        "video_path",
        nargs='?',  # Make it optional
        help="Path to the video file to analyze"
    )
    
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=8,
        help="Number of frames to sample (max 16, default: 8)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom prompt for video analysis"
    )
    
    parser.add_argument(
        "--resize", "-r",
        type=int,
        default=336,
        help="Frame resize dimension (default: 336)"
    )
    
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=100,
        help="Maximum tokens for generation (default: 100)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed progress"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        choices=["HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"],
        help="Model to use for inference (default: 256M-Video-Instruct)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and their specifications"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("🤖 Available Models:")
        print("=" * 50)
        analyzer = VideoAnalyzer()
        models = analyzer.get_available_models()
        for model_id, specs in models.items():
            print(f"📦 {model_id}")
            print(f"   - Size: {specs['size']}")
            print(f"   - Type: {specs['type']}")
            print(f"   - Memory: {specs['memory_usage']}")
            print(f"   - Speed: {specs['speed']}")
            print(f"   - Quality: {specs['quality']}")
            print()
        
        # Check memory requirements
        print("💾 Memory Requirements:")
        print("=" * 30)
        for model_id in models.keys():
            memory_check = analyzer.check_memory_requirements(model_id)
            print(f"📊 {model_id}:")
            print(f"   - GPU Memory: {memory_check['gpu_memory_gb']} GB")
            print(f"   - Model Memory: {memory_check['model_memory_estimate']}")
            print(f"   - Sufficient: {'✅' if memory_check['sufficient_memory'] else '❌'}")
            if 'note' in memory_check:
                print("   - Note: {memory_check['note']}")
            print()
        return
    
    # Validate frame count
    if args.frames > 16:
        print("⚠️ Warning: Frame count exceeds 16, setting to 16 for efficiency")
        args.frames = 16
    elif args.frames < 1:
        print("❌ Error: Frame count must be at least 1")
        sys.exit(1)
    
    # Check if video file exists
    if not args.video_path and not args.list_models:
        print(f"❌ Error: Video file path is required unless --list-models is used.")
        sys.exit(1)
    
    if args.video_path and not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Display configuration
    print("🎥 Chronicon Video Inference Script")
    print("=" * 60)
    if args.video_path:
        print(f"📹 Video: {args.video_path}")
    print(f"🎬 Frames: {args.frames}")
    print(f"🔧 Resize: {args.resize}x{args.resize}")
    print(f"📝 Max tokens: {args.max_tokens}")
    print(f"💾 Save results: {not args.no_save}")
    print(f"🔍 Verbose mode: {args.verbose}")
    print(f"🤖 Model: {args.model}")
    
    # Show model specifications
    analyzer = VideoAnalyzer()
    model_info = analyzer.get_model_info(args.model)
    if "error" not in model_info:
        print(f"📊 Model specs: {model_info['size']} | {model_info['type']} | {model_info['quality']} quality")
    
    if args.prompt:
        print(f"📋 Custom prompt: {args.prompt}")
    
    # Get video file size
    if args.video_path:
        file_size_mb = os.path.getsize(args.video_path) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.1f} MB")
    
    # Run inference
    print("\n🚀 Starting inference process...")
    result = inference_video(
        video_path=args.video_path,
        num_frames=args.frames,
        prompt=args.prompt,
        resize=args.resize,
        max_tokens=args.max_tokens,
        save_results=not args.no_save,
        verbose=args.verbose,
        model_id=args.model
    )
    
    if result:
        print("\n✅ Inference completed successfully!")
        
        if args.verbose:
            print("\n📋 Detailed Results:")
            import json
            print(json.dumps(result, indent=2))
    else:
        print("\n❌ Inference failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
