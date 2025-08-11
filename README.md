# Chronicon - Video Analysis with SmolVLM2

A powerful video analysis system using HuggingFace's SmolVLM2 models for intelligent video understanding and description generation.

## 🚀 Features

### Multi-Model Support
- **256M Model**: Fast, efficient for quick analysis
- **2.2B Model**: High-quality, detailed descriptions
- Automatic memory requirement checking
- Intelligent quantization based on model size

### Advanced Video Processing
- Configurable frame sampling (1-16 frames)
- Customizable resize dimensions
- Proper frame interval calculation (no duplicates)
- RGB conversion and preprocessing

### Robust Error Handling
- Input validation for video frames and prompts
- Graceful quantization fallback
- Processor capability detection
- Comprehensive error reporting

### Flexible Inference
- Command-line interface with extensive options
- Verbose logging for detailed progress tracking
- JSON result export with metadata
- Custom prompts and parameters

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Chronicon

# Activate virtual environment
activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### List Available Models
```bash
python inference_script.py --list-models
```

### Basic Video Analysis
```bash
python inference_script.py test_video.mp4
```

### Advanced Analysis with 2.2B Model
```bash
python inference_script.py test_video.mp4 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --frames 8 \
  --resize 448 \
  --max-tokens 150 \
  --verbose
```

## 🔧 Usage Examples

### Fast Analysis (256M Model)
```bash
python inference_script.py test_video.mp4 --frames 4 --resize 336
```

### High-Quality Analysis (2.2B Model)
```bash
python inference_script.py test_video.mp4 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --frames 12 \
  --resize 512 \
  --max-tokens 200 \
  --prompt "Describe the main characters and their actions in detail"
```

### Custom Analysis
```bash
python inference_script.py test_video.mp4 \
  --frames 6 \
  --resize 448 \
  --max-tokens 100 \
  --prompt "What emotions are displayed in this video?" \
  --verbose
```

## 📊 Model Specifications

| Model | Size | Type | Memory | Speed | Quality |
|-------|------|------|--------|-------|---------|
| SmolVLM2-256M-Video-Instruct | 256M | Video-Instruct | ~2GB | Fast | Good |
| SmolVLM2-2.2B-Instruct | 2.2B | Instruct | ~8GB | Slower | Excellent |

## 🛠️ Technical Improvements

### Fixed Issues
1. **Parameter Passing**: Resize and max_tokens now properly passed through the pipeline
2. **Frame Sampling**: Improved logic to avoid duplicate frames using proper rounding
3. **Model Loading**: Robust quantization with fallback for unsupported configurations
4. **Input Validation**: Comprehensive validation of video frames and prompts
5. **Processor Detection**: Automatic capability checking for video processing

### Enhanced Features
- **Memory Management**: Automatic GPU memory checking and optimization
- **Error Recovery**: Graceful handling of quantization failures
- **Progress Tracking**: Detailed verbose output for debugging
- **Result Metadata**: Comprehensive JSON export with timing and parameters

## 📝 Output Format

Results are saved as JSON files with the following structure:

```json
{
  "video_path": "test_video.mp4",
  "prompt": "Describe the main events...",
  "num_frames": 8,
  "resize": 448,
  "max_tokens": 100,
  "analysis": "The video shows...",
  "timestamp": "2025-08-08T14:30:09.644298",
  "model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
  "inference_parameters": {
    "num_frames": 8,
    "resize": 448,
    "max_tokens": 100,
    "custom_prompt": true
  },
  "timing": {
    "model_load_time": 0.0,
    "inference_time": 22.76,
    "total_time": 22.76
  },
  "video_info": {
    "file_size_mb": 2.13,
    "last_modified": "2025-08-05T20:51:42.890145"
  }
}
```

## 🔍 Command Line Options

```bash
python inference_script.py [VIDEO_PATH] [OPTIONS]

Options:
  --frames, -f          Number of frames to sample (1-16, default: 8)
  --prompt, -p          Custom prompt for analysis
  --resize, -r          Frame resize dimension (default: 336)
  --max-tokens, -t      Maximum tokens for generation (default: 100)
  --model, -m           Model to use (256M or 2.2B)
  --verbose, -v         Enable verbose output
  --no-save             Don't save results to file
  --list-models         List available models and specifications
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use the 256M model instead of 2.2B
   - Reduce frame count or resize dimensions
   - Check available GPU memory with `--list-models`

2. **Model Loading Errors**
   - The system automatically falls back to standard loading if quantization fails
   - Check your transformers version compatibility

3. **Video Processing Issues**
   - Ensure video file is valid and accessible
   - Check frame count and duration
   - Verify video codec compatibility

### Performance Tips

- **Fast Analysis**: Use 256M model with 4-6 frames
- **High Quality**: Use 2.2B model with 8-12 frames
- **Memory Optimization**: Use smaller resize dimensions (336-448)
- **Speed vs Quality**: Balance frame count with model size

## 📈 Performance Comparison

| Configuration | Model | Frames | Time | Quality |
|---------------|-------|--------|------|---------|
| Fast | 256M | 4 | ~10s | Good |
| Balanced | 256M | 8 | ~15s | Good |
| Quality | 2.2B | 6 | ~23s | Excellent |
| Maximum | 2.2B | 12 | ~45s | Excellent |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for the SmolVLM2 models
- The open-source community for video processing tools
- Contributors and testers

---

**Note**: This system is designed for research and development purposes. Ensure you have appropriate licenses for any video content you analyze.