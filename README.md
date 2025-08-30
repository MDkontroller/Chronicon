# Chronicon - AI Video Analysis Platform

A comprehensive video analysis platform featuring SmolVLM2 models with intelligent web interfaces and agentic AI capabilities for automated video understanding and decision-making.

## 🚀 Core Features

### 🎯 Two Main Applications

**🌐 Web Application (`webapp/app.py`)**
- Interactive web interface for video analysis
- Real-time streaming results with live progress updates
- Support for multiple video formats (up to 50MB)
- Customizable analysis parameters
- System health monitoring and debugging

**🤖 AI Agent Application (`webapp_agent/app.py`)**
- Autonomous decision-making based on video content
- Email notification system for detected events
- Keyword-based action triggers (trains, databases, etc.)
- Advanced logging and event tracking
- Automated response workflows

### 🧠 SmolVLM2 Model Support
- **256M Model**: Fast, efficient for quick analysis
- **2.2B Model**: High-quality, detailed descriptions
- Automatic memory requirement checking
- Intelligent quantization based on model size

### ⚡ Advanced Video Processing
- Configurable frame sampling (1-16 frames)
- Motion-based intelligent frame selection
- Customizable resize dimensions
- Proper frame interval calculation (no duplicates)
- RGB conversion and preprocessing

### 🔥 GPU Acceleration & Optimization
- **CUDA Support**: Full GPU acceleration with device placement control
- **4-bit Quantization**: Memory-efficient inference with BitsAndBytes
- **Device Safety**: Automatic CUDA/CPU mismatch prevention
- **Performance Tuning**: Optimized for both speed and memory usage
- **PyTorch 2.x Compilation**: JIT optimization for maximum performance

### 🛡️ Robust Error Handling
- Input validation for video frames and prompts
- Graceful quantization fallback
- Processor capability detection
- Comprehensive error reporting
- Unicode-safe subprocess handling

### 🎛️ Flexible Deployment Options
- Web interfaces with Flask
- Command-line inference scripts
- Subprocess-based isolation
- Real-time progress streaming
- JSON result export with metadata

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Chronicon

# Create and activate virtual environment
python -m venv chronicon_env
# Windows:
chronicon_env\Scripts\activate
# Linux/Mac:
source chronicon_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 📧 Email Setup (for AI Agent)

For the AI Agent features, configure email notifications:

```bash
# Create .env file in webapp_agent/ directory
cp webapp_agent/.env.example webapp_agent/.env

# Edit .env with your settings:
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_16_char_app_password
SMTP_TO=recipient@gmail.com
```

**Note**: Use Gmail App Passwords, not your regular password. Enable 2FA and generate an App Password in your Google Account settings.

## 🎯 Working Applications

### 🚀 Main Web Applications

| Application | Purpose | Key Features | Best For |
|-------------|---------|--------------|----------|
| **`webapp/app.py`** | 🌐 **Interactive Video Analysis** | Web UI, real-time streaming, progress tracking | Manual video analysis, experimentation |
| **`webapp_agent/app.py`** | 🤖 **AI Agent with Automation** | Decision-making, email alerts, event triggers | Automated monitoring, production workflows |

### 🛠️ Supporting Scripts

| Script | Purpose | Key Features | Use Case |
|--------|---------|--------------|----------|
| `webapp_agent/inference.py` | **GPU-Optimized Engine** | 4-bit quantization, CUDA safety, PyTorch 2.x | Backend engine for agent |
| `inference_script_gpu_fixed.py` | **Standalone Inference** | Robust GPU handling, comprehensive error recovery | Command-line analysis |
| `chronicon_infer_4bit.py` | **Performance Optimized** | NF4 quantization, maximum performance | High-throughput scenarios |

### 📊 Legacy/Development Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `inference_script.py` | Original CPU/GPU script | Legacy - use web apps instead |
| `test.py` | CPU testing | Development only |
| `video_analyzer.py` | Core library | Used by other scripts |

## 🚀 Quick Start

### 🌐 Web Application (Interactive Analysis)

```bash
# Navigate to web app directory
cd webapp

# Start the web interface
python app.py

# Open browser to http://localhost:5000
# Upload video and analyze interactively
```

### 🤖 AI Agent (Automated Monitoring)

```bash
# Navigate to agent directory
cd webapp_agent

# Configure email settings (see Email Setup above)
nano .env

# Start the AI agent
python app.py

# Open browser to http://localhost:5000
# Agent will automatically send emails when triggers are detected
```

### 💻 Command Line (Direct Analysis)

```bash
# GPU-optimized standalone analysis
python chronicon_infer_4bit.py test_video.mp4 --verbose

# Robust analysis with error handling
python inference_script_gpu_fixed.py test_video.mp4 --frames 8
```

### 🔧 Advanced Configuration

```bash
# High-quality analysis with 2.2B model
python webapp_agent/inference.py test_video.mp4 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --frames 12 \
  --resize 512 \
  --max-tokens 200 \
  --verbose
```

## 🤖 AI Agent Capabilities

### 🧠 Intelligent Decision Making

The AI Agent (`webapp_agent/app.py`) provides autonomous decision-making based on video content analysis:

- **Keyword Detection**: Automatic detection of trains, railways, databases, and other specified content
- **Action Triggers**: Configurable responses based on detected content
- **Scoring System**: Confidence scoring for detected events
- **Workflow Automation**: Automated responses without human intervention

### 📧 Email Notification System

Powered by `notifier.py`, the system provides:

- **Gmail Integration**: Secure SMTP with App Password authentication
- **Auto-Fallback**: Automatic port switching (587/465) for reliability
- **Rich HTML Emails**: Formatted notifications with analysis details
- **Error Handling**: Comprehensive error reporting for email issues

### 🎯 Agent Workflow

1. **Video Upload**: User uploads video through web interface
2. **Analysis**: SmolVLM2 processes and describes video content
3. **Decision Engine**: AI agent analyzes description for trigger keywords
4. **Action Execution**: Automated email sent if triggers detected
5. **Logging**: Complete audit trail of decisions and actions

### ⚙️ Configuration

#### Agent Decision Rules

```python
# Current triggers (easily customizable):
TRAIN_KEYWORDS = ["train", "rail", "railway", "metro", "locomotive", "db", "Deutsche", "Bahn"]
DATABASE_KEYWORDS = ["database", "db ", " db.", "sql", "postgres", "mysql", "mongodb"]
```

#### Email Settings

```bash
# Required environment variables:
SMTP_HOST=smtp.gmail.com          # SMTP server
SMTP_PORT=587                     # Port (587 or 465)
SMTP_USER=your_email@gmail.com    # Your Gmail address
SMTP_PASSWORD=app_password_16char # Gmail App Password
SMTP_TO=recipient@example.com     # Notification recipient
```

### 🔍 Example Agent Decision

```json
{
  "decision": {
    "label": "train",
    "score": 0.75,
    "reason": "Matched: train, railway"
  },
  "action": "send_email",
  "email_status": "sent"
}
```

## 📊 Real Performance Benchmarks

### Actual Performance Data (test_video.mp4, 256M Model)

| Script Type | Quantization | Frames | Model Load | Inference | Total | Performance |
|-------------|--------------|--------|------------|-----------|-------|-------------|
| **4-bit Optimized** | ✅ 4-bit NF4 | 4 | 6.1s | **13.7s** | **20.3s** | 🚀 **Fastest** |
| **4-bit Optimized** | ✅ 4-bit NF4 | 8 | 7.0s | **60.3s** | **67.3s** | 🚀 **Fast** |
| **Standard GPU** | ❌ No Quant | 4 | 5.5s | 66.0s | 71.1s | 🐌 Slow |
| **Standard GPU** | ❌ No Quant | 8 | 5.4s | 67.0s | 72.4s | 🐌 Slow |
| **CPU Fallback** | ❌ No Quant | 4 | 0.0s | 131.6s | 131.6s | 🐌 **Slowest** |

### Performance Analysis

**Key Findings:**
- **4-bit quantization provides 4-5x speedup** for inference
- **Frame count scaling**: 4→8 frames increases time ~4x (expected due to processing complexity)
- **Model loading**: Consistent 5-7s for GPU, 0s for pre-loaded models
- **CPU fallback**: 2x slower than standard GPU, should be avoided for production

### Application Performance Characteristics

| Application | Backend Engine | Memory Usage | Speed Rating | Features | Best Use Case |
|-------------|----------------|--------------|--------------|----------|---------------|
| **`webapp_agent/app.py`** | GPU-optimized inference.py | ~800MB | ⭐⭐⭐⭐⭐ | AI decisions, email alerts | **Automated monitoring** |
| **`webapp/app.py`** | Subprocess-based | ~1GB | ⭐⭐⭐⭐ | Web UI, real-time streaming | **Interactive analysis** |
| `chronicon_infer_4bit.py` | Direct CUDA | ~800MB | ⭐⭐⭐⭐⭐ | Maximum performance | **Command-line batch** |
| `inference_script_gpu_fixed.py` | Robust CUDA | ~1GB | ⭐⭐⭐⭐ | Error recovery | **Reliable CLI analysis** |

### Memory Requirements

| Model | Standard Loading | 4-bit Quantized | NF4 Optimized |
|-------|------------------|-----------------|---------------|
| SmolVLM2-256M | ~2GB VRAM | ~1GB VRAM | **~800MB VRAM** |
| SmolVLM2-2.2B | ~8GB VRAM | ~3GB VRAM | **~2.5GB VRAM** |

## 🛠️ Technical Improvements

### GPU Optimization Features
1. **Device Placement Control**: Single-GPU placement prevents auto-sharding issues
2. **4-bit Quantization**: BitsAndBytes NF4 quantization reduces memory by 60-75%
3. **CUDA Safety**: Bucketize shim prevents CUDA/CPU tensor device mismatches
4. **Performance Flags**: TF32, cuDNN benchmark optimization for inference
5. **Memory Management**: Automatic GPU memory checking and optimization

### Enhanced Error Handling
- **Quantization Fallback**: Graceful degradation when 4-bit fails
- **Device Mismatch Prevention**: Automatic tensor device alignment
- **Model Loading Recovery**: Multiple loading strategies with fallbacks
- **Comprehensive Validation**: Input validation for all parameters

## 🔧 Usage Examples

### 🌐 Web Application Examples

#### Interactive Video Analysis
```bash
cd webapp
python app.py

# Web interface available at http://localhost:5000
# Features:
# - Drag & drop video upload
# - Real-time analysis streaming
# - Customizable parameters
# - System monitoring dashboard
```

#### Health Check & Debugging
```bash
# Check system status
curl http://localhost:5000/health

# Get debug information
curl http://localhost:5000/debug
```

### 🤖 AI Agent Examples

#### Automated Monitoring Setup
```bash
cd webapp_agent

# Configure environment
echo "SMTP_USER=your_email@gmail.com" > .env
echo "SMTP_PASSWORD=your_app_password" >> .env
echo "SMTP_TO=alerts@company.com" >> .env

python app.py
# Agent active at http://localhost:5000
```

#### Manual Decision Testing
```bash
# Test agent decision logic
curl -X POST http://localhost:5000/decide_and_act \
  -H "Content-Type: application/json" \
  -d '{
    "analysis": "The video shows a train arriving at the station",
    "run_id": "test_123",
    "video_filename": "test.mp4"
  }'
```

### 💻 Command Line Examples

#### Maximum Performance
```bash
# GPU-optimized standalone analysis
python chronicon_infer_4bit.py test_video.mp4 \
  --frames 4 \
  --resize 336 \
  --max-tokens 100 \
  --verbose
# Expected: ~20s total time
```

#### High-Quality Analysis
```bash
# Detailed analysis with larger model
python webapp_agent/inference.py test_video.mp4 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --frames 8 \
  --resize 448 \
  --max-tokens 150 \
  --verbose
# Expected: ~67s total time
```

## 📝 Output Format

Results are saved as JSON files with comprehensive performance metadata:

```json
{
  "video_path": "test_video.mp4",
  "prompt": "Describe the main events...",
  "num_frames": 4,
  "resize": 336,
  "max_tokens": 100,
  "analysis": "The video shows...",
  "timestamp": "2025-08-12T16:04:49.496443",
  "model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
  "device": "cuda:0",
  "quantized": true,
  "timing": {
    "model_load_time": 6.67,
    "inference_time": 13.67,
    "total_time": 20.34
  }
}
```

## 🔍 Configuration Options

### 🌐 Web Interface Parameters

Both web applications support these parameters via the web UI:

```bash
# Video upload limits
MAX_FILE_SIZE = 50MB

# Analysis parameters (configurable in UI)
frames           = 1-16 frames (default: 8)
max_tokens       = 250-500 tokens (default: 250)
min_tokens       = 50-200 tokens (default: 50)
prompt           = Custom analysis prompt
```

### 🤖 AI Agent Configuration

Environment variables for email notifications:

```bash
SMTP_HOST           = SMTP server (default: smtp.gmail.com)
SMTP_PORT           = Port 587 (STARTTLS) or 465 (SSL)
SMTP_USER           = Your Gmail address
SMTP_PASSWORD       = Gmail App Password (16 characters)
SMTP_TO             = Notification recipient email
SMTP_FROM           = Display name (optional)
```

### 💻 Command Line Options

For direct script execution:

```bash
# Common options (inference scripts)
--frames, -f          Number of frames to sample (1-16, default: 8)
--prompt, -p          Custom prompt for analysis
--resize, -r          Frame resize dimension (default: 336)
--max-tokens, -t      Maximum tokens for generation (default: 100)
--min-tokens          Minimum tokens for generation (default: 50)
--model, -m           Model to use (256M or 2.2B)
--verbose, -v         Enable verbose output
--no-save             Don't save results to file

# GPU-specific options
--cuda-device         CUDA device ID (default: 0)
--no-quant            Disable 4-bit quantization
```

### 🎛️ Application-Specific Features

- **Web App**: Real-time streaming, progress tracking, system monitoring
- **AI Agent**: Decision engine, email alerts, event logging, automation
- **Inference Engine**: PyTorch compilation, motion-based frame selection, CUDA safety

## 🐛 Troubleshooting

### 🌐 Web Application Issues

1. **Server Won't Start**
   ```bash
   # Check if port 5000 is in use
   netstat -an | grep 5000
   
   # Install missing dependencies
   pip install -r requirements.txt
   
   # Check Python version (3.8+ required)
   python --version
   ```

2. **Video Upload Fails**
   - Check file size (max 50MB)
   - Ensure video format is supported (MP4, AVI, MOV)
   - Verify sufficient disk space in temp directory

3. **Analysis Hangs or Fails**
   - Check GPU memory availability
   - Review server logs for error messages
   - Test with smaller video or fewer frames

### 🤖 AI Agent Issues

1. **Email Notifications Not Working**
   ```bash
   # Verify environment variables
   echo $SMTP_USER
   echo $SMTP_PASSWORD
   
   # Test Gmail App Password
   # - Enable 2FA in Google Account
   # - Generate 16-character App Password
   # - Use App Password, not regular password
   ```

2. **Agent Not Making Decisions**
   - Check if keywords exist in analysis text
   - Review decision logic in `decide_action()` function
   - Verify `/decide_and_act` endpoint is accessible

### ⚡ Performance Issues

1. **Slow Inference (>60s for 4 frames)**
   - ✅ Use GPU-enabled applications (`webapp_agent/app.py`)
   - ✅ Enable 4-bit quantization in backend
   - ✅ Reduce frame count to 4 for fastest results

2. **CUDA Out of Memory**
   - Use 4-bit quantization (reduces memory by 60-75%)
   - Reduce frame count or resize dimensions in web UI
   - Try 256M model instead of 2.2B

3. **Device Mismatch Errors**
   - Check that inference backend supports CUDA
   - Verify PyTorch CUDA installation
   - Review server logs for device placement errors

### Performance Optimization Tips

| Scenario | Recommended Application | Settings | Expected Time |
|----------|------------------------|----------|---------------|
| **Automated Monitoring** | `webapp_agent/app.py` | 4 frames, AI agent | ~20s + notifications |
| **Interactive Analysis** | `webapp/app.py` | 8 frames, web UI | ~67s with streaming |
| **Maximum Speed** | `chronicon_infer_4bit.py` | 4 frames, 4-bit | ~20s |
| **High Reliability** | `inference_script_gpu_fixed.py` | 4 frames | ~71s |

### Memory Optimization

| VRAM Available | Recommended Configuration |
|----------------|--------------------------|
| **<2GB** | 256M + 4-bit + 4 frames |
| **2-4GB** | 256M + 4-bit + 8 frames |
| **4-8GB** | 2.2B + 4-bit + 4-8 frames |
| **8GB+** | 2.2B + optional 4-bit + 8+ frames |

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Test thoroughly**:
   - Web interfaces: Test both `webapp/app.py` and `webapp_agent/app.py`
   - AI agent: Verify email functionality and decision logic
   - Performance: Benchmark with different video sizes and models
4. **Update documentation** as needed
5. **Submit a pull request**

### 🔧 Development Setup

```bash
# Development environment
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If test suite exists

# Check both web applications
cd webapp && python app.py &
cd webapp_agent && python app.py &
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace** for the SmolVLM2 models and transformers library
- **BitsAndBytes team** for efficient quantization techniques
- **Flask** and **OpenCV** communities for excellent web and video processing tools
- **PyTorch team** for GPU acceleration and optimization features
- **Open-source community** for continuous inspiration and collaboration

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB system memory
- **Storage**: 5GB free space for models
- **OS**: Windows 10+, Linux, macOS

### Recommended for Production
- **GPU**: NVIDIA with 4GB+ VRAM (RTX 3060, RTX 4060, or better)
- **RAM**: 16GB+ system memory
- **Storage**: SSD with 10GB+ free space
- **Network**: Stable internet for model downloads and email notifications

---

**⚠️ Important Notes**: 
- This system is designed for research and development purposes
- Ensure you have appropriate licenses for any video content you analyze
- Gmail App Passwords required for email notifications (regular passwords won't work)
- The AI agent makes automated decisions - review and customize trigger keywords for your use case