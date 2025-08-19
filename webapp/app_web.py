#!/usr/bin/env python3
"""
Chronicon Web App - Video Analysis with SmolVLM2
Flask web interface for video event detection and analysis
"""

from flask import Flask, render_template, request, jsonify, Response
import torch
import cv2
import numpy as np
import io
import base64
import json
import threading
import gc
import time
import os
import sys
from datetime import datetime
from functools import lru_cache
import psutil
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for videos

# Global variables
chronicon_engine = None
model_loaded = False

# Event tracking for n8n integration
events_log = []
inference_results = {}

# --------- HARDENING: device-safe bucketize shim (prevents cuda/cpu mix) ----------
if not hasattr(torch, "_orig_bucketize"):
    torch._orig_bucketize = torch.bucketize
    def _bucketize_device_safe(input, boundaries, *args, **kwargs):
        if torch.is_tensor(boundaries) and input.is_cuda and not boundaries.is_cuda:
            boundaries = boundaries.to(input.device)
        return torch._orig_bucketize(input, boundaries, *args, **kwargs)
    torch.bucketize = _bucketize_device_safe
# ----------------------------------------------------------------------------------

class ChroniconVLM:
    """Optimized Chronicon VLM for web deployment"""
    
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        verbose: bool = False,
        quant_4bit: bool = True,
        cuda_device: int = 0,
    ):
        self.model_id = model_id
        self.verbose = verbose
        self.quant_4bit = quant_4bit
        self.cuda_device_index = cuda_device

        self.model = None
        self.processor = None

        # Device setup with fallback
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cuda_device_index}")
            self.fp_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.fp_dtype = torch.float32
            self.quant_4bit = False  # Disable quantization on CPU
            
        self.cpu_dtype = torch.float32

        # Encourage factory tensors to land on target device
        try:
            torch.set_default_device(self.device)
        except Exception:
            pass

    def load(self):
        """Load model with maximum optimizations"""
        if self.verbose:
            print(f"📦 Loading Chronicon model {self.model_id}")
            print(f"   - Device: {self.device}")
            print(f"   - 4-bit quantization: {self.quant_4bit}")
            if torch.cuda.is_available():
                print(f"   - GPU: {torch.cuda.get_device_name(self.cuda_device_index)}")

        tried = []

        # Try optimized loading
        try:
            if self.quant_4bit and self.device.type == "cuda":
                self._load_gpu_nf4()
                tried.append("GPU-4bit")
            elif self.device.type == "cuda":
                self._load_gpu_fp16()
                tried.append("GPU-fp16")
            else:
                self._load_cpu()
                tried.append("CPU")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Primary load failed: {e}")
            
            # Fallback strategies
            if self.device.type == "cuda" and self.quant_4bit:
                try:
                    print("🔄 Trying GPU without quantization...")
                    self._load_gpu_fp16()
                    tried.append("GPU-fp16-fallback")
                except Exception as e2:
                    print(f"❌ GPU fallback failed: {e2}")
                    print("🔄 Falling back to CPU...")
                    self._load_cpu()
                    tried.append("CPU-fallback")
            else:
                self._load_cpu()
                tried.append("CPU-fallback")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # Setup tokenizer
        if hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

        if self.verbose:
            print(f"✅ Load complete on: {self._model_primary_device()} (tried: {', '.join(tried)})")

        # Device auditing for CUDA
        if self.device.type == "cuda":
            self._audit_and_fix_devices()

        return True

    def _load_gpu_nf4(self):
        """Load with 4-bit quantization"""
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=bnb,
            device_map={"": self.cuda_device_index},
            low_cpu_mem_usage=True,
        ).eval()

    def _load_gpu_fp16(self):
        """Load with FP16 precision"""
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": self.cuda_device_index},
            low_cpu_mem_usage=True,
        ).eval()

    def _load_cpu(self):
        """Load on CPU"""
        self.device = torch.device("cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        ).eval()

    def _model_primary_device(self):
        """Get model's primary device"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            for _, b in self.model.named_buffers():
                return b.device
            return torch.device("cpu")

    def _audit_and_fix_devices(self):
        """Move CPU stragglers to CUDA"""
        cpu_items = []
        for n, p in self.model.named_parameters():
            if p.device.type != "cuda":
                cpu_items.append(("param", n, p.device))
        for n, b in self.model.named_buffers():
            if hasattr(b, "device") and b.device.type != "cuda":
                cpu_items.append(("buffer", n, b.device))

        if cpu_items and self.verbose:
            print(f"⚠️ Moving {len(cpu_items)} items from CPU to CUDA...")

        for _, n, _ in cpu_items:
            parts = n.split(".")[:-1]
            mod = self.model
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is not None:
                try:
                    mod.to(self.device)
                except Exception:
                    pass

    def sample_frames_from_bytes(self, video_bytes: bytes, num_frames: int = 8, resize: int = 336) -> List[np.ndarray]:
        """Sample frames from video bytes"""
        # Save bytes to temporary file
        #temp_path = f"/tmp/temp_video_{int(time.time())}.mp4"   for linux temporal file storage
        #temp_path = f"temp_video_{int(time.time())}.mp4"
        # Replace the temp path lines with:
        temp_path = os.path.join(tempfile.gettempdir(), f"temp_video_{int(time.time())}.mp4")
        try:
            with open(temp_path, 'wb') as f:
                f.write(video_bytes)
            
            return self.sample_frames(temp_path, num_frames, resize)
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def sample_frames(self, video_path: str, num_frames: int = 8, resize: int = 336) -> List[np.ndarray]:
        """Sample frames from video file"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video: {video_path}")

        # Calculate frame indices
        if num_frames == 1:
            indices = [total_frames // 2]
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            indices = sorted(list(dict.fromkeys(indices)))  # Remove duplicates

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Resize frame
            h, w = frame.shape[:2]
            scale = resize / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        
        if not frames:
            raise RuntimeError("No frames could be extracted")
            
        return frames

    def analyze_streaming(self, frames: List[np.ndarray], prompt: str, max_tokens: int = 120):
        """Streaming video analysis"""
        # Build prompt with image tokens
        text_only = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt_text = self.processor.apply_chat_template(text_only, add_generation_prompt=True)
        image_tokens = "<image>" * len(frames)
        prompt_with_images = prompt_text.replace("<|im_start|>User:", f"<|im_start|>User: {image_tokens}")

        # Process inputs
        inputs = self.processor(
            text=prompt_with_images,
            images=frames,
            return_tensors="pt",
        )

        # Move to device
        inputs = self._move_inputs_to_model_device(inputs)

        # Setup streaming
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=30
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.05,
            "use_cache": True,
            "streamer": streamer,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }

        # Generate in background thread
        def generate():
            try:
                with torch.no_grad():
                    if self.device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            self.model.generate(**generation_kwargs)
                    else:
                        self.model.generate(**generation_kwargs)
            except Exception as e:
                print(f"Generation error: {e}")
                streamer.put(f"Error: {str(e)}")
            finally:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        # Stream tokens
        for token in streamer:
            if token and token.strip():
                yield token

        thread.join(timeout=60)

    def analyze(self, frames: List[np.ndarray], prompt: str, max_tokens: int = 120) -> str:
        """Non-streaming analysis"""
        result_tokens = list(self.analyze_streaming(frames, prompt, max_tokens))
        return "".join(result_tokens)

    def _move_inputs_to_model_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move inputs to model device with proper dtypes"""
        model_dev = self._model_primary_device()
        moved = {}

        is_cuda = model_dev.type == "cuda"
        pixel_dtype = self.fp_dtype if is_cuda else self.cpu_dtype

        for k, v in inputs.items():
            if not hasattr(v, "to"):
                moved[k] = v
                continue
            if k == "pixel_values":
                v = v.to(model_dev, dtype=pixel_dtype, non_blocking=is_cuda)
            else:
                v = v.to(model_dev, non_blocking=is_cuda)
            moved[k] = v
        return moved

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.generation_time = 0
        self.preprocessing_time = 0
        self.tokens_generated = 0
        
    def get_stats(self):
        total_time = time.time() - self.start_time
        tokens_per_second = self.tokens_generated / self.generation_time if self.generation_time > 0 else 0
        
        return {
            'total_time': round(total_time, 2),
            'generation_time': round(self.generation_time, 2),
            'preprocessing_time': round(self.preprocessing_time, 2),
            'tokens_generated': self.tokens_generated,
            'tokens_per_second': round(tokens_per_second, 2),
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'gpu_memory': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
        }

perf_monitor = PerformanceMonitor()

def log_event(run_id: str, stage: str, detail: Dict[str, Any]):
    """Log event for n8n tracking"""
    event = {
        "run_id": run_id,
        "stage": stage,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
    events_log.append(event)
    
    # Keep only last 100 events
    if len(events_log) > 100:
        events_log.pop(0)
    
    print(f"📝 Event logged: {stage} for run {run_id}")

def load_chronicon_model():
    """Load Chronicon model with optimizations"""
    global chronicon_engine, model_loaded
    
    print("🚀 Loading Chronicon VLM...")
    
    try:
        chronicon_engine = ChroniconVLM(
            model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
            verbose=True,
            quant_4bit=True,
            cuda_device=0
        )
        
        success = chronicon_engine.load()
        if success:
            model_loaded = True
            print("✅ Chronicon model loaded successfully!")
            return True
        else:
            print("❌ Failed to load Chronicon model")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Flask Routes

@app.route('/')
def index():
    """Main page"""
    return render_template('chronicon_index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read video data
        video_data = file.read()
        
        # Store for processing
        app.current_video_data = video_data
        app.current_video_filename = file.filename
        
        # Get basic video info
        #temp_path = f"/tmp/temp_{int(time.time())}.mp4"
        temp_path = os.path.join(tempfile.gettempdir(), f"temp_video_{int(time.time())}.mp4")
        try:
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # Create thumbnail
            frames = chronicon_engine.sample_frames(temp_path, num_frames=1, resize=400)
            if frames:
                from PIL import Image
                thumbnail = Image.fromarray(frames[0])
                buffer = io.BytesIO()
                thumbnail.save(buffer, format='JPEG', quality=80)
                thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode()
            else:
                thumbnail_b64 = None
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'size_mb': round(len(video_data) / (1024 * 1024), 2),
            'duration': round(duration, 1),
            'resolution': f"{width}x{height}",
            'fps': round(fps, 1),
            'frame_count': frame_count,
            'thumbnail': f"data:image/jpeg;base64,{thumbnail_b64}" if thumbnail_b64 else None
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/analyze')
def analyze_video():
    """Streaming video analysis"""
    if not model_loaded:
        return Response("Error: Model not loaded", mimetype='text/plain')
    
    if not hasattr(app, 'current_video_data'):
        return Response("Error: No video uploaded", mimetype='text/plain')
    
    # Get parameters
    prompt = request.args.get('prompt', 'Detect notable events and summarize what happens in this video.')
    frames = min(int(request.args.get('frames', 8)), 16)  # Max 16 frames
    resize = int(request.args.get('resize', 336))
    max_tokens = min(int(request.args.get('max_tokens', 120)), 300)  # Max 300 tokens
    run_id = request.args.get('run_id', f"web_{int(time.time())}")
    
    def generate():
        try:
            perf_monitor.reset()
            start_time = time.time()
            
            # Log start
            log_event(run_id, "analysis_started", {
                "prompt": prompt,
                "frames": frames,
                "resize": resize,
                "max_tokens": max_tokens,
                "filename": getattr(app, 'current_video_filename', 'unknown')
            })
            
            yield f"data: {json.dumps({'status': 'Extracting frames...'})}\n\n"
            
            # Extract frames
            frame_data = chronicon_engine.sample_frames_from_bytes(
                app.current_video_data, 
                num_frames=frames, 
                resize=resize
            )
            
            yield f"data: {json.dumps({'status': f'Analyzing {len(frame_data)} frames...'})}\n\n"
            
            # Stream analysis
            full_response = ""
            token_count = 0
            
            for token in chronicon_engine.analyze_streaming(frame_data, prompt, max_tokens):
                full_response += token
                token_count += 1
                perf_monitor.tokens_generated = token_count
                yield f"data: {json.dumps({'text': token})}\n\n"
            
            # Calculate timing
            total_time = time.time() - start_time
            perf_monitor.generation_time = total_time
            
            # Store result
            result = {
                "run_id": run_id,
                "video_filename": getattr(app, 'current_video_filename', 'unknown'),
                "prompt": prompt,
                "num_frames": len(frame_data),
                "resize": resize,
                "max_tokens": max_tokens,
                "analysis": full_response,
                "timestamp": datetime.now().isoformat(),
                "timing": {
                    "total_time": round(total_time, 2),
                    "tokens_generated": token_count,
                    "tokens_per_second": round(token_count / total_time, 2) if total_time > 0 else 0
                }
            }
            
            inference_results[run_id] = result
            
            # Log completion
            log_event(run_id, "analysis_completed", {
                "tokens_generated": token_count,
                "total_time": total_time,
                "tokens_per_second": result["timing"]["tokens_per_second"]
            })
            
            # Send completion
            stats = perf_monitor.get_stats()
            yield f"data: {json.dumps({'complete': True, 'stats': stats, 'result': result})}\n\n"
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            log_event(run_id, "analysis_failed", {"error": str(e)})
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

# API Endpoints for n8n integration

@app.route('/events', methods=['GET', 'POST'])
def handle_events():
    """Handle event tracking for n8n"""
    if request.method == 'GET':
        # Return recent events
        return jsonify({
            'events': events_log[-20:],  # Last 20 events
            'count': len(events_log)
        })
    
    elif request.method == 'POST':
        # Log new event
        data = request.get_json()
        run_id = data.get('run_id', f"external_{int(time.time())}")
        stage = data.get('stage', 'unknown')
        detail = data.get('detail', {})
        
        log_event(run_id, stage, detail)
        
        return jsonify({'success': True, 'message': 'Event logged'})

@app.route('/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for video analysis (for n8n)"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get video file
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        video_data = video_file.read()
        
        # Get parameters
        prompt = request.form.get('prompt', 'Detect notable events and summarize what happens in this video.')
        frames = min(int(request.form.get('frames', 8)), 16)
        resize = int(request.form.get('resize', 336))
        max_tokens = min(int(request.form.get('max_tokens', 120)), 300)
        run_id = request.form.get('run_id', f"api_{int(time.time())}")
        
        # Log start
        log_event(run_id, "api_analysis_started", {
            "filename": video_file.filename,
            "frames": frames,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
        })
        
        start_time = time.time()
        
        # Extract frames and analyze
        frame_data = chronicon_engine.sample_frames_from_bytes(video_data, frames, resize)
        analysis = chronicon_engine.analyze(frame_data, prompt, max_tokens)
        
        total_time = time.time() - start_time
        
        # Create result
        result = {
            "run_id": run_id,
            "video_filename": video_file.filename,
            "prompt": prompt,
            "num_frames": len(frame_data),
            "resize": resize,
            "max_tokens": max_tokens,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
            "timing": {
                "total_time": round(total_time, 2),
                "tokens_per_second": round(len(analysis.split()) / total_time, 2) if total_time > 0 else 0
            }
        }
        
        # Store result
        inference_results[run_id] = result
        
        # Log completion
        log_event(run_id, "api_analysis_completed", {
            "total_time": total_time,
            "analysis_length": len(analysis)
        })
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        error_msg = f"API analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        if 'run_id' in locals():
            log_event(run_id, "api_analysis_failed", {"error": str(e)})
        
        return jsonify({'error': error_msg}), 500

@app.route('/results/<run_id>')
def get_result(run_id):
    """Get analysis result by run_id"""
    if run_id in inference_results:
        return jsonify(inference_results[run_id])
    else:
        return jsonify({'error': 'Result not found'}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(chronicon_engine.device) if chronicon_engine else 'unknown',
        'events_count': len(events_log),
        'results_count': len(inference_results),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/performance')
def performance_stats():
    """Performance statistics"""
    stats = {
        'model_loaded': model_loaded,
        'device': str(chronicon_engine.device) if chronicon_engine else 'unknown',
        'cuda_available': torch.cuda.is_available(),
        'system_memory': f"{psutil.virtual_memory().percent}%",
        'events_logged': len(events_log),
        'results_stored': len(inference_results)
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_name': torch.cuda.get_device_name(),
            'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            'gpu_memory_percent': f"{torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%"
        })
    
    return jsonify(stats)

if __name__ == '__main__':
    print("🎥 Chronicon - Video Event Detection Web App")
    print("=" * 60)
    
    # Load model
    if not load_chronicon_model():
        print("\n❌ Model loading failed!")
        print("💡 Quick fixes:")
        print("  1. Check CUDA installation")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Download model manually")
        sys.exit(1)
    
    # Initialize
    app.current_video_data = None
    app.current_video_filename = None
    
    print(f"\n🚀 Starting Chronicon web server...")
    print(f"🌐 Web interface: http://localhost:5000")
    print(f"📊 Health check: http://localhost:5000/health")
    print(f"📈 Performance: http://localhost:5000/performance")
    print(f"🔗 Events API: http://localhost:5000/events")
    print(f"🎬 Analysis API: http://localhost:5000/analyze")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)