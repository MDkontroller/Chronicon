#!/usr/bin/env python3
"""
Chronicon Web App - Adapted for existing chronicon_index.html template
This version runs inference.py as a subprocess and reads JSON results
"""

from flask import Flask, render_template, request, jsonify, Response
import subprocess
import json
import time
import os
import sys
import tempfile
#import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import traceback
import cv2
import psutil
import base64
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB to match template

# Global variables
events_log = []
inference_results = {}
active_processes = {}



# --- add near imports ---
from notifier import send_email

# simple text → decision
def decide_action(analysis_text: str):
    text = analysis_text.lower()
    matched = []
    if any(w in text for w in ["train", "rail", "railway", "metro", "locomotive", "db", "Deutsche", "Bahn"]):
        matched.append("train")
    if any(w in text for w in ["database", "db ", " db.", "sql", "postgres", "mysql", "mongodb"]):
        matched.append("db")

    if not matched:
        return {"label": "none", "score": 0.0, "reason": "No train/DB keywords detected."}

    # simple scoring: more matched categories → higher score
    score = min(1.0, 0.5 + 0.25 * (len(matched)-1))
    # pick priority (you can change order)
    label = "train" if "train" in matched else "db"
    return {"label": label, "score": score, "reason": f"Matched: {', '.join(matched)}"}

@app.post("/decide_and_act")
def decide_and_act():
    from flask import request, jsonify
    data = request.get_json(force=True)
    analysis = data.get("analysis", "")
    run_id = data.get("run_id", "unknown")
    video_filename = data.get("video_filename", "video")

    decision = decide_action(analysis)
    action = "none"
    email_status = None

    if decision["label"] in ("train", "db"):
        action = "send_email"
        try:
            subject = f"[Chronicon Agent] Detected {decision['label'].upper()} in {video_filename}"
            html = f"""
            <h3>Agent Decision</h3>
            <p><b>Run:</b> {run_id}</p>
            <p><b>Video:</b> {video_filename}</p>
            <p><b>Decision:</b> {decision['label']} (score {decision['score']})</p>
            <p><b>Reason:</b> {decision['reason']}</p>
            <hr/>
            <p><b>Analysis:</b></p>
            <pre style="white-space: pre-wrap; font-family: monospace;">{analysis}</pre>
            """
            send_email(subject, html)
            email_status = "sent"
        except Exception as e:
            email_status = f"error: {e}"

    result = {
        "ok": True,
        "decision": decision,
        "action": action,
        "email_status": email_status,
    }
    # also log to in-memory event log so it’s visible in UI
    log_event(run_id, "agent_decision", result)
    return jsonify(result)



def log_event(run_id: str, stage: str, detail: Dict[str, Any]):
    """Log event for tracking"""
    event = {
        "run_id": run_id,
        "stage": stage,
        "detail": detail,
        "timestamp": datetime.now().isoformat()
    }
    events_log.append(event)
    if len(events_log) > 100:
        events_log.pop(0)
    print(f"[EVENT] {stage} for {run_id}")

def save_video_to_temp(video_data: bytes, run_id: str) -> str:
    """Save video to temp file with run_id"""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"chronicon_video_{run_id}.mp4")
    with open(temp_path, 'wb') as f:
        f.write(video_data)
    return temp_path

def cleanup_temp_files(run_id: str):
    """Remove all temp files for a run"""
    temp_dir = tempfile.gettempdir()
    patterns = [
        f"chronicon_video_{run_id}.mp4",
        f"chronicon_thumb_{run_id}.jpg",
        f"inference_results_{run_id}.json"
    ]
    
    for pattern in patterns:
        temp_path = os.path.join(temp_dir, pattern)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[CLEANUP] Cleaned up: {temp_path}")
        except Exception as e:
            print(f"[WARNING] Could not clean up {temp_path}: {e}")

def get_video_info_and_thumbnail(video_path: str, run_id: str) -> Dict[str, Any]:
    """Get video information and generate thumbnail"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0
        
        # Generate thumbnail from middle frame
        thumbnail_data = None
        try:
            middle_frame = frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            if ret:
                # Resize for thumbnail
                thumb_height = 300
                aspect = width / height
                thumb_width = int(thumb_height * aspect)
                thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
                
                # Convert to base64
                _, buffer = cv2.imencode('.jpg', thumbnail)
                thumbnail_data = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[WARNING] Could not generate thumbnail: {e}")
        
        cap.release()
        
        result = {
            "frames": frames,
            "frame_count": frames,  # Template expects this key
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration": round(duration, 2),
            "resolution": f"{width}x{height}"
        }
        
        if thumbnail_data:
            result["thumbnail"] = f"data:image/jpeg;base64,{thumbnail_data}"
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def get_system_stats() -> Dict[str, str]:
    """Get current system statistics"""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = f"{memory.percent:.1f}%"
        
        # GPU memory (if available)
        gpu_memory = "N/A"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory = f"{gpu_mem_allocated:.1f}/{gpu_mem_total:.1f}GB"
        except:
            pass
        
        return {
            "memory_usage": memory_usage,
            "gpu_memory": gpu_memory
        }
    except:
        return {
            "memory_usage": "N/A",
            "gpu_memory": "N/A"
        }

def run_inference_subprocess(
    video_path: str,
    prompt: str,
    num_frames: int,
    max_tokens: int,
    min_tokens: int, 
    run_id: str,
    model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    use_quantization: bool = True,
    cuda_device: int = 0
) -> subprocess.Popen:
    """Start inference as subprocess"""
    
    # Find the inference script (try clean version first, then others)
    script_candidates = [
        "inference_script_clean.py",
        "inference_wrapper.py", 
        "inference.py"
    ]
    
    script_path = None
    for candidate in script_candidates:
        if os.path.exists(candidate):
            script_path = candidate
            break
    
    if not script_path:
        # Try current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in script_candidates:
            candidate_path = os.path.join(current_dir, candidate)
            if os.path.exists(candidate_path):
                script_path = candidate_path
                break
    
    if not script_path:
        raise FileNotFoundError(f"None of these inference scripts found: {', '.join(script_candidates)}")
    
    # Prepare command
    cmd = [
        sys.executable,  # Python executable
        script_path,
        video_path,
        "--prompt", prompt,
        "--frames", str(num_frames),
        "--max-tokens", str(max_tokens),
        "--min-tokens", str(min_tokens),
        "--model", model_name,
        "--cuda-device", str(cuda_device),
        "--verbose"
    ]
    
    if not use_quantization:
        cmd.append("--no-quant")
    
    print(f"[SUBPROCESS] Starting inference subprocess: {' '.join(cmd)}")
    
    # Set up environment for Unicode support on Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "1"  # For Python 3.6+ on Windows
    
    # Start subprocess with proper encoding
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of failing
            bufsize=1,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    except TypeError:
        # Fallback for older Python versions that don't support encoding parameter
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    
    return process

def monitor_inference_process(
    process: subprocess.Popen,
    run_id: str,
    video_filename: str
) -> Dict[str, Any]:
    """Monitor subprocess and return results"""
    
    log_event(run_id, "subprocess_started", {"pid": process.pid})
    
    # Collect output
    output_lines = []
    
    try:
        # Read output line by line with encoding error handling
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if line:
                    output_lines.append(line)
                    print(f"📄 [{run_id}] {line}")
            except UnicodeDecodeError as e:
                # Handle Unicode decode errors gracefully
                error_line = f"[Unicode decode error: {str(e)}]"
                output_lines.append(error_line)
                print(f"📄 [{run_id}] {error_line}")
                continue
        
        # Wait for process to complete
        return_code = process.wait()
        
        log_event(run_id, "subprocess_completed", {
            "return_code": return_code,
            "output_lines": len(output_lines)
        })
        
        if return_code != 0:
            error_msg = f"Inference process failed with code {return_code}"
            if output_lines:
                # Find the most relevant error message
                error_lines = [line for line in output_lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback'])]
                if error_lines:
                    error_msg += f"\nError details: {error_lines[-1]}"
                else:
                    error_msg += f"\nLast output: {output_lines[-1] if output_lines else 'No output'}"
            
            log_event(run_id, "subprocess_error", {"error": error_msg})
            return {"error": error_msg, "output": output_lines}
        
        # Look for the JSON output file
        # The inference script saves with timestamp, so we need to find the latest one
        temp_dir = tempfile.gettempdir()
        json_files = [f for f in os.listdir(temp_dir) if f.startswith("inference_results_") and f.endswith(".json")]
        
        if not json_files:
            # Look in current directory as well
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_files = [f for f in os.listdir(current_dir) if f.startswith("inference_results_") and f.endswith(".json")]
            json_dir = current_dir
        else:
            json_dir = temp_dir
        
        if not json_files:
            return {"error": "No JSON output file found", "output": output_lines}
        
        # Get the most recent JSON file
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(json_dir, x)), reverse=True)
        latest_json = json_files[0]
        json_path = os.path.join(json_dir, latest_json)
        
        print(f"[RESULTS] Reading results from: {json_path}")
        
        # Read JSON results
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # Add metadata
            result["run_id"] = run_id
            result["video_filename"] = video_filename
            result["subprocess_output"] = output_lines
            result["json_file"] = latest_json
            
            log_event(run_id, "results_loaded", {
                "json_file": latest_json,
                "analysis_length": len(result.get("analysis", ""))
            })
            
            # Clean up the JSON file after reading
            try:
                os.remove(json_path)
                print(f"[CLEANUP] Cleaned up JSON file: {json_path}")
            except:
                pass
            
            return result
            
        except Exception as e:
            return {"error": f"Could not read JSON file: {e}", "json_path": json_path, "output": output_lines}
    
    except Exception as e:
        log_event(run_id, "monitoring_error", {"error": str(e)})
        return {"error": f"Error monitoring process: {e}", "output": output_lines}

def analyze_with_subprocess(
    video_path: str,
    prompt: str,
    max_tokens: int,
    min_tokens: int,
    num_frames: int,
    run_id: str,
    video_filename: str
):
    """Run analysis via subprocess and yield progress"""
    
    try:
        log_event(run_id, "analysis_start", {
            "prompt": prompt,
            "frames": num_frames,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens	
        })
        
        yield f"data: {json.dumps({'status': 'Starting inference process...'})}\n\n"
        
        # Start subprocess
        process = run_inference_subprocess(
            video_path=video_path,
            prompt=prompt,
            num_frames=num_frames,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            run_id=run_id,
            use_quantization=True,  # You can make this configurable
            cuda_device=0
        )
        
        # Store process reference
        active_processes[run_id] = process
        
        yield f"data: {json.dumps({'status': f'Analyzing {num_frames} frames...', 'pid': process.pid})}\n\n"
        
        # Monitor in a separate thread to allow for progress updates
        result_container = {}
        
        def monitor_thread():
            result_container['result'] = monitor_inference_process(process, run_id, video_filename)
        
        monitor = threading.Thread(target=monitor_thread)
        monitor.start()
        
        # Provide periodic updates while waiting
        start_time = time.time()
        last_update = start_time
        
        while monitor.is_alive():
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Send update every 3 seconds
            if current_time - last_update >= 3:
                yield f"data: {json.dumps({'status': f'Processing... ({elapsed:.0f}s elapsed)'})}\n\n"
                last_update = current_time
            
            time.sleep(1)
        
        monitor.join()
        
        # Clean up process reference
        if run_id in active_processes:
            del active_processes[run_id]
        
        # Get results
        result = result_container.get('result', {'error': 'No result from monitoring thread'})
        
        if 'error' in result:
            log_event(run_id, "analysis_error", {"error": result['error']})
            yield f"data: {json.dumps({'error': result['error']})}\n\n"
        else:
            # Stream the analysis text word by word for nice effect
            analysis_text = result.get('analysis', '')
            if analysis_text:
                words = analysis_text.split()
                
                yield f"data: {json.dumps({'status': f'Streaming {len(words)} words...'})}\n\n"
                
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'text': word + ' '})}\n\n"
                    time.sleep(0.03)  # Small delay for streaming effect
            
            # Calculate tokens generated (approximate)
            tokens_generated = len(analysis_text.split())
            inference_time = result.get('timing', {}).get('inference_time', 0)
            tokens_per_second = round(tokens_generated / inference_time, 2) if inference_time > 0 else 0
            
            # Update timing with token stats
            if 'timing' in result:
                result['timing']['tokens_generated'] = tokens_generated
                result['timing']['tokens_per_second'] = tokens_per_second
            
            # Get current system stats
            stats = get_system_stats()
            
            # Store results
            inference_results[run_id] = result
            log_event(run_id, "analysis_complete", {
                "analysis_length": len(analysis_text),
                "total_time": result.get('timing', {}).get('total_time', 0),
                "tokens_generated": tokens_generated
            })
            
            yield f"data: {json.dumps({'complete': True, 'result': result, 'stats': stats})}\n\n"
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Analysis error: {error_msg}")
        traceback.print_exc()
        
        log_event(run_id, "analysis_exception", {"error": error_msg})
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    finally:
        # Clean up temp files
        cleanup_temp_files(run_id)
        
        # Clean up process reference
        if run_id in active_processes:
            try:
                process = active_processes[run_id]
                if process.poll() is None:  # Still running
                    process.terminate()
                del active_processes[run_id]
            except:
                pass

# Routes

@app.route('/')
def index():
    """Main page - uses the existing chronicon_index.html template"""
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload - matches template expectations"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Read video data
        video_data = file.read()
        file_size_mb = len(video_data) / (1024 * 1024)
        
        if file_size_mb > 48:  # Leave some margin for the 50MB limit
            return jsonify({'error': f'File too large: {file_size_mb:.1f}MB (max 48MB)'}), 400
        
        # Store video data in app context
        app.current_video_data = video_data
        app.current_video_filename = file.filename
        
        # Create a temporary file to get video info
        run_id = f"upload_{int(time.time())}"
        temp_path = save_video_to_temp(video_data, run_id)
        
        try:
            video_info = get_video_info_and_thumbnail(temp_path, run_id)
        finally:
            cleanup_temp_files(run_id)
        
        if 'error' in video_info:
            return jsonify({'error': f'Invalid video file: {video_info["error"]}'}), 400
        
        # Response format expected by template
        response_data = {
            'success': True,
            'filename': file.filename,
            'size_mb': round(file_size_mb, 2),
            **video_info
        }
        
        log_event("upload", "video_uploaded", response_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze')
def analyze():
    """Stream analysis via subprocess - matches template expectations"""
    
    if not hasattr(app, 'current_video_data') or not app.current_video_data:
        return Response("No video uploaded", mimetype='text/plain', status=400)
    
    # Get parameters (template sends these)
    prompt = request.args.get('prompt', 'Detect notable events and summarize what happens in this video. Focus on key actions, objects, and any significant moments.')
    frames = max(1, min(int(request.args.get('frames', 8)), 16))
    max_tokens = max(250, min(int(request.args.get('max_tokens', 250)), 500))
    min_tokens = max(200, min(int(request.args.get('min_tokens', 200)), 200))
    run_id = request.args.get('run_id', f"web_{int(time.time())}")
    
    def generate():
        temp_video_path = None
        try:
            # Save video to temporary file
            temp_video_path = save_video_to_temp(app.current_video_data, run_id)
            
            # Run analysis via subprocess
            yield from analyze_with_subprocess(
                video_path=temp_video_path,
                prompt=prompt,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                num_frames=frames,
                run_id=run_id,
                video_filename=app.current_video_filename
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Generate error: {error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
        finally:
            # Clean up is handled in analyze_with_subprocess
            pass
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health')
def health():
    """Health check endpoint"""
    
    # Check if inference script exists
    script_exists = os.path.exists("inference.py")
    
    # Check Python/CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.current_device() if cuda_available else None
        cuda_devices = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        cuda_available = False
        cuda_device = None
        cuda_devices = 0
    
    # System stats
    stats = get_system_stats()
    
    return jsonify({
        'status': 'ok',
        'inference_script_exists': script_exists,
        'active_processes': len(active_processes),
        'events_logged': len(events_log),
        'inference_results': len(inference_results),
        'cuda_available': cuda_available,
        'cuda_device': cuda_device,
        'cuda_devices': cuda_devices,
        'temp_dir': tempfile.gettempdir(),
        **stats
    })

@app.route('/debug')
def debug():
    """Debug information"""
    
    info = {
        'python_version': sys.version,
        'current_directory': os.getcwd(),
        'script_exists': os.path.exists("inference.py"),
        'temp_directory': tempfile.gettempdir(),
        'active_processes': list(active_processes.keys()),
        'recent_events': events_log[-10:] if events_log else [],
        'has_video_data': hasattr(app, 'current_video_data') and app.current_video_data is not None
    }
    
    if hasattr(app, 'current_video_filename'):
        info['current_video'] = app.current_video_filename
    
    # Try to get system info
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['current_gpu'] = torch.cuda.current_device()
            try:
                info['gpu_name'] = torch.cuda.get_device_name()
                info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            except:
                pass
    except ImportError:
        info['torch_available'] = False
    
    try:
        import cv2
        info['opencv_version'] = cv2.__version__
    except ImportError:
        info['opencv_available'] = False
    
    # Add system stats
    info.update(get_system_stats())
    
    return jsonify(info)

@app.route('/cancel/<run_id>')
def cancel_analysis(run_id):
    """Cancel a running analysis"""
    try:
        if run_id in active_processes:
            process = active_processes[run_id]
            process.terminate()
            del active_processes[run_id]
            log_event(run_id, "cancelled", {"user_requested": True})
            cleanup_temp_files(run_id)
            return jsonify({"success": True, "message": f"Cancelled analysis {run_id}"})
        else:
            return jsonify({"error": f"No active process found for {run_id}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results/<run_id>')
def get_results(run_id):
    """Get results for a specific run"""
    if run_id in inference_results:
        return jsonify(inference_results[run_id])
    else:
        return jsonify({"error": "Results not found"}), 404

if __name__ == '__main__':
    print("Chronicon Web App - Adapted for Existing Template")
    print("=" * 60)
    
    # Check requirements
    print(f"[INFO] Python: {sys.version}")
    
    # Check if inference script exists
    script_path = "inference.py"
    if os.path.exists(script_path):
        print(f"[SUCCESS] Found inference script: {script_path}")
    else:
        print(f"[ERROR] Inference script not found: {script_path}")
        print("   Make sure inference.py is in the same directory")
    
    # Check if template exists
    template_path = "templates/index.html"
    #template_path = "templates/back_up_index.html"
    if os.path.exists(template_path):
        print(f"[SUCCESS] Found template: {template_path}")
    else:
        print(f"[ERROR] Template not found: {template_path}")
        print("   Make sure to create templates/chronicon_index.html")
    
    # Check dependencies
    try:
        import torch
        print(f"[SUCCESS] PyTorch: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU count: {torch.cuda.device_count()}")
            print(f"[INFO] Current GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("[WARNING] PyTorch not found (needed for health checks)")
    
    try:
        import cv2
        print(f"[SUCCESS] OpenCV: {cv2.__version__}")
    except ImportError:
        print("[ERROR] OpenCV not found (required for video processing)")
        sys.exit(1)
    
    try:
        import psutil
        print(f"[SUCCESS] psutil: {psutil.__version__}")
    except ImportError:
        print("[WARNING] psutil not found (optional, for system stats)")
    
    # Initialize app state
    app.current_video_data = None
    app.current_video_filename = None
    
    print(f"\n[STARTUP] Starting server...")
    print(f"[INFO] Web interface: http://localhost:5000")
    print(f"[INFO] Health check: http://localhost:5000/health")
    print(f"[INFO] Debug info: http://localhost:5000/debug")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"[ERROR] Server startup error: {e}")
        traceback.print_exc()