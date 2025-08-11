import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

class VideoAnalyzer:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", verbose: bool = False):
        """
        Initialize the VideoAnalyzer with SmolVLM2 model for video understanding.
        
        Args:
            model_id: HuggingFace model identifier. Options:
                - "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" (default, faster, smaller)
                - "HuggingFaceTB/SmolVLM2-2.2B-Instruct" (larger, better quality)
            verbose: Enable verbose output
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.verbose = verbose
        # TEMPORARY FIX: Force CPU to avoid device mismatch issues with SmolVLM quantization
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        if self.verbose:
            if torch.cuda.is_available():
                print("⚠️ CUDA available but using CPU to avoid device mismatch issues with SmolVLM quantization")
        
        # Model specifications
        self.model_specs = {
            "HuggingFaceTB/SmolVLM2-256M-Video-Instruct": {
                "size": "256M",
                "type": "Video-Instruct",
                "memory_usage": "~2GB",
                "speed": "Fast",
                "quality": "Good"
            },
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct": {
                "size": "2.2B",
                "type": "Instruct",
                "memory_usage": "~8GB",
                "speed": "Slower",
                "quality": "Excellent"
            }
        }
        
    def get_available_models(self):
        """Get list of available models with their specifications."""
        return self.model_specs
        
    def get_model_info(self, model_id: str = None):
        """Get information about a specific model."""
        if model_id is None:
            model_id = self.model_id
            
        if model_id in self.model_specs:
            return self.model_specs[model_id]
        else:
            return {"error": f"Model {model_id} not found in specifications"}
        
    def check_memory_requirements(self, model_id: str = None):
        """Check if the system has enough memory for the specified model."""
        if model_id is None:
            model_id = self.model_id
            
        if model_id not in self.model_specs:
            return {"error": f"Model {model_id} not found"}
            
        # Get GPU memory info if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            model_memory = self.model_specs[model_id]["memory_usage"]
            
            return {
                "gpu_memory_gb": round(gpu_memory, 2),
                "model_memory_estimate": model_memory,
                "sufficient_memory": gpu_memory >= 4 if "2.2B" in model_id else gpu_memory >= 2
            }
        else:
            return {
                "gpu_memory_gb": 0,
                "model_memory_estimate": self.model_specs[model_id]["memory_usage"],
                "sufficient_memory": False,
                "note": "CUDA not available, will use CPU"
            }
        
    def load_model(self):
        """Load the model with 4-bit quantization for efficiency."""
        if self.verbose:
            print(f"📦 Loading model {self.model_id}...")
            print(f"   - Device: {self.device}")
            print(f"   - Quantization: 4-bit")
            print(f"   - Trust remote code: True")
            
            # Show model specifications
            model_info = self.get_model_info()
            print(f"   - Model size: {model_info['size']}")
            print(f"   - Model type: {model_info['type']}")
            print(f"   - Estimated memory: {model_info['memory_usage']}")
            print(f"   - Expected quality: {model_info['quality']}")
            print(f"   - Expected speed: {model_info['speed']}")
        
        # Check memory requirements
        memory_check = self.check_memory_requirements()
        if self.verbose:
            print(f"   - GPU memory: {memory_check['gpu_memory_gb']} GB")
            print(f"   - Sufficient memory: {memory_check['sufficient_memory']}")
        
        # Configure quantization based on device and model size
        if self.device == "cpu":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32
            )
            if self.verbose:
                print(f"   - CPU quantization: nf4")
        else:
            # For larger models, use more aggressive quantization
            if "2.2B" in self.model_id:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                if self.verbose:
                    print(f"   - GPU quantization: fp4 with double quantization")
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                if self.verbose:
                    print(f"   - GPU quantization: fp4")
        
        # Load model - try quantization first, fallback if device issues
        try:
            # Use explicit device mapping to ensure all components go to the same device
            if self.device == "cuda":
                device_map = "cuda:0"
            else:
                device_map = "cpu"
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=bnb_config
            ).eval()
            
            if self.verbose:
                print(f"   - Quantized model loaded successfully on {device_map}")
                
        except Exception as e:
            if self.verbose:
                print(f"   - Warning: Quantization failed: {str(e)}")
                print(f"   - Loading without quantization for compatibility...")
            
            # Fallback without quantization for device compatibility
            if self.device == "cuda":
                device_map = "cuda:0"
            else:
                device_map = "cpu"
                
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            ).eval()
            
            if self.verbose:
                print(f"   - Standard model loaded successfully on {device_map}")
        
        if self.verbose:
            print(f"   - Model loaded successfully")
            print(f"   - Loading processor...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        
        if self.verbose:
            print(f"✅ Model and processor loaded successfully on {self.device}")
            
            # Show final model info
            if hasattr(self.model, 'config'):
                print(f"   - Model config: {self.model.config.model_type}")
                if hasattr(self.model.config, 'num_parameters'):
                    print(f"   - Parameters: {self.model.config.num_parameters:,}")
            
            # Verify quantization status
            if hasattr(self.model, 'is_loaded_in_4bit'):
                print(f"   - 4-bit quantization: {self.model.is_loaded_in_4bit}")
            else:
                print(f"   - 4-bit quantization: Unknown (may not be supported)")
            
            # Check processor capabilities
            self.check_processor_capabilities()
        
    def reload_model(self):
        """Force reload the model with current parameters."""
        if self.verbose:
            print(f"🔄 Reloading model {self.model_id}...")
        
        # Clear existing model
        self.model = None
        self.processor = None
        
        # Reload
        self.load_model()
        
    def ensure_model_loaded(self):
        """Ensure model is loaded with current parameters."""
        if self.model is None:
            self.load_model()
        
    def sample_frames(self, video_path: str, num_frames: int = 8, resize: int = 336) -> List:
        """
        Extract evenly spaced frames from a video.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            resize: Target size for frame resizing
            
        Returns:
            List of RGB frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.verbose:
            print(f"🎬 Opening video file: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.verbose:
            print(f"   - Total frames: {frame_count:,}")
            print(f"   - FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
            print(f"   - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"   - Duration: {frame_count / cap.get(cv2.CAP_PROP_FPS):.1f} seconds")
        
        if frame_count == 0:
            raise ValueError(f"Invalid video file: {video_path}")
        
        if self.verbose:
            print(f"   - Sampling {num_frames} frames...")
            print(f"   - Frame interval: {frame_count / num_frames:.1f} frames")
        
        # Use improved evenly spaced indices function
        def evenly_spaced_indices(total, n):
            xs = np.linspace(0, total - 1, n)
            idxs = np.round(xs).astype(int)
            # enforce strictly nondecreasing without duplicates
            for i in range(1, len(idxs)):
                if idxs[i] <= idxs[i-1]:
                    idxs[i] = min(idxs[i-1] + 1, total - 1)
            return idxs.tolist()
        
        # Use proper rounding to avoid duplicates and ensure even distribution
        if num_frames == 1:
            idxs = [frame_count // 2]  # Middle frame for single frame
        else:
            idxs = evenly_spaced_indices(frame_count, num_frames)
        
        frames = []
        
        for i, idx in enumerate(idxs):
            if self.verbose:
                print(f"     * Frame {i+1}/{len(idxs)}: Extracting frame {idx:,}")
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if self.verbose:
                    print(f"       ⚠️ Failed to read frame {idx}")
                continue
                
            h, w = frame.shape[:2]
            scale = resize / max(h, w)
            new_w, new_h = int(w*scale), int(h*scale)
            
            if self.verbose:
                print(f"       - Original: {w}x{h}")
                print(f"       - Resized: {new_w}x{new_h}")
                print(f"       - Scale: {scale:.3f}")
            
            frame = cv2.resize(frame, (new_w, new_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if self.verbose:
                print(f"       ✅ Frame {i+1} processed")
        
        cap.release()
        
        if self.verbose:
            print(f"✅ Successfully extracted {len(frames)} frames")
        
        return frames
        
    def analyze_video(self, video_path: str, prompt: str = "Describe the main events in these frames.", 
                     num_frames: int = 8, resize: int = 336, max_tokens: int = 100) -> Dict[str, Any]:
        """
        Analyze a video and generate a description.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt for the model
            num_frames: Number of frames to sample
            resize: Target size for frame resizing
            max_tokens: Maximum tokens for generation
            
        Returns:
            Dictionary containing analysis results
        """
        # Always ensure model is loaded with current parameters
        self.ensure_model_loaded()
        
        if self.verbose:
            print(f"🎥 Starting video analysis...")
            print(f"   - Video: {video_path}")
            print(f"   - Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"   - Frames: {num_frames}")
            print(f"   - Resize: {resize}x{resize}")
            print(f"   - Max tokens: {max_tokens}")
        
        # Sample frames from video
        if self.verbose:
            print(f"\n📹 Step 1: Extracting frames...")
            
        frames = self.sample_frames(video_path, num_frames, resize)
        
        if not frames:
            return {"error": "No frames could be extracted from the video"}
        
        # Validate input format
        if not self.validate_video_input(frames, prompt):
            return {"error": "Invalid video input format"}
        
        if self.verbose:
            print(f"✅ Frame extraction complete: {len(frames)} frames")
            print(f"\n🤖 Step 2: Preparing model input...")
            print(f"   - Creating message template...")
            print(f"   - Adding video frames...")
            print(f"   - Adding text prompt...")
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "frames": frames},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        if self.verbose:
            print(f"   - Message structure created")
            print(f"   - Video frames: {len(frames)}")
            print(f"   - Text prompt: {len(prompt)} characters")
        
        # Process inputs
        if self.verbose:
            print(f"\n🔧 Step 3: Processing with model...")
            print(f"   - Applying chat template...")
            print(f"   - Binding frames to prompt...")
            print(f"   - Converting to tensors...")
            print(f"   - Moving to device: {self.device}")
        
        # CRITICAL FIX: We need to separate text processing from frame processing
        # Then combine them properly for SmolVLM
        
        if self.verbose:
            print(f"   - Processing text and frames separately...")
            print(f"   - Message contains {len(frames)} video frames")
        
        # Step 1: Get the text prompt without frames
        text_only_messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        try:
            # Get just the text template
            prompt_text = self.processor.apply_chat_template(
                text_only_messages,
                add_generation_prompt=True
            )
            
            if self.verbose:
                print(f"   - Generated text prompt: {len(prompt_text)} chars")
                print(f"   - Prompt preview: {prompt_text[:100]}...")
            
            # Step 2: Insert image tokens to match frame count  
            # SmolVLM expects image tokens in text to match number of frames
            image_tokens = "<image>" * len(frames)
            prompt_with_images = prompt_text.replace("<|im_start|>User:", f"<|im_start|>User: {image_tokens}")
            
            if self.verbose:
                print(f"   - Added {len(frames)} image tokens to prompt")
                print(f"   - Modified prompt preview: {prompt_with_images[:100]}...")
            
            # Now process text + frames together to get pixel_values + input_ids
            # Ensure frames are in the right format for quantized model
            proc_inputs = self.processor(
                text=prompt_with_images,
                images=frames,  # Use images parameter for frame sequence
                return_tensors="pt"
            )
            
            # CRITICAL FIX: Convert pixel_values to float16 and move to correct device
            if 'pixel_values' in proc_inputs:
                proc_inputs['pixel_values'] = proc_inputs['pixel_values'].to(device=self.device, dtype=torch.float16)
                if self.verbose:
                    print(f"   - Converted pixel_values to: {proc_inputs['pixel_values'].dtype} on {proc_inputs['pixel_values'].device}")
            
            if self.verbose:
                print(f"   - Successfully combined text + frames")
                
        except Exception as e:
            if self.verbose:
                print(f"   - Error combining text + frames: {str(e)}")
                print(f"   - Trying direct message processing...")
                
            # Fallback: try the original approach
            try:
                proc_inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            except Exception as e2:
                if self.verbose:
                    print(f"   - Direct message processing failed: {str(e2)}")
                    print(f"   - Using simple text fallback...")
                
                # Last resort: simple text only
                simple_prompt = f"User: {prompt}\n\nAssistant:"
                proc_inputs = self.processor.tokenizer(
                    simple_prompt,
                    return_tensors="pt"
                )
        
        if self.verbose:
            print(f"   - Input type: {type(proc_inputs)}")
            if isinstance(proc_inputs, dict):
                print(f"   - Input keys: {list(proc_inputs.keys())}")
                for key, value in proc_inputs.items():
                    if hasattr(value, 'shape'):
                        print(f"     - {key} shape: {value.shape}")
                    if hasattr(value, 'device'):
                        print(f"     - {key} device: {value.device}")
                        
                # Validate that we have both text and visual inputs
                has_input_ids = 'input_ids' in proc_inputs
                has_pixel_values = 'pixel_values' in proc_inputs
                print(f"   - Has input_ids (text): {has_input_ids}")
                print(f"   - Has pixel_values (frames): {has_pixel_values}")
                
                if not has_pixel_values:
                    print(f"   ⚠️ WARNING: No pixel_values found - frames may not be bound to model!")
                    
            elif isinstance(proc_inputs, str):
                print(f"   ⚠️ WARNING: proc_inputs is string - frames not processed!")
                # Convert string to tensors
                proc_inputs = self.processor.tokenizer(proc_inputs, return_tensors="pt")
                print(f"   - Tokenized to: {type(proc_inputs)}")
                
            print(f"   - Starting generation...")
        
        # Move tensors to device with proper dtype for quantized model
        if isinstance(proc_inputs, dict):
            for k, v in proc_inputs.items():
                if hasattr(v, "to"):
                    # Ensure all tensors go to the same device with proper dtype
                    if k == 'pixel_values':
                        # Pixel values need to match model precision (float16 for quantized model)
                        proc_inputs[k] = v.to(self.device, dtype=torch.float16)
                        if self.verbose:
                            print(f"     - {k} converted to: {proc_inputs[k].dtype} on {proc_inputs[k].device}")
                    elif k == 'input_ids' or k == 'attention_mask':
                        # Text tokens should be long integers on the target device
                        proc_inputs[k] = v.to(self.device)
                        if self.verbose:
                            print(f"     - {k}: {proc_inputs[k].dtype} on {proc_inputs[k].device}")
                    else:
                        # Any other tensors should also be moved to target device
                        proc_inputs[k] = v.to(self.device)
                        if self.verbose and hasattr(v, 'dtype'):
                            print(f"     - {k}: {proc_inputs[k].dtype} on {proc_inputs[k].device}")
        else:
            if hasattr(proc_inputs, 'to'):
                proc_inputs = proc_inputs.to(self.device)
        
        # CRITICAL FIX: Ensure the model itself is on the correct device
        if self.model is not None:
            # Force model to be on the target device to prevent device mismatch
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
                if self.verbose:
                    print(f"     - Model moved to: {self.device}")
                    
        # Additional safety check: verify all input tensors are on the same device
        if isinstance(proc_inputs, dict) and self.verbose:
            devices = set()
            for k, v in proc_inputs.items():
                if hasattr(v, 'device'):
                    devices.add(str(v.device))
            print(f"     - All input tensors on devices: {devices}")
            if len(devices) > 1:
                print(f"     ⚠️ WARNING: Multiple devices detected in inputs!")
            else:
                print(f"     ✅ All inputs on single device: {list(devices)[0]}")
        
        # Generate response with greedy decoding for evaluation
        with torch.no_grad():
            if self.verbose:
                print(f"   - Generating response (max_new_tokens={max_tokens})...")
                
            try:
                outputs = self.model.generate(
                    **proc_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e) and self.device == "cuda":
                    if self.verbose:
                        print(f"   ⚠️ Device mismatch error on GPU, falling back to CPU...")
                        print(f"   - Moving model and inputs to CPU...")
                    
                    # Fallback to CPU
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    
                    # Move inputs to CPU
                    if isinstance(proc_inputs, dict):
                        for k, v in proc_inputs.items():
                            if hasattr(v, "to"):
                                proc_inputs[k] = v.to("cpu")
                    
                    if self.verbose:
                        print(f"   - Retrying generation on CPU...")
                    
                    # Retry on CPU
                    outputs = self.model.generate(
                        **proc_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        num_beams=1,
                        repetition_penalty=1.05,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    if self.verbose:
                        print(f"   ✅ Generation successful on CPU")
                else:
                    # Re-raise if it's a different error
                    raise e
            
            if self.verbose:
                print(f"   - Generation complete")
                print(f"   - Output shape: {outputs.shape}")
        
        if self.verbose:
            print(f"\n📝 Step 4: Decoding response...")
            
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        if self.verbose:
            print(f"   - Response length: {len(response)} characters")
            print(f"   - Response preview: {response[:100]}...")
        
        # Extract only the assistant's response (remove the user prompt)
        if "Assistant:" in response:
            # Find the assistant's response
            assistant_start = response.find("Assistant:")
            if assistant_start != -1:
                assistant_response = response[assistant_start + len("Assistant:"):].strip()
                if self.verbose:
                    print(f"   - Extracted assistant response: {assistant_response[:100]}...")
                response = assistant_response
            else:
                if self.verbose:
                    print(f"   - Could not find 'Assistant:' in response, using full response")
        else:
            if self.verbose:
                print(f"   - No 'Assistant:' found in response, using full response")
        
        result = {
            "video_path": video_path,
            "prompt": prompt,
            "num_frames": len(frames),
            "resize": resize,
            "max_tokens": max_tokens,
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_id
        }
        
        if self.verbose:
            print(f"✅ Analysis complete!")
            print(f"   - Result keys: {list(result.keys())}")
        
        return result
        
    def analyze_video_batch(self, video_paths: List[str], 
                           prompt: str = "Describe the main events in these frames.",
                           num_frames: int = 8) -> List[Dict[str, Any]]:
        """
        Analyze multiple videos in batch.
        
        Args:
            video_paths: List of video file paths
            prompt: Text prompt for the model
            num_frames: Number of frames to sample per video
            
        Returns:
            List of analysis results
        """
        results = []
        for i, video_path in enumerate(video_paths):
            if self.verbose:
                print(f"\n🎬 Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            try:
                result = self.analyze_video(video_path, prompt, num_frames)
                results.append(result)
                
                if self.verbose:
                    print(f"✅ Video {i+1} completed")
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error processing video {i+1}: {str(e)}")
                    
                results.append({
                    "video_path": video_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        return results
        
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis: Analysis results dictionary
            output_path: Path to save the JSON file
        """
        if self.verbose:
            print(f"💾 Saving analysis to: {output_path}")
            print(f"   - Analysis keys: {list(analysis.keys())}")
            print(f"   - Analysis size: {len(json.dumps(analysis))} characters")
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        if self.verbose:
            print(f"✅ Analysis saved successfully")
            print(f"   - File size: {os.path.getsize(output_path)} bytes")

    def validate_video_input(self, frames: List, prompt: str) -> bool:
        """
        Validate that the video input format is correct for the model.
        
        Args:
            frames: List of video frames
            prompt: Text prompt
            
        Returns:
            True if input is valid, False otherwise
        """
        if not frames:
            if self.verbose:
                print("❌ Error: No frames provided")
            return False
            
        if len(frames) == 0:
            if self.verbose:
                print("❌ Error: Empty frame list")
            return False
            
        # Check frame format
        for i, frame in enumerate(frames):
            if not isinstance(frame, (list, tuple, np.ndarray)):
                if self.verbose:
                    print(f"❌ Error: Frame {i} is not a valid image format")
                return False
                
            if hasattr(frame, 'shape') and len(frame.shape) != 3:
                if self.verbose:
                    print(f"❌ Error: Frame {i} is not a 3D array (RGB)")
                return False
        
        if not prompt or len(prompt.strip()) == 0:
            if self.verbose:
                print("❌ Error: Empty prompt")
            return False
            
        if self.verbose:
            print(f"✅ Input validation passed:")
            print(f"   - Frames: {len(frames)}")
            print(f"   - Frame shapes: {[f.shape if hasattr(f, 'shape') else 'unknown' for f in frames[:3]]}")
            print(f"   - Prompt length: {len(prompt)} characters")
            
        return True

    def check_processor_capabilities(self):
        """
        Check if the processor supports video input properly.
        
        Returns:
            Dictionary with capability information
        """
        capabilities = {
            "supports_video": False,
            "supports_chat_template": False,
            "processor_type": type(self.processor).__name__,
            "available_methods": []
        }
        
        if hasattr(self.processor, 'apply_chat_template'):
            capabilities["supports_chat_template"] = True
            capabilities["available_methods"].append("apply_chat_template")
            
        # Check for video-specific methods
        if hasattr(self.processor, 'process_video'):
            capabilities["supports_video"] = True
            capabilities["available_methods"].append("process_video")
            
        if hasattr(self.processor, 'process_images'):
            capabilities["available_methods"].append("process_images")
            
        if hasattr(self.processor, 'tokenizer'):
            capabilities["available_methods"].append("tokenizer")
            
        if self.verbose:
            print(f"🔍 Processor capabilities:")
            print(f"   - Type: {capabilities['processor_type']}")
            print(f"   - Supports video: {capabilities['supports_video']}")
            print(f"   - Supports chat template: {capabilities['supports_chat_template']}")
            print(f"   - Available methods: {capabilities['available_methods']}")
            
        return capabilities

def main():
    """Main function to demonstrate video analysis."""
    analyzer = VideoAnalyzer(verbose=True)
    
    # Test with a sample video (you'll need to provide your own video file)
    video_path = "test_video.mp4"
    
    if os.path.exists(video_path):
        print(f"Analyzing video: {video_path}")
        result = analyzer.analyze_video(video_path)
        print("\nAnalysis Result:")
        print(result["analysis"])
        
        # Save analysis
        analyzer.save_analysis(result, "video_analysis.json")
    else:
        print(f"Video file not found: {video_path}")
        print("Please provide a video file named 'test_video.mp4' or modify the path in the code.")

if __name__ == "__main__":
    main()
