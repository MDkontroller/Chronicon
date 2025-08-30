#!/usr/bin/env python3
"""
Chronicon – SmolVLM Standalone GPU Inference (CUDA-safe)
- Forces the entire model to a single CUDA device (no auto sharding)
- Optional 4-bit quantization with bitsandbytes
- Audits and fixes CPU-straggler params/buffers to prevent device mismatch
- Bucketize shim to stop cuda/cpu boundary mismatches in SmolVLM vision
- Clean CPU fallback path


python inference_script_gpu_fixed.py test_video.mp4 --frames 8 -v
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# --------- HARDENING: device-safe bucketize shim (prevents cuda/cpu mix) ----------
if not hasattr(torch, "_orig_bucketize"):
    torch._orig_bucketize = torch.bucketize
    def _bucketize_device_safe(input, boundaries, *args, **kwargs):
        # If input is CUDA and boundaries is CPU, move boundaries to input's device
        if torch.is_tensor(boundaries) and input.is_cuda and not boundaries.is_cuda:
            boundaries = boundaries.to(input.device)
        return torch._orig_bucketize(input, boundaries, *args, **kwargs)
    torch.bucketize = _bucketize_device_safe
# ----------------------------------------------------------------------------------


class ChroniconVLM:
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

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Please run on a machine with a CUDA-capable GPU.")

        self.device = torch.device(f"cuda:{self.cuda_device_index}")
        self.fp_dtype = torch.float16  # preferred on GPU
        self.cpu_dtype = torch.float32

        # Encourage all factory-created tensors in 3rd-party code to land on CUDA
        try:
            torch.set_default_device(self.device)
        except Exception:
            # Older PyTorch: harmless to ignore
            pass

    # ---------------------- Loading ----------------------

    def load(self):
        if self.verbose:
            print(f"📦 Loading model {self.model_id}")
            print(f"   - Torch: {torch.__version__} | CUDA: {torch.version.cuda}")
            print(f"   - GPU[{self.cuda_device_index}]: {torch.cuda.get_device_name(self.cuda_device_index)}")
            print(f"   - 4-bit quantization: {self.quant_4bit}")

        tried = []

        try:
            if self.quant_4bit:
                self._load_gpu_nf4()
            else:
                self._load_gpu_fp16()
            tried.append("GPU")
        except Exception as e:
            if self.verbose:
                print(f"❌ GPU load failed ({'4-bit' if self.quant_4bit else 'fp16'}): {e}")
            # try non-quantized if quantized failed
            if self.quant_4bit:
                try:
                    self._load_gpu_fp16()
                    tried.append("GPU-fp16")
                except Exception as e2:
                    if self.verbose:
                        print(f"❌ GPU fp16 load failed: {e2}")

        if self.model is None:
            # CPU fallback
            if self.verbose:
                print("⚠️ Falling back to CPU...")
            self._load_cpu()
            tried.append("CPU")

        # Processor
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # ensure a pad token for generate()
        if getattr(self.processor, "tokenizer", None):
            tok = self.processor.tokenizer
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

        if self.verbose:
            print(f"✅ Load complete on: {self._model_primary_device()} (tried: {', '.join(tried)})")

        # Final audit (only meaningful if on CUDA)
        if self._is_cuda():
            self._audit_and_fix_devices()
            self._assert_all_cuda()

    def _load_gpu_nf4(self):
        # Force single-GPU placement: no "auto"
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
            device_map={"": self.cuda_device_index},  # single device, no sharding
            low_cpu_mem_usage=True,
        ).eval()
                   # Try to JIT compile with PyTorch 2.x
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
            if self.verbose:
                print("⚡ torch.compile enabled (max-autotune) nf4")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ torch.compile unavailable: {e}")

    def _load_gpu_fp16(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": self.cuda_device_index},  # single device, no sharding
            low_cpu_mem_usage=True,
        ).eval()
           # Try to JIT compile with PyTorch 2.x
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
            if self.verbose:
                print("⚡ torch.compile enabled (max-autotune) fp16")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ torch.compile unavailable: {e}")

    def _load_cpu(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        ).eval()

    # ---------------------- Audits ----------------------

    def _is_cuda(self) -> bool:
        return "cuda" in str(self._model_primary_device())

    def _model_primary_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # no params? fallback to first buffer
            for _, b in self.model.named_buffers():
                return b.device
            return torch.device("cpu")

    def _audit_and_fix_devices(self):
        """Try to move any CPU-straggler submodules (params/buffers) to CUDA."""
        cpu_items = []
        for n, p in self.model.named_parameters():
            if p.device.type != "cuda":
                cpu_items.append(("param", n, p.device))
        for n, b in self.model.named_buffers():
            # buffers like indices etc.
            if hasattr(b, "device") and b.device.type != "cuda":
                cpu_items.append(("buffer", n, b.device))

        if cpu_items and self.verbose:
            print("⚠️ Found items on CPU, moving parents to CUDA (best effort):")
            for kind, n, dev in cpu_items[:12]:
                print(f"   - {kind}: {n} on {dev}")

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
                    pass  # some quantized wrappers may not accept .to()

    def _assert_all_cuda(self):
        """Fail early if anything remains on CPU (prevents bucketize mismatch)."""
        leftover = []
        for n, p in self.model.named_parameters():
            if p.device.type != "cuda":
                leftover.append(("param", n, str(p.device)))
        for n, b in self.model.named_buffers():
            if hasattr(b, "device") and b.device.type != "cuda":
                leftover.append(("buffer", n, str(b.device)))
        if leftover:
            details = ", ".join([f"{k}:{n}@{d}" for k, n, d in leftover[:8]])
            raise RuntimeError(f"Model still has CPU items: {details} ...")

    # ---------------------- Video IO ----------------------

    def sample_frames_(self, video_path: str, num_frames: int, resize: int) -> List[np.ndarray]:


        time_frames = time.time()	
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Invalid/empty video: {video_path}")

        if num_frames == 1:
            indices = [total // 2]
        else:
            # Motion-based frame selection
            sample_rate = max(1, total // 200)  # Analyze max 200 frames for speed
            sample_indices = list(range(0, total, sample_rate))
            
            motion_scores = []
            prev_gray = None
            
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Small grayscale for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 120))
                
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_score = np.mean(diff)
                else:
                    motion_score = 0
                    
                motion_scores.append((frame_idx, motion_score))
                prev_gray = gray
            
            if motion_scores:
                # Sort by motion (highest first)
                motion_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Select high-motion frames with temporal diversity
                indices = []
                min_gap = total // (num_frames * 2)
                
                for frame_idx, _ in motion_scores:
                    if not indices or all(abs(frame_idx - s) >= min_gap for s in indices):
                        indices.append(frame_idx)
                        if len(indices) >= num_frames:
                            break
                
                # Fill remaining with uniform spacing if needed
                while len(indices) < num_frames:
                    uniform = np.linspace(0, total - 1, num_frames).astype(int)
                    for u in uniform:
                        if u not in indices:
                            indices.append(u)
                            break
                
                indices = sorted(indices[:num_frames])
            else:
                # Fallback to uniform if motion detection fails
                indices = np.linspace(0, total - 1, num_frames).astype(int)
                indices = sorted(list(dict.fromkeys(indices)))

        # Extract frames at selected indices
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = resize / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()

        print(f"Time taken: {time.time() - time_frames} seconds")
        if not frames:
            raise RuntimeError("No frames could be extracted.")
        return frames



    def sample_frames(self, video_path: str, num_frames: int, resize: int) -> List[np.ndarray]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Invalid/empty video: {video_path}")

        if num_frames == 1:
            indices = [total // 2]
        else:
            indices = np.linspace(0, total - 1, num_frames).astype(int)
            indices = sorted(list(dict.fromkeys(indices)))  # dedupe, keep order

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = resize / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        if not frames:
            raise RuntimeError("No frames could be extracted.")
        return frames

    # ---------------------- Core Inference ----------------------

    def analyze(
        self,
        video_path: str,
        prompt: str,
        num_frames: int = 4,
        resize: int = 336,
        max_tokens: int = 200,
        min_tokens: int = 100,	
    ) -> Dict[str, Any]:

        # 1) Sample frames
        frames = self.sample_frames(video_path, num_frames=num_frames, resize=resize)

        # 2) Build prompt text with image tokens
        text_only = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        prompt_text = self.processor.apply_chat_template(text_only, add_generation_prompt=True)
        image_tokens = "<image>" * len(frames)
        prompt_with_images = prompt_text.replace("<|im_start|>User:", f"<|im_start|>User: {image_tokens}")

        # 3) Processor -> tensors
        inputs = self.processor(
            text=prompt_with_images,
            images=frames,
            return_tensors="pt",
        )

        # 4) Move inputs to the **exact** model device, with proper dtypes
        inputs = self._move_inputs_to_model_device(inputs)

        # 5) Optional: last sanity check
        if self.verbose:
            devs = {k: str(v.device) for k, v in inputs.items() if hasattr(v, "device")}
            print("🧭 input devices:", devs)
            print("🧭 model device:", self._model_primary_device())

        # 6) Generate
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # 7) Decode
        text = self.processor.decode(out[0], skip_special_tokens=True)
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()

        return {
            "video_path": video_path,
            "prompt": prompt,
            "num_frames": len(frames),
            "resize": resize,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "analysis": text,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_id,
            "device": str(self._model_primary_device()),
            "quantized": self.quant_4bit,
        }

    def _move_inputs_to_model_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move processor outputs to the model's primary device.
        - pixel_values -> float16 on CUDA, float32 on CPU
        - integer tensors (ids/masks) keep dtype, just move device
        """
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


def parse_args():
    ap = argparse.ArgumentParser("Chronicon – SmolVLM GPU Inference")
    ap.add_argument("video_path", help="Path to the input video")
    ap.add_argument("--frames", "-f", type=int, default=8, help="Frames to sample")
    ap.add_argument("--resize", "-r", type=int, default=336, help="Resize long side")
    ap.add_argument("--max-tokens", "-t", type=int, default=100, help="Max new tokens to generate")
    ap.add_argument("--min-tokens", type=int, default=50, help="Min new tokens to generate")
    ap.add_argument("--prompt", "-p", type=str, default="Describe the main events, actions, and content in this video.")
    ap.add_argument("--model", "-m", type=str, default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    ap.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization (use fp16)")
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (default 0)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Video not found: {args.video_path}")
        sys.exit(1)

    print("🎥 Chronicon – SmolVLM GPU Inference")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"🎬 Frames: {args.frames} | 📝 Max tokens: {args.max_tokens}")
    print(f"🤖 Model: {args.model} | 4-bit: {not args.no_quant}")
    print(f"🖥️ CUDA device: {args.cuda_device} | 💾 Save: {not args.no_save}")

    t0 = time.time()
    try:
        engine = ChroniconVLM(
            model_id=args.model,
            verbose=args.verbose,
            quant_4bit=not args.no_quant,
            cuda_device=args.cuda_device,
        )
        engine.load()
        load_s = time.time() - t0

        t1 = time.time()
        result = engine.analyze(
            video_path=args.video_path,
            prompt=args.prompt,
            num_frames=args.frames,
            resize=args.resize,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens, 

        )
        infer_s = time.time() - t1
        total_s = time.time() - t0

        result["timing"] = {
            "model_load_time": round(load_s, 2),
            "inference_time": round(infer_s, 2),
            "total_time": round(total_s, 2),
        }

        print("\n✅ Done.")
        print(f"⏱️ Load: {load_s:.2f}s | Inference: {infer_s:.2f}s | Total: {total_s:.2f}s")
        print(f"🖥️ Device: {result['device']} | Quantized: {result['quantized']}")
        print(f"📝 Preview: {result['analysis'][:200]}{'...' if len(result['analysis']) > 200 else ''}")

        if not args.no_save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"inference_results_{ts}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved: {out}")

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
