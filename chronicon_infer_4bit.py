#!/usr/bin/env python3
"""
Chronicon – SmolVLM Standalone Inference (CUDA + 4-bit NF4)
- Single-GPU placement (no auto sharding)
- bitsandbytes 4-bit quantization (NF4 + double quant, fp16 compute)
- CUDA/CPU mismatch guard (bucketize shim)
- CUDA default device & perf flags
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

# ==================== Speed / stability flags ====================
torch.backends.cuda.matmul.allow_tf32 = True  # ok for inference
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --------- HARDENING: device-safe bucketize shim (prevents cuda/cpu mix) ----------
if not hasattr(torch, "_orig_bucketize"):
    torch._orig_bucketize = torch.bucketize
    def _bucketize_device_safe(input, boundaries, *args, **kwargs):
        if torch.is_tensor(boundaries) and input.is_cuda and not boundaries.is_cuda:
            boundaries = boundaries.to(input.device)
        return torch._orig_bucketize(input, boundaries, *args, **kwargs)
    torch.bucketize = _bucketize_device_safe
# ----------------------------------------------------------------------------------


class ChroniconVLM4bit:
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        verbose: bool = False,
        cuda_device: int = 0,
        force_cpu: bool = False,
    ):
        self.model_id = model_id
        self.verbose = verbose
        self.cuda_device_index = cuda_device
        self.force_cpu = force_cpu

        self.model = None
        self.processor = None

        if not self.force_cpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Use --cpu to run on CPU (much slower).")

        self.device = torch.device("cpu" if self.force_cpu else f"cuda:{self.cuda_device_index}")
        self.fp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Encourage factory tensors to land on our target device
        try:
            torch.set_default_device(self.device)
        except Exception:
            pass

    # ---------------------- Model load ----------------------

    def load(self):
        if self.verbose:
            print(f"📦 Loading model {self.model_id}")
            print(f"   - Torch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'n/a'}")
            if self.device.type == "cuda":
                print(f"   - GPU[{self.cuda_device_index}]: {torch.cuda.get_device_name(self.cuda_device_index)}")
                prop = torch.cuda.get_device_properties(self.cuda_device_index)
                print(f"   - VRAM: {prop.total_memory/1024**3:.1f} GB")
            print(f"   - Target device: {self.device}")
            print(f"   - Quantization: 4-bit NF4 (fp16 compute, double-quant)")

        if self.device.type == "cuda":
            self._load_gpu_4bit_nf4()
        else:
            self._load_cpu_float32()

        # Processor
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # ensure a pad token for generate()
        if getattr(self.processor, "tokenizer", None):
            tok = self.processor.tokenizer
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

        # generation pad id too (some models read from config)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            if getattr(self.model.generation_config, "pad_token_id", None) is None:
                self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        if self.verbose:
            print(f"✅ Load complete on: {self._model_primary_device()}")

        # Final audit (CUDA only)
        if self.device.type == "cuda":
            self._audit_and_fix_devices()
            self._assert_all_cuda()

    def _load_gpu_4bit_nf4(self):
        # Force single-GPU placement
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

    def _load_cpu_float32(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        ).eval()

    # ---------------------- Audits ----------------------

    def _model_primary_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            for _, b in self.model.named_buffers():
                return b.device
            return torch.device("cpu")

    def _audit_and_fix_devices(self):
        """Find any params/buffers left on CPU and push their parent modules to CUDA."""
        cpu_items = []
        for n, p in self.model.named_parameters():
            if p.device.type != "cuda":
                cpu_items.append(("param", n, p.device))
        for n, b in self.model.named_buffers():
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
                    pass

    def _assert_all_cuda(self):
        leftover = []
        for n, p in self.model.named_parameters():
            if p.device.type != "cuda":
                leftover.append(("param", n, str(p.device)))
        for n, b in self.model.named_buffers():
            if hasattr(b, "device") and b.device.type != "cuda":
                leftover.append(("buffer", n, str(b.device)))
        if leftover:
            details = ", ".join([f"{k}:{n}@{d}" for k, n, d in leftover[:12]])
            raise RuntimeError(f"Model still has CPU items: {details} ...")

    # ---------------------- Video IO ----------------------

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
            # dedupe, keep order
            indices = sorted(list(dict.fromkeys(indices)))

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
        num_frames: int = 8,
        resize: int = 336,
        max_tokens: int = 100,
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

        # 4) Move inputs to the model device with correct dtypes
        inputs = self._move_inputs_to_model_device(inputs)

        if self.verbose:
            devs = {k: str(v.device) for k, v in inputs.items() if hasattr(v, "device")}
            print("🧭 input devices:", devs)
            print("🧭 model device:", self._model_primary_device())

        # 5) Generate
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # 6) Decode
        text = self.processor.decode(out[0], skip_special_tokens=True)
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()

        return {
            "video_path": video_path,
            "prompt": prompt,
            "num_frames": len(frames),
            "resize": resize,
            "max_tokens": max_tokens,
            "analysis": text,
            "timestamp": datetime.now().isoformat(),
            "model": self.model_id,
            "device": str(self._model_primary_device()),
            "quantized": True,
        }

    def _move_inputs_to_model_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move processor outputs to the model's primary device.
        - pixel_values -> float16 on CUDA, float32 on CPU
        - integer tensors (ids/masks) keep dtype, just move
        """
        model_dev = self._model_primary_device()
        moved = {}

        is_cuda = model_dev.type == "cuda"
        pixel_dtype = torch.float16 if is_cuda else torch.float32

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
    ap = argparse.ArgumentParser("Chronicon – SmolVLM 4-bit Inference")
    ap.add_argument("video_path", help="Path to the input video")
    ap.add_argument("--frames", "-f", type=int, default=8, help="Frames to sample (1–16)")
    ap.add_argument("--resize", "-r", type=int, default=336, help="Resize long side")
    ap.add_argument("--max-tokens", "-t", type=int, default=100, help="Max new tokens to generate")
    ap.add_argument("--prompt", "-p", type=str, default="Describe the main events, actions, and content in this video.")
    ap.add_argument("--model", "-m", type=str, default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (default 0)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (debug only; very slow)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Video not found: {args.video_path}")
        sys.exit(1)

    print("🎥 Chronicon – SmolVLM 4-bit Inference")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"🎬 Frames: {args.frames} | 📝 Max tokens: {args.max_tokens}")
    print(f"🤖 Model: {args.model} | Quantized: 4-bit NF4")
    print(f"🖥️ Device: {'CPU' if args.cpu else f'cuda:{args.cuda_device}'} | 💾 Save: {not args.no_save}")

    t0 = time.time()
    try:
        engine = ChroniconVLM4bit(
            model_id=args.model,
            verbose=args.verbose,
            cuda_device=args.cuda_device,
            force_cpu=args.cpu,
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
            out = f"inference_results_4bit_{ts}.json"
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
