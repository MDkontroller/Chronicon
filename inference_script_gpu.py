#!/usr/bin/env python3
"""
GPU-Forced Inference Script for Chronicon
- Leaves your existing code untouched.
- Forces CUDA + 4-bit quantization (bitsandbytes) if available.
- Audits device placement to avoid CPU/CUDA mismatches.
"""

import argparse
import sys
import time
from datetime import datetime
import os
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from analizer_test import VideoAnalyzer  # reuse your class for sampling/IO/etc.

# ---- A thin subclass that only overrides model loading & device moves ----
class VideoAnalyzerGPU(VideoAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force CUDA if available; otherwise bail early with a clean message
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Run this GPU script on a machine with a CUDA-capable GPU.")
        self.device = "cuda"

    def load_model(self):
        if self.verbose:
            print(f"📦(GPU) Loading model {self.model_id} with 4-bit quantization on CUDA...")

        # 4-bit config (works well on consumer GPUs)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",       # nf4 is usually most stable
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True   # helps memory on larger models
        )

        # Force everything onto GPU 0. Avoid "auto" to prevent CPU shard spill.
        device_map = {"": 0}

        # Load quantized model directly on CUDA
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device_map
        ).eval()

        # Processor (tokenizer + image/video processor)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)


                # pad token safety (prevents generate() warnings/crashes)
        if getattr(self.processor, "tokenizer", None):
            tok = self.processor.tokenizer
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token_id = tok.eos_token_id

        if self.verbose:
            print("✅ Model + processor loaded on CUDA")
            self._audit_model_devices()



    # Ensure all tensors go to the model device before generate()
    def _ensure_inputs_on_device(self, proc_inputs):
        device = torch.device(self.device)
        moved = {}
        for k, v in proc_inputs.items():
            if hasattr(v, "to"):  # Tensor-like
                # Use FP16 for pixel_values on GPU for speed and memory savings
                if k == "pixel_values" and device.type == "cuda":
                    moved[k] = v.to(device, dtype=torch.float16, non_blocking=True)
                else:
                    moved[k] = v.to(device, non_blocking=True)
            else:
                moved[k] = v  # Non-tensor objects
        return moved


    def _audit_model_devices(self):
        # Spot check a few modules to confirm they’re really on CUDA
        on_cuda = True
        for name, param in list(self.model.named_parameters())[:5]:
            if param.device.type != "cuda":
                on_cuda = False
                break
        if on_cuda:
            print("🔍 Device audit: model parameters are on CUDA")
        else:
            print("⚠️ Device audit: some parameters are NOT on CUDA")

    def _generate(self, proc_inputs, max_tokens: int):
        device = torch.device("cuda")
        self.model = self.model.to(device)

        # recursive move
        proc_inputs = self._move_to_device(proc_inputs, device, force_fp16_for_pixels=True)

        # quick audit
        if self.verbose:
            try:
                print("🔍 gen model device:", next(self.model.parameters()).device)
                print("🔍 gen inputs:", {k: str(v.device) for k, v in proc_inputs.items() if hasattr(v, "device")})
            except Exception:
                pass

        with torch.no_grad():
            return self.model.generate(
                **proc_inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )



def main():
    parser = argparse.ArgumentParser(description="Chronicon GPU Inference (forced CUDA + 4-bit)")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--frames", "-f", type=int, default=8, help="Frames to sample (1-16)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Custom prompt")
    parser.add_argument("--resize", "-r", type=int, default=336, help="Resize long side")
    parser.add_argument("--max-tokens", "-t", type=int, default=100, help="Max new tokens")
    parser.add_argument("--model", "-m", type=str,
                        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                        choices=["HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                                 "HuggingFaceTB/SmolVLM2-2.2B-Instruct"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Video not found: {args.video_path}")
        sys.exit(1)

    if args.frames < 1:
        print("❌ --frames must be >= 1")
        sys.exit(1)
    if args.frames > 16:
        print("⚠️ --frames capped to 16")
        args.frames = 16

    print("🎥 Chronicon (GPU) Inference")
    print("=" * 60)
    print(f"📹 Video: {args.video_path}")
    print(f"🎬 Frames: {args.frames}  | 🔧 Resize: {args.resize} | 📝 Max tokens: {args.max_tokens}")
    print(f"🤖 Model: {args.model}   | ⚙️  Quant: 4-bit (nf4) on CUDA")
    print(f"🔍 Verbose: {args.verbose} | 💾 Save: {not args.no_save}")

    t0 = time.time()
    analyzer = VideoAnalyzerGPU(model_id=args.model, verbose=args.verbose)
    load_s = time.time() - t0

    prompt = args.prompt or "Describe the main events, actions, and content in this video."
    t1 = time.time()
    result = analyzer.analyze_video(
        video_path=args.video_path,
        prompt=prompt,
        num_frames=args.frames,
        resize=args.resize,
        max_tokens=args.max_tokens
    )
    infer_s = time.time() - t1
    total_s = time.time() - t0

    # attach timing
    result.update({
        "timing": {
            "model_load_time": round(load_s, 2),
            "inference_time": round(infer_s, 2),
            "total_time": round(total_s, 2)
        }
    })

    print("\n✅ Done.")
    print(f"⏱️ Load: {load_s:.2f}s | Inference: {infer_s:.2f}s | Total: {total_s:.2f}s")
    print(f"📝 Preview: {result['analysis'][:180]}...")

    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"inference_results_gpu_{ts}.json"
        analyzer.save_analysis(result, out)
        print(f"💾 Saved: {out}")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
