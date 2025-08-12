# app.py — FastAPI wrapper around your ChroniconVLM class
from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
import uvicorn, shutil, os
from chronicon_vlm import ChroniconVLM  # your working script as a class

app = FastAPI()
engine = ChroniconVLM(
    model_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    verbose=False,
    quant_4bit=True,        # << keep quantization on
    cuda_device=0
)
engine.load()

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    prompt: str = Form("Detect notable events and summarize."),
    frames: int = Form(8),
    resize: int = Form(336),
    max_tokens: int = Form(120),
):
    # save temp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = f"/tmp/{ts}_{video.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        result = engine.analyze(
            video_path=temp_path,
            prompt=prompt,
            num_frames=frames,
            resize=resize,
            max_tokens=max_tokens,
        )
        # OPTIONAL: persist result locally
        out_dir = "inference_results"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"inference_results_{ts}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            import json; json.dump(result, f, indent=2, ensure_ascii=False)

        return {"ok": True, "result": result, "saved": out_file}
    finally:
        try: os.remove(temp_path)
        except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
