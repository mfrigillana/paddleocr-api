"""PaddleOCR (PP-OCRv4 via ONNX Runtime) API sized for Render's free tier (512 MB RAM)."""
import os
import threading

import cv2
import numpy as np
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from rapidocr_onnxruntime import RapidOCR

API_KEY = os.getenv("API_KEY", "")
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))        # downscale big photos to save RAM
MAX_BYTES = int(os.getenv("MAX_BYTES", str(8 * 1024 * 1024)))

# One thread each: free tier has 0.1 CPU; more threads only add memory.
engine = RapidOCR(
    intra_op_num_threads=1,
    inter_op_num_threads=1,
    det_limit_side_len=int(os.getenv("DET_LIMIT", "736")),
    det_limit_type="max",   # cap detector input size -> caps peak RAM
    rec_batch_num=1,
)
lock = threading.Lock()  # one OCR job at a time keeps peak RAM predictable
app = FastAPI(title="PaddleOCR API")


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image (jpg/png/webp/bmp)")

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def run():
        with lock:
            return engine(img)[0]

    result = await run_in_threadpool(run)
    lines = [
        {"text": t, "confidence": round(float(c), 4), "box": [[int(x), int(y)] for x, y in b]}
        for b, t, c in (result or [])
    ]
    return {"text": "\n".join(l["text"] for l in lines), "lines": lines}
