import io
import time
from pathlib import Path

import anyio.to_thread
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "model_fp32.onnx"
IMAGE_SIZE = 28

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

app = FastAPI(title="MNIST Digit Recognizer API (ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    pixels = np.array(image, dtype=np.float32)

    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    pixels = pixels / 255.0
    return pixels[None, None, :, :]  # (1, 1, 28, 28)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def run_inference(tensor: np.ndarray) -> np.ndarray:
    # Runs on a worker thread (see anyio.to_thread.run_sync below), not the
    # event loop thread, so it no longer blocks other requests while it runs.
    infer_start = time.time()
    logits = session.run(["logits"], {"input": tensor})[0][0]
    infer_end = time.time()
    print(f"[predict] infer_start={infer_start:.4f} infer_end={infer_end:.4f} duration={infer_end - infer_start:.4f}s", flush=True)
    return logits


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/sleep")
async def debug_sleep():
    # Temporary diagnostic: a pure Python CPU-bound blocking call with no
    # I/O, no numpy, no C-extension. Isolates whether "sync code inside an
    # async def" truly serializes requests, independent of onnxruntime.
    time.sleep(0.5)
    return {"status": "done"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        tensor = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    logits = await anyio.to_thread.run_sync(run_inference, tensor)
    probabilities = softmax(logits)
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    return PredictionResponse(
        digit=digit,
        confidence=confidence,
        probabilities=[float(p) for p in probabilities.tolist()],
    )
