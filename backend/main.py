import asyncio
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import anyio.to_thread
import numpy as np
import onnxruntime as ort
import redis.asyncio as redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "model_fp32.onnx"
IMAGE_SIZE = 28
BATCH_TIMEOUT = 0.002  # 2ms window, starting from the first request in a new batch
BATCH_MAX_SIZE = 4
CACHE_TTL_SECONDS = 60

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

app = FastAPI(title="MNIST Digit Recognizer API (ONNX, dynamic batching)")

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


@dataclass
class BatchItem:
    tensor: np.ndarray
    future: "asyncio.Future[np.ndarray]"


request_queue: "asyncio.Queue[BatchItem]" = asyncio.Queue()


def run_batch_inference(batch_tensor: np.ndarray) -> np.ndarray:
    # Runs on a worker thread (see anyio.to_thread.run_sync below), not the
    # event loop thread, so a slow batch still doesn't block new requests
    # from queuing up for the *next* batch while this one runs.
    infer_start = time.time()
    logits_batch = session.run(["logits"], {"input": batch_tensor})[0]
    infer_end = time.time()
    print(
        f"[batch] size={batch_tensor.shape[0]} infer_start={infer_start:.4f} "
        f"infer_end={infer_end:.4f} duration={infer_end - infer_start:.4f}s",
        flush=True,
    )
    return logits_batch


async def batch_worker():
    loop = asyncio.get_event_loop()
    while True:
        first_item = await request_queue.get()
        batch = [first_item]
        deadline = loop.time() + BATCH_TIMEOUT

        while len(batch) < BATCH_MAX_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(request_queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break

        batch_tensor = np.concatenate([item.tensor for item in batch], axis=0)
        logits_batch = await anyio.to_thread.run_sync(run_batch_inference, batch_tensor)

        for i, item in enumerate(batch):
            item.future.set_result(logits_batch[i])


@app.on_event("startup")
async def launch_batch_worker():
    asyncio.create_task(batch_worker())


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

    request_start = time.time()
    image_bytes = await file.read()

    cache_key = "predict:" + hashlib.sha256(image_bytes).hexdigest()
    cached = await redis_client.get(cache_key)
    if cached is not None:
        print(f"[cache] HIT  key={cache_key[:20]}... lookup={time.time() - request_start:.4f}s", flush=True)
        return PredictionResponse(**json.loads(cached))
    print(f"[cache] MISS key={cache_key[:20]}...", flush=True)

    try:
        tensor = preprocess(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    future: "asyncio.Future[np.ndarray]" = asyncio.get_event_loop().create_future()
    await request_queue.put(BatchItem(tensor=tensor, future=future))
    logits = await future

    probabilities = softmax(logits)
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])

    response = PredictionResponse(
        digit=digit,
        confidence=confidence,
        probabilities=[float(p) for p in probabilities.tolist()],
    )
    await redis_client.set(cache_key, response.model_dump_json(), ex=CACHE_TTL_SECONDS)
    print(f"[cache] computed and stored in {time.time() - request_start:.4f}s (ttl={CACHE_TTL_SECONDS}s)", flush=True)
    return response
