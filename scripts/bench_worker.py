import argparse
import json
import time

import numpy as np
import psutil
from PIL import Image

from model import IMAGE_SIZE, MODELS_DIR

ONNX_FP32_PATH = MODELS_DIR / "model_fp32.onnx"
ONNX_INT8_PATH = MODELS_DIR / "model_int8.onnx"
STUDENT_PATH = MODELS_DIR / "student_distilled.pt"
SAMPLE_IMAGE_PATH = MODELS_DIR / "sample_digit.png"

NUM_REQUESTS = 200
PROCESS = psutil.Process()


def rss_mb():
    return PROCESS.memory_info().rss / 1e6


def load_sample_batch(n):
    image = Image.open(SAMPLE_IMAGE_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.array(image, dtype=np.float32) / 255.0
    return np.stack([pixels] * n)[:, None, :, :]  # (n, 1, 28, 28)


def time_calls(fn, batch, warmup=10):
    for i in range(warmup):
        fn(batch[i % len(batch)])

    latencies = []
    for i in range(len(batch)):
        start = time.perf_counter()
        fn(batch[i])
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    return np.array(latencies)


def build_torch_infer():
    import torch

    from model import build_model

    model = build_model()

    def infer(sample):
        with torch.no_grad():
            return model(torch.from_numpy(sample).unsqueeze(0)).numpy()

    return infer, (MODELS_DIR / "resnet18_mnist_baseline.pt")


def build_onnx_infer(path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def infer(sample):
        return session.run(["logits"], {"input": sample[None, ...]})[0]

    return infer, path


def build_student_infer():
    import torch

    from student_model import StudentCNN

    student = StudentCNN()
    student.load_state_dict(torch.load(STUDENT_PATH, map_location="cpu", weights_only=True))
    student.eval()

    def infer(sample):
        with torch.no_grad():
            return student(torch.from_numpy(sample).unsqueeze(0)).numpy()

    return infer, STUDENT_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["torch", "onnx_fp32", "onnx_int8", "student"])
    args = parser.parse_args()

    mem_before = rss_mb()
    if args.variant == "torch":
        infer, weights_path = build_torch_infer()
    elif args.variant == "onnx_fp32":
        infer, weights_path = build_onnx_infer(ONNX_FP32_PATH)
    elif args.variant == "onnx_int8":
        infer, weights_path = build_onnx_infer(ONNX_INT8_PATH)
    else:
        infer, weights_path = build_student_infer()
    mem_after_load = rss_mb()

    batch = load_sample_batch(NUM_REQUESTS)
    latencies_ms = time_calls(infer, batch)
    logits = infer(batch[0])

    result = {
        "variant": args.variant,
        "mean_latency_ms": float(latencies_ms.mean()),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "mem_delta_mb": mem_after_load - mem_before,
        "disk_bytes": weights_path.stat().st_size,
        "predicted_digit": int(np.argmax(logits)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
