# SYS-304 Milestone 3 — Scaling & Optimization

Optimized deployment of the MNIST digit recognizer (ResNet18 baseline from Milestone 1/2), covering two independent tracks: **model-level optimization** (Week 5) and **infrastructure-level optimization** (Week 6), all benchmarked against the naive Milestone 2 baseline.

## Architecture

```
Client (POST /predict)
        │
        ▼
  Cache check (SHA-256 of image bytes → Redis)
    ├── HIT  → return stored result immediately
    └── MISS → preprocess → join queue (own Future)
                    │
                    ▼
              Batch worker
        (4 requests collected OR 2ms elapsed,
              whichever comes first)
                    │
                    ▼
        Run ONNX model on batch,
     offloaded to a thread pool
        (anyio.to_thread.run_sync)
                    │
                    ▼
      Fill each request's Future
         with its own result
                    │
                    ▼
      Store in Redis (TTL 60s) → return response
```

Full diagram: see `Deployment_3_Architecture.pdf` / the report.

## Repository structure

```
deployment3/
├── backend/
│   ├── main.py              # FastAPI app: caching, batching queue, thread-pool inference
│   ├── model_fp32.onnx      # ONNX-exported model used by the backend
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   ├── resnet18_mnist_baseline.pt   # Milestone 1/2 baseline weights
│   ├── model_fp32.onnx              # ONNX export
│   ├── model_int8.onnx              # Dynamically quantized (INT8)
│   └── student_distilled.pt         # Distilled student model
├── scripts/
│   ├── model.py              # Shared model-loading (mirrors Deployment_2's build_model)
│   ├── export_onnx.py        # Week 5: ONNX export
│   ├── quantize.py           # Week 5: dynamic INT8 quantization
│   ├── student_model.py      # Week 5: distilled student architecture
│   ├── distill.py            # Week 5: distillation training loop
│   ├── bench_worker.py       # Runs one model variant in an isolated subprocess
│   ├── benchmark.py          # Orchestrates all model-level benchmarks
│   └── load_test.py          # Week 6: concurrent-load testing tool
├── docker-compose.yml         # backend + redis services
└── README.md
```

## Week 5 — Model-level optimization

Three independent techniques, all starting from the same trained baseline:

| Variant | Latency (mean) | Speedup | RSS Memory | Disk size | Accuracy |
|---|---|---|---|---|---|
| Naive PyTorch (baseline) | 104.5 ms | 1.0x | 94.6 MB | 44.8 MB | 97.52% |
| ONNX FP32 | 19.3 ms | 5.4x | 79.0 MB | 44.7 MB | 97.52% |
| ONNX INT8 (dynamic quant.) | 49.1 ms | 2.1x | 36.7 MB | 11.2 MB | not measured |
| Distilled student | 1.6 ms | 67.3x | 1.3 MB | 39.4 KB | 97.06% |

- **ONNX export**: converts the trained model into ONNX's graph format for a faster runtime, with zero change to weights or accuracy.
- **INT8 quantization (dynamic)**: weights converted to INT8 offline; activations quantized at runtime — no calibration dataset required. Slower than FP32 here due to per-call conversion overhead on an already-fast model.
- **Distillation**: a new, independent ~9k-parameter CNN trained to mimic the frozen teacher's softened output distribution (temperature = 4.0, hard/soft loss weighted 0.3/0.7).

Run: `python scripts/export_onnx.py`, `python scripts/quantize.py`, `python scripts/distill.py`, then `python scripts/benchmark.py`.

## Week 6 — Infrastructure-level optimization

Three techniques layered on the ONNX FP32 backend, measured under 20 concurrent requests to `/predict`:

| Configuration | Total wall time | Mean per-request | Throughput |
|---|---|---|---|
| Baseline (blocking, single-threaded) | 689.0 ms | 620.8 ms | 29.0 req/s |
| Thread-pool offload only | 1042.6 ms | 954.2 ms | 19.2 req/s |
| Dynamic batching (max 4 / 2ms window) | 577.2 ms | 449.1 ms | 34.6 req/s |

- **Thread-pool offload**: moves the blocking `session.run()` call off the main event loop. Alone, this made things *worse* — 20 threads, each paying fixed dispatch overhead on a task that only takes ~1-2ms.
- **Dynamic batching**: groups requests before touching threads at all (max 4 requests, or 2ms elapsed, whichever first). Reduced 20 individual calls to 6 batched calls, fixing the overhead problem and beating both prior versions.
- **Caching**: exact-match (SHA-256 hash) via Redis, 60s TTL. Real-world benefit is limited for this app, since users draw unique digits rather than repeating identical inputs — implemented and benchmarked separately via direct cache-hit/miss timing rather than under the concurrent-load test above.

Run: `python scripts/load_test.py --concurrency 20`

## Running the service

```bash
docker-compose up --build
```

This starts two containers: `redis` (cache store) and `backend` (FastAPI + ONNX Runtime, port 8000). `REDIS_HOST` is injected automatically via Docker Compose networking.

Health check: `GET /health`
Predict: `POST /predict` (multipart image upload)

## Full report

See the combined benchmark + architecture PDF for the complete write-up with charts.
