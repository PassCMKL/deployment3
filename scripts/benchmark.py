import json
import subprocess
import sys

VARIANTS = [
    ("torch", "Naive PyTorch (FP32)"),
    ("onnx_fp32", "ONNX Runtime (FP32)"),
    ("onnx_int8", "ONNX Runtime (INT8, dynamic)"),
    ("student", "Distilled student (PyTorch)"),
]


def run_variant(name):
    proc = subprocess.run(
        [sys.executable, "bench_worker.py", "--variant", name],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def format_disk(disk_bytes):
    mb = disk_bytes / 1e6
    return f"{disk_bytes / 1e3:.1f} KB" if mb < 1 else f"{mb:.1f} MB"


def main():
    results = {}
    for variant, label in VARIANTS:
        print(f"Running {label} in an isolated subprocess...")
        result = run_variant(variant)
        results[variant] = result

        print(f"\n{label}")
        print(f"  mean latency       : {result['mean_latency_ms']:.3f} ms")
        print(f"  p95 latency        : {result['p95_latency_ms']:.3f} ms")
        print(f"  throughput         : {1000 / result['mean_latency_ms']:.1f} req/s (single-threaded)")
        print(f"  memory footprint   : {result['mem_delta_mb']:.1f} MB (RSS growth after loading model, isolated process)")
        print(f"  model size on disk : {format_disk(result['disk_bytes'])}")
        print(f"  predicted digit    : {result['predicted_digit']}")
        print()

    print("Agreement check (all four should predict the same digit on the sample image):")
    for variant, label in VARIANTS:
        print(f"  {label:<30}: {results[variant]['predicted_digit']}")

    baseline = results["torch"]["mean_latency_ms"]
    print("\nSpeedup summary (relative to naive PyTorch mean latency):")
    for variant, label in VARIANTS[1:]:
        print(f"  {label:<30}: {baseline / results[variant]['mean_latency_ms']:.2f}x")

    print("\nModel size summary (relative to naive PyTorch disk size):")
    baseline_bytes = results["torch"]["disk_bytes"]
    for variant, label in VARIANTS[1:]:
        ratio = baseline_bytes / results[variant]["disk_bytes"]
        print(f"  {label:<30}: {ratio:.1f}x smaller")


if __name__ == "__main__":
    main()
