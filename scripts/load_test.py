import argparse
import asyncio
import time

import httpx

from model import MODELS_DIR

SAMPLE_IMAGE_PATH = MODELS_DIR / "sample_digit.png"


async def send_one(client, index, results):
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    start = time.perf_counter()
    response = await client.post(
        "/predict",
        files={"file": (f"digit_{index}.png", image_bytes, "image/png")},
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    results.append((index, start, elapsed))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    results = []
    async with httpx.AsyncClient(base_url=args.url, timeout=60.0) as client:
        wall_start = time.perf_counter()
        await asyncio.gather(*[send_one(client, i, results) for i in range(args.concurrency)])
        wall_elapsed = time.perf_counter() - wall_start

    per_request = [r[2] * 1000 for r in results]
    starts = [r[1] - wall_start for r in results]

    print(f"Fired {args.concurrency} concurrent requests")
    print(f"Total wall-clock time : {wall_elapsed * 1000:.1f} ms")
    print(f"Mean per-request time : {sum(per_request) / len(per_request):.1f} ms")
    print(f"Min / Max request time: {min(per_request):.1f} / {max(per_request):.1f} ms")
    print(f"Spread of request start times: {max(starts) * 1000:.1f} ms "
          f"(near 0 = all fired at once, as intended)")
    print(f"\nIf requests ran truly concurrently, total wall-clock time would be close to")
    print(f"the mean per-request time ({sum(per_request) / len(per_request):.1f} ms).")
    print(f"If they were fully serialized, total time would be close to the sum of all")
    print(f"individual times (~{sum(per_request):.1f} ms).")


if __name__ == "__main__":
    asyncio.run(main())
