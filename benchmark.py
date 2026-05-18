"""Benchmark script for Triton Inference Server HTTP API."""
import json, sys, time, struct, argparse, concurrent.futures, urllib.request
import numpy as np

# Match the ONNX model input shape
SHAPE = [1, 256, 256]
DTYPE = "FP32"
TOTAL_ELEMENTS = 1 * 256 * 256  # 65536


def build_request(model_name):
    """Build a Triton inference request payload."""
    data = np.zeros(SHAPE, dtype=np.float32).flatten().tolist()
    return json.dumps({
        "inputs": [{
            "name": "input",
            "shape": SHAPE,
            "datatype": DTYPE,
            "data": data,
        }]
    })


def send_one(url, model_name):
    """Send one inference request, return (ok, ttft_ms)."""
    t0 = time.time()
    try:
        body = build_request(model_name).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return True, (time.time() - t0) * 1000
    except Exception as e:
        return False, (time.time() - t0) * 1000


def warmup(url, model_name, n=5):
    """Send warmup requests."""
    print(f"Warming up ({n} requests)...")
    for _ in range(n):
        send_one(url, model_name)


def run(url, model_name, concurrency, duration_sec):
    """Run benchmark and return stats."""
    deadline = time.time() + duration_sec
    results = []
    req_count = 0

    def worker():
        nonlocal req_count
        while time.time() < deadline:
            ok, ttft = send_one(url, model_name)
            results.append({"ok": ok, "ttft_ms": ttft})
            req_count += 1

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker) for _ in range(concurrency)]
        concurrent.futures.wait(futures)
    elapsed = time.time() - t0

    ok_results = [r for r in results if r["ok"]]
    err_count = len(results) - len(ok_results)
    ttft_list = sorted([r["ttft_ms"] for r in ok_results])
    n = len(ttft_list)
    throughput = len(ok_results) / elapsed if elapsed > 0 else 0

    return {
        "concurrency": concurrency,
        "duration_s": round(elapsed, 1),
        "total_requests": len(results),
        "ok": len(ok_results),
        "errors": err_count,
        "throughput_rps": round(throughput, 1),
        "ttft_avg_ms": round(sum(ttft_list) / n, 1) if n else 0,
        "ttft_p50_ms": round(ttft_list[n // 2], 1) if n else 0,
        "ttft_p99_ms": round(ttft_list[int(n * 0.99)], 1) if n else 0,
        "ttft_min_ms": round(ttft_list[0], 1) if n else 0,
        "ttft_max_ms": round(ttft_list[-1], 1) if n else 0,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark Triton Inference Server")
    p.add_argument("--url", default="http://localhost:8000/v2/models/identity_onnx/infer")
    p.add_argument("--model", default="identity_onnx")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--duration", type=int, default=10,
                   help="Duration per concurrency level (seconds)")
    p.add_argument("--sweep", type=str, default=None,
                   help="Comma-separated concurrency levels to sweep (e.g. '1,2,4,8')")
    args = p.parse_args()

    url = args.url
    model_name = args.model

    warmup(url, model_name)

    if args.sweep:
        levels = [int(x) for x in args.sweep.split(",")]
    else:
        levels = [args.concurrency]

    print(f"\n{'='*65}")
    print(f"Target: {url}")
    print(f"Sweep:  {levels}  |  Duration: {args.duration}s per level")
    print(f"{'='*65}")

    all_results = []
    for c in levels:
        print(f"\n--- Concurrency = {c} ---")
        stats = run(url, model_name, c, args.duration)
        all_results.append(stats)
        print(f"  Throughput: {stats['throughput_rps']} req/s")
        print(f"  Latency:    avg={stats['ttft_avg_ms']}ms  "
              f"P50={stats['ttft_p50_ms']}ms  P99={stats['ttft_p99_ms']}ms")
        print(f"  Requests:   {stats['ok']}/{stats['total_requests']} "
              f"(errors: {stats['errors']})")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"{'Concurrency':<14}{'Throughput':<14}{'P50':<10}{'P99':<10}{'Errors'}")
        print("-" * 65)
        for s in all_results:
            print(f"{s['concurrency']:<14}{s['throughput_rps']:<14.1f}"
                  f"{s['ttft_p50_ms']:<10}{s['ttft_p99_ms']:<10}{s['errors']}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
