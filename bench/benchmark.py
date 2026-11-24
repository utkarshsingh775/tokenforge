"""Load-test any OpenAI-compatible endpoint (TokenForge, vLLM, Token Factory...).

Measures the metrics inference engineers care about:
  TTFT  - time to first token (how long the user stares at a blank screen)
  TPOT  - time per output token after the first (decode speed)
  p50/p95 latency, and total output throughput (tokens/sec) across all users.

Usage:
  python -m bench.benchmark --url http://localhost:9000/v1 --model mock-small \
      --key sk-dev-123 --requests 40 --concurrency 8
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx


async def one_request(client, url, model, key, max_tokens):
    payload = {
        "model": model, "stream": True, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Explain KV caching in two sentences."}],
    }
    start = time.perf_counter()
    ttft, n = None, 0
    async with client.stream("POST", f"{url}/chat/completions", json=payload,
                             headers={"Authorization": f"Bearer {key}"}) as r:
        if r.status_code != 200:
            return {"error": r.status_code}
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            if json.loads(line[6:])["choices"][0]["delta"].get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                n += 1
    total = time.perf_counter() - start
    tpot = (total - ttft) / (n - 1) if n > 1 else 0
    return {"ttft": ttft, "total": total, "tokens": n, "tpot": tpot}


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


async def run(url, model, key, requests, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        async def guarded():
            async with sem:
                return await one_request(client, url, model, key, max_tokens)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded() for _ in range(requests)])
        wall = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    errors = len(results) - len(ok)
    if not ok:
        print(f"All {errors} requests failed: {results[0]}")
        return
    ttfts = [r["ttft"] * 1000 for r in ok]
    totals = [r["total"] * 1000 for r in ok]
    tpots = [r["tpot"] * 1000 for r in ok]
    tokens = sum(r["tokens"] for r in ok)

    print(f"\nModel: {model}  |  requests: {requests}  concurrency: {concurrency}  errors: {errors}")
    print(f"{'metric':<22}{'p50':>10}{'p95':>10}{'mean':>10}")
    for name, vals in [("TTFT (ms)", ttfts), ("TPOT (ms/token)", tpots), ("E2E latency (ms)", totals)]:
        print(f"{name:<22}{pct(vals, 50):>10.1f}{pct(vals, 95):>10.1f}{statistics.mean(vals):>10.1f}")
    print(f"Output throughput: {tokens / wall:.1f} tokens/sec  ({ok.__len__() / wall:.2f} req/sec)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9000/v1")
    ap.add_argument("--model", default="mock-small")
    ap.add_argument("--key", default="sk-dev-123")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.model, a.key, a.requests, a.concurrency, a.max_tokens))
