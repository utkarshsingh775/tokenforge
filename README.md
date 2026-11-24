# TokenForge — a mini inference gateway

An OpenAI-compatible API that sits in front of LLMs, like a tiny version of
Jarvislabs Token Factory, OpenRouter or Together AI. It handles API keys, rate
limits, streaming, usage metering and cost tracking, and comes with a benchmark tool.

Works on a laptop with **no GPU** (mock models). Later you point it at a real
vLLM server on a rented GPU.

```
client ──► TokenForge gateway ──► mock model     (laptop, free)
           auth · rate limit  └─► vLLM / SGLang   (GPU, Phase 4)
           metering · cost    └─► any OpenAI-compatible API
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                   # 5 tests should pass
uvicorn gateway.main:app --port 9000        # start the gateway
```

In a second terminal:

```bash
curl localhost:9000/v1/chat/completions -H "Authorization: Bearer sk-dev-123" \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-small","messages":[{"role":"user","content":"hi"}]}'

python -m bench.benchmark --model mock-small --requests 40 --concurrency 8
curl localhost:9000/v1/usage -H "Authorization: Bearer sk-dev-123"
```

It also works with the official `openai` Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="sk-dev-123")
print(client.chat.completions.create(model="mock-small",
      messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
```

## Layout

| File | What it does |
|---|---|
| `gateway/main.py` | FastAPI app: `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/v1/usage` |
| `gateway/backends.py` | `MockBackend` (simulates TTFT + decode speed) and `OpenAIBackend` (proxies vLLM, etc.) |
| `gateway/ratelimit.py` | Token-bucket rate limiter per API key |
| `gateway/usage.py` | SQLite usage log + cost calculation |
| `models.json` | Model registry, prices, API keys |
| `bench/benchmark.py` | Load tester: TTFT, TPOT, p50/p95 latency, throughput |

See **ROADMAP.md** for what to build next and what each phase teaches.
