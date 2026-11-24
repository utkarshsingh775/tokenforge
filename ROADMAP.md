# TokenForge roadmap — build and learn on the go

Phase 1 is done (the starter code). Each later phase has things to **learn**,
things to **build**, and a **done when** check. Keep a `NOTES.md` with your
benchmark numbers; they turn into your resume bullets and blog post.

---

## Phase 1 — Understand the starter (2–3 days) ✅ code given

**Learn**
- What an OpenAI-compatible API looks like (`/v1/chat/completions`, SSE streaming)
- **Prefill vs decode**: why the first token is slow (TTFT) and the rest stream
  at a steady speed (TPOT). `MockBackend` simulates exactly this.
- Token-bucket rate limiting

**Do**
- Read every file. Change `ttft_ms` / `tokens_per_sec` in `models.json`, rerun
  the benchmark, and check the numbers move the way you expect.
- Run the benchmark at concurrency 1, 4, 16, 64. Write down what happens.

**Done when:** you can explain TTFT, TPOT, p95 and throughput without notes.

---

## Phase 2 — Make it production-ish (1 week)

**Build**
1. **Real tokenizer**: replace `gateway/tokens.py` with `tiktoken` or a HF tokenizer.
2. **Request logging + `/metrics`**: expose Prometheus metrics (request count,
   latency histogram, tokens/sec) with `prometheus-client`.
3. **Timeouts + retries**: if a backend fails, retry once, then return a clean 502.
4. **Docker**: write a `Dockerfile` and `docker-compose.yml`.
5. **More tests**: streaming usage gets recorded, the backend error path.

**Done when:** `docker compose up` gives you a working gateway with `/metrics`.

---

## Phase 3 — Smart gateway features (1–2 weeks)

**Build**
1. **Model router**: a `"model": "auto"` option that sends short or easy prompts
   to `mock-small` and long or hard ones to `mock-large`. Start with rules (prompt
   length, keywords), then try a tiny classifier.
2. **Fallback**: if the primary model errors or times out, try the next one.
3. **Response cache**: an exact-match cache first, then a *semantic* cache
   (embed the prompt and reuse the answer if cosine similarity > 0.95). Measure hit rate.
4. **Redis rate limiter**: move `ratelimit.py` to Redis so 2+ replicas share limits.

**Learn:** cost/quality trade-offs, caching, distributed state.

**Done when:** you can show "auto routing saved X% cost" from `/v1/usage`.

---

## Phase 4 — Real GPU inference (1–2 weeks, ~$10–20 of GPU time)

**Learn**
- vLLM internals: **PagedAttention, continuous batching, KV cache, prefix caching**
- Quantization: FP16 vs AWQ/GPTQ INT4 vs FP8, and the quality/speed/memory trade-off
- How GPU memory limits batch size and context length

**Build**
1. Rent a GPU (Jarvislabs L4 or A100, billed per second). Stop it when you're done.
2. Start vLLM:
   ```bash
   pip install vllm
   vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
   ```
3. Point the `qwen-7b` entry in `models.json` at it (SSH tunnel:
   `ssh -L 8000:localhost:8000 <gpu-host>`) and run the benchmark through TokenForge.
4. **The experiment matrix**. Benchmark each combination and fill in a table:

   | Setup | TTFT p50 | TPOT | tokens/sec @ conc 32 | GPU $/hr | $ per 1M tokens |
   |---|---|---|---|---|---|
   | HF transformers (no batching) | | | | | |
   | vLLM FP16 | | | | | |
   | vLLM AWQ INT4 | | | | | |
   | vLLM + prefix caching (long shared system prompt) | | | | | |
   | SGLang FP16 | | | | | |
   | Same on L4 vs A100 vs H100 | | | | | |

   **$ per 1M tokens = GPU $/hr ÷ (tokens/sec × 3600) × 1,000,000**

5. Quality check: run ~100 questions on FP16 and INT4 and compare the answers
   (exact match or LLM-as-judge), so you can show quantization didn't break quality.

**Done when:** the table is full and you can explain *why* each number changed.

---

## Phase 5 — Ship and show it (1 week)

1. **Dashboard** ✅ done: `/dashboard`. Exercise: add a GPU-cost panel once Phase 4 runs on real hardware,
   and a model filter for the charts.
2. **Scale-to-zero** (stretch goal): deploy with Jarvislabs Serverless or Modal and measure cold-start time.
3. **README polish**: architecture diagram, benchmark table, GIF demo.
4. **Blog post**: "I built a mini inference platform: here's what made LLMs 5× cheaper."
   Post it on LinkedIn, X and dev.to, and send it to companies you're targeting.

---

## Resume bullets (fill in YOUR real numbers)

**TokenForge — LLM Inference Gateway** | Python, FastAPI, vLLM, Redis, Docker, Prometheus
- Built an OpenAI-compatible inference gateway with API-key auth, token-bucket rate
  limiting, SSE streaming, and per-request token/cost metering, serving X models.
- Benchmarked Qwen2.5-7B across HF Transformers, vLLM and SGLang on L4/A100/H100;
  continuous batching raised throughput Xx and AWQ INT4 cut cost to $X per 1M tokens
  with <X% quality loss.
- Added difficulty-based model routing, provider fallback and semantic caching,
  reducing cost by X% at a X% cache hit rate.

## Interview questions this project prepares you for
- Why is TTFT different from TPOT? What makes each one slow?
- What is continuous batching, and why does it beat static batching?
- How does PagedAttention save GPU memory?
- When would you quantize a model, and how do you check you didn't hurt quality?
- How would you rate-limit across 10 gateway replicas?
- A customer says "your API is slow." How do you debug it? (FDE favorite)
