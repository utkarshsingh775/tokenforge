"""TokenForge: a mini OpenAI-compatible inference gateway.

Run:  uvicorn gateway.main:app --port 9000
"""
import json
import time
import uuid

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .backends import build_backend
from .config import load_config
from .ratelimit import RateLimiter
from .schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from .tokens import count_tokens
from .usage import UsageStore, compute_cost


def create_app(config: dict | None = None, db_path: str = "usage.db") -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="TokenForge")
    models = config["models"]
    backends = {name: build_backend(cfg) for name, cfg in models.items()}
    keys = config["api_keys"]
    limiter = RateLimiter()
    usage = UsageStore(db_path)

    key_names = {k: v["name"] for k, v in keys.items()}

    def auth(authorization: str = Header(default="")) -> str:
        key = authorization.removeprefix("Bearer ").strip()
        if key not in keys:
            usage.record_error(key, "-", 401)
            raise HTTPException(401, "invalid API key")
        if not limiter.allow(key, keys[key]["requests_per_minute"]):
            usage.record_error(key, "-", 429)
            raise HTTPException(429, "rate limit exceeded")
        return key

    def admin_auth(authorization: str = Header(default="")) -> str:
        # Separate from `auth` so dashboard polling never eats into rate limits.
        key = authorization.removeprefix("Bearer ").strip()
        if not keys.get(key, {}).get("admin"):
            raise HTTPException(403, "admin API key required")
        return key

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}

    @app.get("/v1/usage")
    def get_usage(key: str = Depends(auth)):
        return {"api_key": keys[key]["name"], "usage": usage.summary(key)}

    @app.get("/v1/stats")
    def get_stats(_: str = Depends(admin_auth)):
        return usage.stats(key_names)

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).parent / "static" / "dashboard.html")

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, key: str = Depends(auth)):
        if req.model not in backends:
            usage.record_error(key, req.model, 404)
            raise HTTPException(404, f"model '{req.model}' not found")
        backend = backends[req.model]
        prompt_tokens = sum(count_tokens(m.content) for m in req.messages)
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        start = time.perf_counter()

        def finish(ttft_ms: float, completion_tokens: int):
            total_ms = (time.perf_counter() - start) * 1000
            cost = compute_cost(models[req.model], prompt_tokens, completion_tokens)
            usage.record(key, req.model, prompt_tokens, completion_tokens, ttft_ms, total_ms, cost)

        if not req.stream:
            parts, ttft_ms = [], None
            async for tok in backend.stream(req):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                parts.append(tok)
            text = "".join(parts)
            completion_tokens = len(parts)
            finish(ttft_ms or 0, completion_tokens)
            return ChatResponse(
                id=req_id, created=created, model=req.model,
                choices=[Choice(message=Message(role="assistant", content=text))],
                usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens),
            )

        async def sse():
            ttft_ms, n = None, 0
            async for tok in backend.stream(req):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                n += 1
                chunk = {
                    "id": req_id, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            finish(ttft_ms or 0, n)
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


app = create_app()
