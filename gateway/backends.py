"""Model backends. Each backend yields text chunks (tokens) as an async stream.

- MockBackend: fake model that simulates time-to-first-token (TTFT) and decode speed.
  Lets you build and benchmark everything without a GPU.
- OpenAIBackend: forwards to any OpenAI-compatible server (vLLM, SGLang, Ollama,
  Jarvislabs Token Factory, ...).
"""
import asyncio
import json
import random
from typing import AsyncIterator

import httpx

from .schemas import ChatRequest

LOREM = (
    "inference engineering is the craft of making models fast cheap and reliable "
    "in production by batching requests caching prefixes quantizing weights and "
    "choosing the right hardware for the workload"
).split()


class MockBackend:
    def __init__(self, tokens_per_sec: float = 100, ttft_ms: float = 100, **_):
        self.tokens_per_sec = tokens_per_sec
        self.ttft_ms = ttft_ms

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        # Prefill phase: model reads the whole prompt -> time to first token.
        await asyncio.sleep(self.ttft_ms / 1000)
        n = min(req.max_tokens, random.randint(20, 60))
        for i in range(n):
            yield (" " if i else "") + LOREM[i % len(LOREM)]
            # Decode phase: one token at a time.
            await asyncio.sleep(1 / self.tokens_per_sec)


class OpenAIBackend:
    def __init__(self, base_url: str, upstream_model: str, api_key: str = "none", **_):
        self.base_url = base_url.rstrip("/")
        self.upstream_model = upstream_model
        self.api_key = api_key

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        payload = req.model_dump()
        payload["model"] = self.upstream_model
        payload["stream"] = True
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]


def build_backend(cfg: dict):
    kind = cfg.get("backend")
    if kind == "mock":
        return MockBackend(**cfg)
    if kind == "openai":
        return OpenAIBackend(**cfg)
    raise ValueError(f"unknown backend: {kind}")
