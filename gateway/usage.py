"""Usage metering: every request is logged to SQLite with tokens, latency and cost.
This is the 'billing' part of a token factory.
"""
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, api_key TEXT, model TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    ttft_ms REAL, total_ms REAL, cost_usd REAL
)
"""


class UsageStore:
    def __init__(self, path: str = "usage.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.lock = threading.Lock()

    def record(self, api_key, model, prompt_tokens, completion_tokens, ttft_ms, total_ms, cost_usd):
        with self.lock:
            self.conn.execute(
                "INSERT INTO usage (ts, api_key, model, prompt_tokens, completion_tokens,"
                " ttft_ms, total_ms, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), api_key, model, prompt_tokens, completion_tokens, ttft_ms, total_ms, cost_usd),
            )
            self.conn.commit()

    def summary(self, api_key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),"
            " AVG(ttft_ms), SUM(cost_usd) FROM usage WHERE api_key=? GROUP BY model",
            (api_key,),
        ).fetchall()
        return [
            {"model": m, "requests": n, "prompt_tokens": p, "completion_tokens": c,
             "avg_ttft_ms": round(t, 1), "cost_usd": round(cost, 6)}
            for m, n, p, c, t, cost in rows
        ]


def compute_cost(model_cfg: dict, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * model_cfg.get("price_per_1m_input", 0)
        + completion_tokens * model_cfg.get("price_per_1m_output", 0)
    ) / 1_000_000
