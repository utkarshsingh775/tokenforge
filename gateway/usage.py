"""Usage metering: every request is logged to SQLite with tokens, latency, cost
and HTTP status. This is the 'billing' part of a token factory, and the data
source for the dashboard.
"""
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, api_key TEXT, model TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    ttft_ms REAL, total_ms REAL, cost_usd REAL,
    status INTEGER DEFAULT 200
)
"""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


class UsageStore:
    def __init__(self, path: str = "usage.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(usage)")]
        if "status" not in cols:  # migrate databases created before the dashboard
            self.conn.execute("ALTER TABLE usage ADD COLUMN status INTEGER DEFAULT 200")
        self.lock = threading.Lock()

    def record(self, api_key, model, prompt_tokens, completion_tokens, ttft_ms, total_ms,
               cost_usd, status=200):
        with self.lock:
            self.conn.execute(
                "INSERT INTO usage (ts, api_key, model, prompt_tokens, completion_tokens,"
                " ttft_ms, total_ms, cost_usd, status) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), api_key, model, prompt_tokens, completion_tokens,
                 ttft_ms, total_ms, cost_usd, status),
            )
            self.conn.commit()

    def record_error(self, api_key: str, model: str, status: int):
        self.record(api_key, model, 0, 0, 0, 0, 0, status)

    def summary(self, api_key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),"
            " AVG(ttft_ms), SUM(cost_usd) FROM usage WHERE api_key=? AND status=200"
            " GROUP BY model",
            (api_key,),
        ).fetchall()
        return [
            {"model": m, "requests": n, "prompt_tokens": p, "completion_tokens": c,
             "avg_ttft_ms": round(t, 1), "cost_usd": round(cost, 6)}
            for m, n, p, c, t, cost in rows
        ]

    def stats(self, key_names: dict[str, str], window_minutes: int = 30) -> dict:
        """Everything the dashboard needs, in one call."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT ts, api_key, model, prompt_tokens, completion_tokens, ttft_ms,"
                " total_ms, cost_usd, status FROM usage ORDER BY ts"
            ).fetchall()
        ok = [r for r in rows if r[8] == 200]
        name = lambda k: key_names.get(k, "invalid key")

        # Per-model latency, throughput and cost.
        models = {}
        for ts, key, model, p, c, ttft, total, cost, status in ok:
            m = models.setdefault(model, {"ttft": [], "total": [], "tps": [], "p": 0, "c": 0, "cost": 0.0})
            m["ttft"].append(ttft)
            m["total"].append(total)
            if total > ttft and c > 1:
                m["tps"].append((c - 1) / ((total - ttft) / 1000))  # decode speed
            m["p"] += p
            m["c"] += c
            m["cost"] += cost
        per_model = [
            {"model": name_, "requests": len(m["ttft"]),
             "prompt_tokens": m["p"], "completion_tokens": m["c"],
             "ttft_p50_ms": round(percentile(m["ttft"], 50), 1),
             "ttft_p95_ms": round(percentile(m["ttft"], 95), 1),
             "e2e_p50_ms": round(percentile(m["total"], 50), 1),
             "e2e_p95_ms": round(percentile(m["total"], 95), 1),
             "decode_tokens_per_sec": round(sum(m["tps"]) / len(m["tps"]), 1) if m["tps"] else 0,
             "cost_usd": round(m["cost"], 6)}
            for name_, m in sorted(models.items())
        ]

        # Per-API-key totals (who is using the platform).
        keys = {}
        for ts, key, model, p, c, ttft, total, cost, status in rows:
            k = keys.setdefault(name(key), {"requests": 0, "errors": 0, "tokens": 0, "cost": 0.0})
            k["requests"] += 1
            k["errors"] += status != 200
            k["tokens"] += p + c
            k["cost"] += cost
        per_key = [{"key": k, **v, "cost": round(v["cost"], 6)} for k, v in sorted(keys.items())]

        # Requests per minute per model, for the last `window_minutes`.
        now_min = int(time.time() // 60)
        minutes = list(range(now_min - window_minutes + 1, now_min + 1))
        series = {m["model"]: [0] * window_minutes for m in per_model}
        for ts, key, model, *_ in ok:
            i = int(ts // 60) - minutes[0]
            if 0 <= i < window_minutes and model in series:
                series[model][i] += 1

        errors = [r for r in rows if r[8] != 200]
        recent = [
            {"ts": ts, "key": name(key), "model": model, "prompt_tokens": p,
             "completion_tokens": c, "ttft_ms": round(ttft, 1), "total_ms": round(total, 1),
             "cost_usd": cost, "status": status}
            for ts, key, model, p, c, ttft, total, cost, status in reversed(rows[-25:])
        ]
        return {
            "totals": {
                "requests": len(rows),
                "successful": len(ok),
                "rate_limited": sum(r[8] == 429 for r in errors),
                "errors": len(errors),
                "tokens": sum(r[3] + r[4] for r in ok),
                "cost_usd": round(sum(r[7] for r in ok), 6),
                "ttft_p50_ms": round(percentile([r[5] for r in ok], 50), 1),
            },
            "per_model": per_model,
            "per_key": per_key,
            "timeseries": {"minutes": [m * 60 for m in minutes], "series": series},
            "recent": recent,
        }


def compute_cost(model_cfg: dict, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * model_cfg.get("price_per_1m_input", 0)
        + completion_tokens * model_cfg.get("price_per_1m_output", 0)
    ) / 1_000_000
