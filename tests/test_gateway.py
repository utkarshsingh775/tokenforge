from fastapi.testclient import TestClient

from gateway.config import load_config
from gateway.main import create_app


def client(tmp_path):
    return TestClient(create_app(load_config(), db_path=str(tmp_path / "u.db")))


AUTH = {"Authorization": "Bearer sk-dev-123"}
BODY = {"model": "mock-small", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}


def test_rejects_bad_key(tmp_path):
    r = client(tmp_path).post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_chat_completion_and_usage(tmp_path):
    c = client(tmp_path)
    r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["completion_tokens"] <= 10
    u = c.get("/v1/usage", headers=AUTH).json()["usage"]
    assert u[0]["model"] == "mock-small" and u[0]["requests"] == 1


def test_streaming(tmp_path):
    with client(tmp_path).stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=AUTH) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and len(lines) > 2


def test_unknown_model(tmp_path):
    r = client(tmp_path).post("/v1/chat/completions", json={**BODY, "model": "gpt-9"}, headers=AUTH)
    assert r.status_code == 404


def test_rate_limit(tmp_path):
    c = client(tmp_path)
    free = {"Authorization": "Bearer sk-free-456"}  # 5 requests/minute
    codes = [c.post("/v1/chat/completions", json=BODY, headers=free).status_code for _ in range(7)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


def test_stats_requires_admin_key(tmp_path):
    c = client(tmp_path)
    assert c.get("/v1/stats", headers={"Authorization": "Bearer sk-free-456"}).status_code == 403
    assert c.get("/v1/stats", headers=AUTH).status_code == 200


def test_stats_counts_success_and_rejections(tmp_path):
    c = client(tmp_path)
    c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    c.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer nope"})
    c.post("/v1/chat/completions", json={**BODY, "model": "gpt-9"}, headers=AUTH)
    s = c.get("/v1/stats", headers=AUTH).json()
    assert s["totals"]["requests"] == 3
    assert s["totals"]["successful"] == 1
    assert s["totals"]["errors"] == 2
    assert s["per_model"][0]["model"] == "mock-small"
    assert sum(s["timeseries"]["series"]["mock-small"]) == 1
    assert {r["status"] for r in s["recent"]} == {200, 401, 404}


def test_dashboard_page_served(tmp_path):
    r = client(tmp_path).get("/dashboard")
    assert r.status_code == 200 and "TokenForge" in r.text
