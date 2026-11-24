"""Token-bucket rate limiter, one bucket per API key.

Each key gets `capacity` requests; the bucket refills at capacity/60 per second.
LEARN: this lives in process memory, so it breaks with 2+ gateway replicas.
       Phase 3 exercise: move it to Redis.
"""
import time


class TokenBucket:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_per_sec = capacity / 60
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_sec)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimiter:
    def __init__(self):
        self.buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str, rpm: int) -> bool:
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(rpm)
        return self.buckets[key].allow()
