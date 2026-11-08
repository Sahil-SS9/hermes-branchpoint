import json
import os

import redis  # new runtime dependency

_client = redis.Redis.from_url(os.environ["REDIS_URL"])
TTL_S = 60


def get(customer_id: int):
    raw = _client.get(f"report:{customer_id}")
    return json.loads(raw) if raw else None


def put(customer_id: int, report: dict) -> None:
    _client.setex(f"report:{customer_id}", TTL_S, json.dumps(report))
