import hashlib
import json
import time

from fastapi import FastAPI, HTTPException, Request, Response

from .reports import build_report

app = FastAPI(title="Reports service")

_MEMO: dict[int, tuple[float, dict]] = {}   # idea borrowed from route A: clients that never revalidate are served from memory
_MEMO_TTL_S = 60


def _report(customer_id: int) -> dict:
    now = time.monotonic()
    hit = _MEMO.get(customer_id)
    if hit and now - hit[0] < _MEMO_TTL_S:
        return hit[1]
    report = build_report(customer_id)      # raises KeyError for an unknown customer, which is never memoised
    _MEMO[customer_id] = (now, report)
    return report


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/report/{customer_id}")
def get_report(customer_id: int, request: Request, response: Response):
    try:
        report = _report(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown customer")
    etag = '"' + hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16] + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60"
    return report
