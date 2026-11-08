import hashlib
import json

from fastapi import FastAPI, HTTPException, Request, Response

from .reports import build_report

app = FastAPI(title="Reports service")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/report/{customer_id}")
def get_report(customer_id: int, request: Request, response: Response):
    try:
        report = build_report(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown customer")
    etag = '"' + hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16] + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60"
    return report
