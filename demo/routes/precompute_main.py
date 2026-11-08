from fastapi import FastAPI, HTTPException

from .reports import build_report, start_refresher

app = FastAPI(title="Reports service")


@app.on_event("startup")
def _warm() -> None:
    start_refresher()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/report/{customer_id}")
def get_report(customer_id: int) -> dict:
    try:
        return build_report(customer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown customer")
