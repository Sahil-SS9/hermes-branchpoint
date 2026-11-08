"""Customer revenue reports. The slow aggregation runs on a schedule (see precompute.py)."""
import time

from . import precompute

REPORT_DELAY_S = 0.25  # stands in for an expensive aggregation query

ORDERS = {
    1: [120.0, 80.5, 42.0],
    2: [999.0],
    3: [10.0, 20.0, 30.0, 40.0],
}


def _aggregate(customer_id: int) -> dict:
    time.sleep(REPORT_DELAY_S)
    orders = ORDERS[customer_id]
    return {
        "customer_id": customer_id,
        "orders": len(orders),
        "revenue": round(sum(orders), 2),
        "avg_order": round(sum(orders) / len(orders), 2),
    }


def build_report(customer_id: int) -> dict:
    """Served from the precomputed table; computed on demand only before the first refresh. KeyError if unknown."""
    hit = precompute.lookup(customer_id)
    if hit is not None:
        return hit
    return _aggregate(customer_id)


def start_refresher() -> None:
    precompute.start(ORDERS, _aggregate)
