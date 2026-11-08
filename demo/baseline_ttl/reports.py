"""Customer revenue reports. Building one is slow (it aggregates a lot of order history)."""
import time

REPORT_DELAY_S = 0.25  # stands in for an expensive aggregation query
CACHE_TTL_S = 60.0

ORDERS = {
    1: [120.0, 80.5, 42.0],
    2: [999.0],
    3: [10.0, 20.0, 30.0, 40.0],
}

_cache: dict[int, tuple[float, dict]] = {}


def build_report(customer_id: int) -> dict:
    """Aggregate a customer's orders (cached per process for CACHE_TTL_S). KeyError if unknown."""
    now = time.monotonic()
    hit = _cache.get(customer_id)
    if hit and now - hit[0] < CACHE_TTL_S:
        return hit[1]
    time.sleep(REPORT_DELAY_S)
    orders = ORDERS[customer_id]
    report = {
        "customer_id": customer_id,
        "orders": len(orders),
        "revenue": round(sum(orders), 2),
        "avg_order": round(sum(orders) / len(orders), 2),
    }
    _cache[customer_id] = (now, report)
    return report
