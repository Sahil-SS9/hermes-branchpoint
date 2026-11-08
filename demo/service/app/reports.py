"""Customer revenue reports. Building one is slow (it aggregates a lot of order history)."""
import time

REPORT_DELAY_S = 0.25  # stands in for an expensive aggregation query

ORDERS = {
    1: [120.0, 80.5, 42.0],
    2: [999.0],
    3: [10.0, 20.0, 30.0, 40.0],
}


def build_report(customer_id: int) -> dict:
    """Aggregate a customer's orders. Raises KeyError for an unknown customer."""
    time.sleep(REPORT_DELAY_S)
    orders = ORDERS[customer_id]
    return {
        "customer_id": customer_id,
        "orders": len(orders),
        "revenue": round(sum(orders), 2),
        "avg_order": round(sum(orders) / len(orders), 2),
    }
