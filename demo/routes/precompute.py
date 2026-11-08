"""Refresh every customer's report on a schedule so requests never wait for the aggregation."""
import threading
import time

REFRESH_S = 30.0
_table: dict[int, dict] = {}
_started = False


def refresh(orders_by_customer: dict, build) -> None:
    for customer_id in orders_by_customer:
        _table[customer_id] = build(customer_id)


def lookup(customer_id: int):
    return _table.get(customer_id)


def start(orders_by_customer: dict, build) -> None:
    global _started
    if _started:
        return
    _started = True
    refresh(orders_by_customer, build)

    def loop():
        while True:
            time.sleep(REFRESH_S)
            refresh(orders_by_customer, build)

    threading.Thread(target=loop, daemon=True, name="report-refresher").start()
