from app import cache


def test_cache_round_trip():
    cache.put(1, {"customer_id": 1})
    assert cache.get(1) == {"customer_id": 1}
