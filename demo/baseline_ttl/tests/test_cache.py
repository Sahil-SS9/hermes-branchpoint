from app import reports


def test_second_call_is_served_from_cache(monkeypatch):
    reports._cache.clear()
    calls = []
    real_sleep = reports.time.sleep
    monkeypatch.setattr(reports.time, "sleep", lambda s: calls.append(s) or real_sleep(0))
    reports.build_report(1)
    reports.build_report(1)
    assert len(calls) == 1
