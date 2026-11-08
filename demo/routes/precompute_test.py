from app import precompute, reports


def test_requests_are_served_from_the_precomputed_table(monkeypatch):
    precompute._table.clear()
    precompute.refresh(reports.ORDERS, reports._aggregate)
    monkeypatch.setattr(reports, "_aggregate", lambda cid: (_ for _ in ()).throw(AssertionError("computed on the request path")))
    assert reports.build_report(2)["revenue"] == 999.0
