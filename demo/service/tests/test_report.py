def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_report_values(client):
    body = client.get("/report/1").json()
    assert body == {"customer_id": 1, "orders": 3, "revenue": 242.5, "avg_order": 80.83}


def test_unknown_customer_is_404(client):
    assert client.get("/report/999").status_code == 404


def test_report_is_stable_across_calls(client):
    assert client.get("/report/3").json() == client.get("/report/3").json()
