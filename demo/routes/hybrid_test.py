from app import main


def test_second_request_is_served_from_memory(client, monkeypatch):
    calls = []
    real = main.build_report
    monkeypatch.setattr(main, "build_report", lambda customer_id: calls.append(customer_id) or real(customer_id))
    main._MEMO.clear()
    first = client.get("/report/1")
    second = client.get("/report/1")
    assert calls == [1] and first.json() == second.json()


def test_unknown_customer_is_not_memoised(client):
    main._MEMO.clear()
    assert client.get("/report/999").status_code == 404 and 999 not in main._MEMO
