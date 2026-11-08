def test_report_has_etag_and_cache_control(client):
    r = client.get("/report/1")
    assert r.status_code == 200 and r.headers["etag"] and r.headers["cache-control"] == "private, max-age=60"


def test_conditional_get_returns_304(client):
    etag = client.get("/report/1").headers["etag"]
    r = client.get("/report/1", headers={"If-None-Match": etag})
    assert r.status_code == 304 and r.content == b""
