import pytest
from fastapi.testclient import TestClient

from app import reports
from app.main import app


@pytest.fixture(autouse=True)
def fast_reports(monkeypatch):
    """Keep the suite in milliseconds: the real delay is only for humans watching latency."""
    monkeypatch.setattr(reports, "REPORT_DELAY_S", 0.0)


@pytest.fixture
def client():
    return TestClient(app)
