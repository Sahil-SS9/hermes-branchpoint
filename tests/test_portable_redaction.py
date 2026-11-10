"""Best-effort prompt scrubbing must not depend on private host imports."""
from __future__ import annotations

import sys
from bp_helpers import mod


def test_scrubber_works_without_a_host_redactor(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.redact", None)
    L = mod("llm")
    text = 'api_key="EXAMPLE_ONLY" password=PLACEHOLDER https://someone:example@localhost/path'
    result = L.redact_for_model(text)
    assert "EXAMPLE_ONLY" not in result and "PLACEHOLDER" not in result and "someone:example" not in result
    assert "localhost/path" in result and L.redaction_available()


def test_scrubber_preserves_normal_code():
    L = mod("llm")
    text = 'def get_report(customer_id):\n    return {"tokens": 150, "value": "hello"}\n'
    assert L.redact_for_model(text) == text
