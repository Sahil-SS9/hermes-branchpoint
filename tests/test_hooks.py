"""Hooks: route confinement/budget/logging/feedback in a child, and the anchor snapshot in the parent."""
from __future__ import annotations

import json
import sys
import time

import pytest
from bp_helpers import git, mod


@pytest.fixture
def route(repo, monkeypatch, tmp_path):
    """A real linked worktree with a route marker, and the cwd set inside it (as the child process would be)."""
    R, P = mod("route"), mod("protocol")
    wt = tmp_path / "wt" / "B"
    R.worktree_add(repo, wt, git(repo, "rev-parse", "HEAD"), branch="branchpoint/t/B")
    events = tmp_path / "events.jsonl"
    clog = tmp_path / "checks.jsonl"
    marker = {"route": "B", "strategy": {"axis": "infrastructure", "thesis": "use redis", "key_decisions": ["a"]}, "confine_root": str(wt),
              "events_path": str(events), "checks_log": str(clog), "check_every": 5, "max_tool_calls": 12, "token_cap": 1000,
              "checks": [{"kind": "test", "argv": [sys.executable, "-c", "print('FAILED tests/x.py::t - boom'); raise SystemExit(1)"], "timeout": 30}],
              "base_failures": []}
    P.write_marker(wt, marker)
    monkeypatch.chdir(wt)
    return wt, events, clog


def test_marker_is_private_to_the_worktree(route):
    wt, *_ = route
    assert not any("branchpoint-route" in l for l in git(wt, "status", "--porcelain", "--ignored").splitlines())
    assert mod("protocol").read_marker(wt / "sub" if (wt / "sub").exists() else wt)["route"] == "B"


def test_confinement_blocks_escapes(route, repo):
    H = mod("hooks")
    wt, events, _ = route
    h = H.Hooks()
    ok = h.pre_tool_call(tool_name="write_file", args={"path": "x.py", "content": "1"})
    assert ok is None
    for args in ({"path": str(repo / "service.py")}, {"path": "../../elsewhere/x"}, {"path": str(repo.parent / "other-dir" / "y.txt")}):
        res = h.pre_tool_call(tool_name="write_file", args=args)
        assert res and res["action"] == "block" and "outside this route's worktree" in res["message"]
    (wt / "esc").symlink_to(repo)
    assert h.pre_tool_call(tool_name="patch", args={"path": "esc/service.py"})["action"] == "block"          # symlink escape
    v4a = f"*** Begin Patch\n*** Update File: ok.py\n*** Update File: {repo.parent}/evil.py\n*** End Patch"
    assert h.pre_tool_call(tool_name="patch", args={"patch": v4a, "mode": "patch"})["action"] == "block"      # multi-file V4A
    assert h.pre_tool_call(tool_name="write_file", args={"path": ".env", "content": "K=1"})["action"] == "block"  # secrets
    assert h.pre_tool_call(tool_name="terminal", args={"command": "ls"})["action"] == "block"                  # file tools only
    assert h.pre_tool_call(tool_name="execute_code", args={})["action"] == "block"
    logged = [json.loads(l) for l in events.read_text().splitlines()]
    assert len([e for e in logged if e.get("blocked")]) >= 6


def test_read_and_search_cannot_send_secret_files_to_the_model(route):
    wt, *_ = route
    (wt / ".env").write_text("EXAMPLE_PLACEHOLDER=not-a-real-secret\n")
    (wt / "innocent.txt").symlink_to(wt / ".env")
    h = mod("hooks").Hooks()
    for path in (".env", "innocent.txt"):
        blocked = h.pre_tool_call(tool_name="read_file", args={"path": path})
        assert blocked and blocked["action"] == "block" and "secret" in blocked["message"]
    blocked = h.pre_tool_call(tool_name="search_files", args={"path": ".", "pattern": "EXAMPLE_PLACEHOLDER", "target": "content"})
    assert blocked and blocked["action"] == "block" and "secret" in blocked["message"]
    assert h.pre_tool_call(tool_name="read_file", args={"path": "service.py"}) is None


def test_budget_blocks_after_cap(route):
    H = mod("hooks")
    h = H.Hooks()
    for i in range(12):
        assert h.pre_tool_call(tool_name="read_file", args={"path": "service.py"}) is None
        h.post_tool_call(tool_name="read_file", args={"path": "service.py"}, result="{}", duration_ms=3, status="ok")
    res = h.pre_tool_call(tool_name="read_file", args={"path": "service.py"})
    assert res["action"] == "block" and "budget" in res["message"]
    ctx = h.pre_llm_call(session_id="s", user_message="go")
    assert "STOP" in ctx["context"] and "HARD CONSTRAINT" in ctx["context"]


def test_token_cap_via_usage(route):
    H = mod("hooks")
    h = H.Hooks()
    h.post_api_request(usage={"input_tokens": 900, "output_tokens": 200})
    assert "token budget" in h.pre_tool_call(tool_name="read_file", args={"path": "a"})["message"]


def test_events_checks_and_feedback(route):
    H = mod("hooks")
    wt, events, clog = route
    h = H.Hooks()
    for i in range(5):
        h.post_tool_call(tool_name="write_file", args={"path": f"f{i}.py"}, result="{}", duration_ms=5 + i, status="ok")
    deadline = time.time() + 20
    while time.time() < deadline and not (clog.exists() and clog.read_text()):
        time.sleep(0.1)
    evs = [json.loads(l) for l in events.read_text().splitlines()]
    assert [e["tool"] for e in evs] == ["write_file"] * 5 and evs[2]["duration_ms"] == 7
    entry = json.loads(clog.read_text().splitlines()[0])
    assert entry["calls"] == 5 and entry["results"][0]["passed"] is False
    out = h.transform_tool_result(tool_name="read_file", result="CONTENT")
    assert out.startswith("CONTENT") and "Branchpoint checks after 5 tool calls" in out and "boom" in out
    assert h.transform_tool_result(tool_name="read_file", result="CONTENT") is None       # delivered once


def test_error_events_recorded(route):
    H = mod("hooks")
    wt, events, _ = route
    h = H.Hooks()
    h.post_tool_call(tool_name="patch", args={"path": "a.py"}, result='{"error": "no match"}', duration_ms=1, status="ok")
    e = json.loads(events.read_text().splitlines()[0])
    assert e["status"] == "error" and "no match" in e["error"]


def test_hooks_are_inert_outside_a_route(repo, monkeypatch):
    H = mod("hooks")
    monkeypatch.chdir(repo)
    h = H.Hooks()
    assert h.pre_tool_call(tool_name="terminal", args={"command": "ls"}) is None
    assert h.transform_tool_result(tool_name="x", result="r") is None
    assert h.pre_llm_call(session_id="", user_message="") is None
    h.post_tool_call(tool_name="x", args={}, result="")


def test_parent_hook_records_anchor(repo, data, monkeypatch):
    H, P, A = mod("hooks"), mod("paths"), mod("anchor")
    P.set_data_dir_override(data)
    try:
        class Ctx:
            def session_cwd(self):
                return str(repo)
        monkeypatch.chdir(repo)
        h = H.Hooks(Ctx(), lambda k, d: d)
        h.pre_llm_call(session_id="sess1", user_message="/branchpoint status", conversation_history=[])     # slash: ignored
        assert A.load_turns(data, repo, "sess1") == []
        h.pre_llm_call(session_id="sess1", user_message="add caching please", conversation_history=[{"role": "user", "content": "earlier"}])
        for _ in range(50):
            turns = A.load_turns(data, repo, "sess1")
            if turns:
                break
            time.sleep(0.1)
        assert turns[0]["n"] == 2 and turns[0]["prompt"] == "add caching please"
        off = H.Hooks(Ctx(), lambda k, d: False if k == "anchor_on_turn" else d)
        off.pre_llm_call(session_id="sess2", user_message="x", conversation_history=[])
        assert A.load_turns(data, repo, "sess2") == []
    finally:
        P.set_data_dir_override(None)
