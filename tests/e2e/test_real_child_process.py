"""Real-process proof: a real `hermes chat` build agent, with this plugin loaded, driven by a scripted model.

Needs the `hermes` binary (any recent Hermes); no credentials and no spend: the "model" is a local
deterministic endpoint (tests/fake_model.py). Run with:  pytest tests/e2e -m e2e
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from bp_helpers import IN_MEMORY, git, make_llm, mod
from fake_model import FakeModel

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e
if not shutil.which("hermes"):
    pytest.skip("hermes binary not on PATH", allow_module_level=True)


@pytest.fixture(scope="module")
def hermes_home():
    # A fresh, disposable HERMES_HOME (the host prepares an isolated runtime for it: ~30 s the first time).
    # It must live OUTSIDE any real Hermes home: inside one, the host treats it as the same tenant and refuses
    # to start a second gateway ("nothing to start"), so /tmp, not the per-user scratch dir.
    home = Path(tempfile.mkdtemp(prefix="branchpoint-e2e-", dir="/tmp" if os.path.isdir("/tmp") else None))
    plugin = home / "plugins" / "branchpoint"
    shutil.copytree(ROOT, plugin, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "tests", "demo", "docs", ".ruff_cache", "*.pyc"))
    return home


def write_config(home: Path, url: str):
    (home / "config.yaml").write_text(
        f"model:\n  provider: custom\n  default: fake-model\n  base_url: {url}\n  api_key: fake\n"
        "plugins:\n  enabled:\n    - branchpoint\napprovals:\n  mode: manual\n")


def test_child_agent_is_confined_checked_and_measured(repo, data, hermes_home, monkeypatch):
    R, AG = mod("run"), mod("agent")
    fixed = "from service import get_report\n\n\ndef test_new():\n    assert get_report(2) is get_report(2)\n"
    steps = [
        ("read_file", {"path": "service.py"}),
        ("write_file", {"path": str(repo / "service.py"), "content": "# escape attempt on the real tree\n"}),
        ("write_file", {"path": "tests/test_new.py", "content": "def test_new():\n    assert False, 'red first'\n"}),
        ("write_file", {"path": "service.py", "content": IN_MEMORY}),
        ("read_file", {"path": "tests/test_service.py"}),          # 5th tool call: checks start in the background
        ("read_file", {"path": "service.py"}, 4.0),                # the checks started above finish while the "model" thinks
        ("read_file", {"path": "tests/test_service.py"}),
        ("write_file", {"path": "tests/test_new.py", "content": fixed}),
        ("terminal", {"command": "ls"}),                           # not in the allowlist
    ]
    with FakeModel(steps) as fm:
        write_config(hermes_home, fm.url)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        for k in ("HERMES_YOLO_MODE", "HERMES_SESSION_KEY", "HERMES_GATEWAY_SESSION"):
            monkeypatch.delenv(k, raising=False)
        A = mod("anchor")
        anchor = A.resolve_fork(cwd=repo, goal="add caching to the slow /report endpoint")
        run = R.Run.create(data, anchor=anchor, mode="fork", goal=anchor.goal, llm=make_llm(),
                           settings={"n": 3, "execute": "top1", "budget_tokens": 5_000_000, "route_seconds": 240, "route_max_turns": 20},
                           agent=AG.HermesAgentRunner())
        run.prepare()
        run.go(background=False)
        assert run.meta["status"] == "decided", run.meta.get("warnings")
        rid = run.meta["executed"][0]
        rdir = run.dir / "routes" / rid
        led = next(l for l in run.ledgers() if l.route == rid)

    # --- the real tree is untouched, and the escape attempt was refused and logged
    assert git(repo, "status", "--porcelain") == ""
    assert "escape attempt" not in (repo / "service.py").read_text()
    events = [json.loads(l) for l in (rdir / "tool_events.jsonl").read_text().splitlines()]
    blocked = [e for e in events if e.get("blocked")]
    assert any("outside this route's worktree" in b["blocked"] for b in blocked), blocked
    # a tool outside the file toolset never ran (the host refuses it, or our allowlist blocks it)
    assert not any(e.get("tool") == "terminal" and not e.get("blocked") and e.get("status") == "ok" for e in events)
    # --- every tool call has a duration; real usage was captured from the provider payload
    tool_events = [e for e in events if e.get("tool") and not e.get("blocked")]
    assert tool_events and all("duration_ms" in e for e in tool_events)
    assert led.cost.estimated is False and led.cost.tokens >= 150, led.cost
    # --- checks ran mid-build (red first), and the agent was shown the result
    clog = [json.loads(l) for l in (rdir / "checks_log.jsonl").read_text().splitlines()]
    assert clog and clog[0]["calls"] == 5 and clog[0]["results"][0]["passed"] is False
    seen = json.dumps(fm.requests)
    assert "Branchpoint checks after 5 tool calls" in seen and "red first" in seen or "test_new" in seen
    assert "HARD CONSTRAINT" in seen                                   # pre_llm_call injected the strategy
    # --- the ledger recorded the transient failure as a bug hit along the way, and the final state is green
    assert any("mid-build" in i.detail for i in led.issues)
    assert not [i for i in led.issues if i.kind == "test" and i.severity == "high"]
    assert led.diff_stats["files"] == 2 and led.status == "done" and led.is_complete, led.completeness()
    assert sys.version_info >= (3, 11)
    assert os.path.exists(rdir / "agent.log")
