"""Release-truth regressions: a refused model override must be reported, not silently swapped; the token
budget's "hard stop" must actually bind the build agent's share; the standalone CLI must fail loudly.

These pin the claims made in README "Cost" / "Honest limits" and in the plugin catalog.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from bp_helpers import git, make_llm, mod

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- model override truth


class PluginTrustError(PermissionError):
    """Stands in for the host's PluginLlmTrustError: the plugin was not granted a model override."""


class Reply:
    def __init__(self, text: str = "ok", model: str = "your-default") -> None:
        self.text = text
        self.model = model
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5, cost_usd=None)


class RefusingBackend:
    """A ctx.llm clone that refuses model overrides for one model and answers on the default otherwise."""

    def __init__(self, refuse: str = "judge-x") -> None:
        self.refuse = refuse
        self.calls: list[dict] = []

    def complete(self, *, messages, **kw):
        self.calls.append(dict(kw))
        if kw.get("model"):
            raise PluginTrustError(f"plugin 'branchpoint' is not allowed to use model '{kw['model']}'")
        return Reply()


def test_a_refused_override_is_reported_on_the_run_wide_llm_not_only_the_meter_clone():
    """The run reads warnings from its own LLM object; per-route clones must not swallow them."""
    L = mod("llm")
    llm = L.CtxLLM(SimpleNamespace(llm=RefusingBackend()))
    route_llm = llm.with_meter(L.Meter(10_000))
    assert route_llm.complete("s", "u", model="judge-x", purpose="judge").text == "ok"
    assert any("judge-x" in w for w in llm.warnings), llm.warnings       # RED: clone kept it to itself
    assert any("judge-x" in w for w in route_llm.warnings)


def test_a_refused_model_is_not_re_requested_on_every_council_call():
    """Three reviewers on one refused model must cost one refusal, not three, and warn once."""
    L = mod("llm")
    backend = RefusingBackend()
    llm = L.CtxLLM(SimpleNamespace(llm=backend))
    for _ in range(3):
        llm.complete("s", "u", model="judge-x", purpose="judge")
    assert len([c for c in backend.calls if c.get("model")]) == 1, backend.calls
    assert len(llm.warnings) == 1, llm.warnings
    assert "allow_model_override" in llm.warnings[0]


def test_a_permission_error_that_is_not_a_model_refusal_is_not_silently_downgraded():
    """Only a trust refusal may drop the model; anything else must surface unchanged."""
    L = mod("llm")
    calls: list[dict] = []

    class Backend:
        def complete(self, *, messages, **kw):
            calls.append(dict(kw))
            raise PermissionError("/var/lib/branchpoint is not writable")

    llm = L.CtxLLM(SimpleNamespace(llm=Backend()))
    with pytest.raises(PermissionError, match="not writable"):
        llm.complete("s", "u", model="m1", purpose="judge")
    assert len(calls) == 1, calls       # RED: retried once without the model, hiding the real error
    assert not llm.warnings


def build_run(repo, data, *, llm=None, extra_settings=None):
    A, R, AG = mod("anchor"), mod("run"), mod("agent")
    anchor = A.resolve_fork(cwd=repo, goal="add caching to the slow /report endpoint")
    settings = {"n": 3, "execute": "none", "budget_tokens": 5_000_000, "max_parallel": 2,
                "approach": "assigned", **(extra_settings or {})}
    return R.Run.create(data, anchor=anchor, mode="fork", goal=anchor.goal, settings=settings,
                        llm=llm or make_llm(), agent=AG.CallableRunner(lambda job: None)), anchor


class ScriptedBackend:
    """A host ``ctx.llm`` that answers from the shared test script but refuses one model override."""

    def __init__(self, script, refuse: str = "judge-x") -> None:
        self.script, self.refuse, self.calls = script, refuse, []

    def complete(self, *, messages, **kw):
        self.calls.append(dict(kw))
        model = kw.get("model")
        if model and model == self.refuse:
            raise PluginTrustError(f"plugin 'branchpoint' may not override model '{model}'")
        purpose = str(kw.get("purpose") or "").replace("branchpoint:", "")
        handler = self.script.get(purpose)
        if handler is None:
            raise RuntimeError(f"no script for purpose {purpose!r}")
        system, user = messages[0]["content"], messages[1]["content"]
        raw = handler(system, user) if callable(handler) else handler
        if isinstance(raw, list):
            idx = sum(1 for c in self.calls if str(c.get("purpose") or "").endswith(purpose)) - 1
            raw = raw[min(idx, len(raw) - 1)]
        return Reply(str(raw))


def test_a_refusal_reaches_the_run_warnings_and_the_decision_log(repo, data):
    """A refused override during planning must show up in the run's warnings and the decision log."""
    L = mod("llm")
    backend = ScriptedBackend(make_llm().script)
    run, _anchor = build_run(repo, data, llm=L.CtxLLM(SimpleNamespace(llm=backend), default_model="judge-x"),
                             extra_settings={"judge_model": "judge-x", "reviewer_model": "judge-x"})
    run.prepare()       # RED: the refusal lived on one object and no run surface ever said it
    assert any("judge-x" in w for w in run.meta["warnings"]), run.meta["warnings"]
    assert run.meta.get("models_refused") == ["judge-x"], run.meta.get("models_refused")
    entries = mod("declog").read_entries(run.dir)
    assert any("judge-x" in json.dumps(e) for e in entries), entries


def test_the_start_message_does_not_claim_a_model_that_was_refused(repo, data):
    """"Reviewers and judge: X" must name the model that actually runs once a refusal is known."""
    import importlib
    L = mod("llm")
    commands = importlib.import_module("hermes_branchpoint.commands")
    backend = ScriptedBackend(make_llm().script)
    run, anchor = build_run(repo, data, llm=L.CtxLLM(SimpleNamespace(llm=backend), default_model="judge-x"),
                            extra_settings={"judge_model": "judge-x", "reviewer_model": "judge-x"})
    strategies = run.prepare()
    text = commands.BranchpointService(None, llm=run.llm)._start_text(run, strategies, anchor, "", run.settings)
    models_line = next(l for l in text.splitlines() if l.startswith("Models: "))
    assert "was refused" in models_line, models_line


def test_the_standalone_cli_treats_a_failed_run_as_a_failure(repo, data):
    C, P = mod("commands"), mod("paths")
    P.set_data_dir_override(data)
    try:
        run, _anchor = build_run(repo, data)
        run.set_status("failed")
        assert C.BranchpointService(None, llm=make_llm()).last_run_failed() is True
    finally:
        P.set_data_dir_override(None)



# --------------------------------------------------------------------------- budget hard stop


def test_remaining_budget_accounts_for_the_parent_meter():
    L = mod("llm")
    m = L.Meter(100_000)
    assert m.remaining() == 100_000
    m.charge(95_000)
    assert m.remaining() == 5_000
    assert L.Meter().remaining() is None                     # no cap: unlimited
    child = m.child("route B", 17_500)
    assert child.remaining() == 5_000                        # the run's remaining budget wins over the share


def test_a_route_share_cannot_exceed_what_is_left_of_the_run_budget():
    R = mod("route")
    assert R.child_token_cap(17_500, 4_000) == 4_000         # RED: the marker carried the full share
    assert R.child_token_cap(17_500, None) == 17_500
    assert R.child_token_cap(None, 4_000) == 4_000
    assert R.child_token_cap(None, None) is None
    assert R.child_token_cap(17_500, 0) == 0


@pytest.fixture
def zero_cap_route(repo, monkeypatch, tmp_path):
    """A route marker whose remaining token budget is exactly zero."""
    R, P = mod("route"), mod("protocol")
    wt = tmp_path / "wt" / "B"
    R.worktree_add(repo, wt, git(repo, "rev-parse", "HEAD"), branch="branchpoint/t/B")
    marker = {"route": "B", "strategy": {"axis": "infrastructure", "thesis": "use redis", "key_decisions": []},
              "confine_root": str(wt), "events_path": str(tmp_path / "events.jsonl"),
              "checks_log": str(tmp_path / "checks.jsonl"), "check_every": 5, "max_tool_calls": 60,
              "token_cap": 0, "checks": [], "base_failures": []}
    P.write_marker(wt, marker)
    monkeypatch.chdir(wt)
    return wt


def test_an_exhausted_share_of_exactly_zero_blocks_further_tool_calls(zero_cap_route):
    """`token_cap: 0` means nothing left, but a falsy check treated it as 'no cap at all'."""
    H = mod("hooks")
    res = H.Hooks().pre_tool_call(tool_name="read_file", args={"path": "service.py"})
    assert res and res["action"] == "block", res
    assert "token budget" in res["message"], res["message"]


# --------------------------------------------------------------------------- standalone CLI exit code


def test_cli_exits_nonzero_when_the_command_reports_an_error(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "branchpoint"), "fork", "--state-dir", str(tmp_path / "state")],
                       capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert r.returncode == 1, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"        # RED: main() always returned 0
    assert "Give the goal" in (r.stdout + r.stderr)


def test_cli_keeps_zero_for_a_command_that_worked(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "branchpoint"), "list", "--state-dir", str(tmp_path / "state")],
                       capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert r.returncode == 0, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
