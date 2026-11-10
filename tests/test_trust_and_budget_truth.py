"""Release-truth regressions: a refused model override must stop the run, not be swapped for another model;
the token budget's "hard stop" must actually bind the build agent's share; the standalone CLI must fail loudly.

These pin the claims made in README "Cost" / "Honest limits" and in the plugin catalog.
"""
from __future__ import annotations

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


def test_a_refused_override_raises_instead_of_answering_on_another_model():
    """Every model Branchpoint passes is a deliberate choice: a refusal must stop the call, never swap the model."""
    L = mod("llm")
    backend = RefusingBackend()
    llm = L.CtxLLM(SimpleNamespace(llm=backend))
    with pytest.raises(L.BranchpointError) as caught:
        llm.complete("s", "u", model="judge-x", purpose="judge")
    assert isinstance(caught.value, L.ModelOverrideRefused)
    assert "judge-x" in str(caught.value) and "allow_model_override" in str(caught.value), str(caught.value)
    assert len(backend.calls) == 1, backend.calls       # RED: it retried once without the model and returned "ok"


def test_a_refused_model_stays_refused_on_every_later_call_from_any_meter_clone():
    """One refusal, then fail closed: later calls must not quietly answer on the default model."""
    L = mod("llm")
    backend = RefusingBackend()
    llm = L.CtxLLM(SimpleNamespace(llm=backend))
    route_llm = llm.with_meter(L.Meter(10_000))
    for caller in (route_llm, llm, route_llm):
        with pytest.raises(L.BranchpointError, match="judge-x"):
            caller.complete("s", "u", model="judge-x", purpose="judge")
    assert len(backend.calls) == 1, backend.calls       # RED: 3 refusals and 2 default-model answers
    assert llm.refused_models == {"judge-x"}, llm.refused_models


def test_a_refused_default_model_is_an_explicit_choice_too():
    """A ``default_model`` on the backend is still a model the run chose, so a refusal there must also stop."""
    L = mod("llm")
    backend = RefusingBackend("judge-x")
    llm = L.CtxLLM(SimpleNamespace(llm=backend), default_model="judge-x")
    with pytest.raises(L.BranchpointError, match="allow_model_override"):
        llm.complete("s", "u")
    assert len(backend.calls) == 1 and backend.calls[0]["model"] == "judge-x", backend.calls


def test_a_permission_error_that_is_not_a_model_refusal_propagates_unchanged():
    """Only a trust refusal is ours to interpret; any other permission error must surface as it is."""
    L = mod("llm")
    calls: list[dict] = []

    class Backend:
        def complete(self, *, messages, **kw):
            calls.append(dict(kw))
            raise PermissionError("/var/lib/branchpoint is not writable")

    llm = L.CtxLLM(SimpleNamespace(llm=Backend()))
    with pytest.raises(PermissionError, match="not writable"):
        llm.complete("s", "u", model="m1", purpose="judge")
    assert len(calls) == 1, calls
    assert llm.refused_models == set()


def test_a_model_shaped_error_with_no_model_requested_is_not_translated():
    """With no model asked for there is no override to refuse: the host's error must pass straight through."""
    L = mod("llm")
    calls: list[dict] = []

    class Backend:
        def complete(self, *, messages, **kw):
            calls.append(dict(kw))
            raise PermissionError("the plugin may not override model policy")

    llm = L.CtxLLM(SimpleNamespace(llm=Backend()))
    with pytest.raises(PermissionError, match="may not override"):
        llm.complete("s", "u")
    assert "model" not in calls[0], calls
    assert llm.refused_models == set()


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


def test_a_refused_reviewer_model_fails_the_run_instead_of_deciding_on_a_default(repo, data):
    """A refused judge/reviewer model must end the run, not produce a Decision Record from another model."""
    L = mod("llm")
    backend = ScriptedBackend(make_llm().script)
    run, _anchor = build_run(repo, data, llm=L.CtxLLM(SimpleNamespace(llm=backend)),
                             extra_settings={"judge_model": "judge-x", "reviewer_model": "judge-x"})
    run.prepare()                       # planning calls carry no model override, so they still work
    assert run.meta["status"] == "planned"
    run.go(background=False)            # simulate (no builds) then the council
    assert run.meta["status"] == "failed", run.meta
    assert "judge-x" in str(run.meta.get("error") or ""), run.meta.get("error")
    assert not (run.dir / "decision.json").exists()          # RED: a decision was written by the default model
    assert not (run.dir / "decision.md").exists()


def test_a_refused_model_stops_the_fork_before_any_start_message(repo, data):
    """The command surface must report the refusal as an error, not print a run that is secretly on another model."""
    L, C, P = mod("llm"), mod("commands"), mod("paths")
    P.set_data_dir_override(data)
    try:
        svc = C.BranchpointService(None, llm=L.CtxLLM(SimpleNamespace(llm=RefusingBackend("judge-x")), default_model="judge-x"))
        svc.cwd = lambda: str(repo)
        # `assigned` planning asks the model for strategies during start, so the refusal happens before any run text
        out = svc.handle('fork "add caching to the slow /report endpoint" --n 2 --execute none --approach assigned')
        assert out.startswith("Branchpoint: "), out
        assert "judge-x" in out and "allow_model_override" in out, out
        assert "Models:" not in out, out                     # RED: it printed a start message with a swapped model
        assert svc.last_run_failed() is True
    finally:
        P.set_data_dir_override(None)


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
