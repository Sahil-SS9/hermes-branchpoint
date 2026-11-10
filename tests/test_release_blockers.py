"""Release blockers from the independent QA pass.

D1 a build agent that did not finish must not count as a finished route (and must never win);
D2 a refused model during converge must leave a failed, honest run that can be retried;
D3 a refusal belongs to one run, so a new run in the same session asks the host again;
D4 the README must not promise a whole-run token hard stop that the Cost section contradicts.

Every test here is written against observed behaviour, not intent: the RED line notes what the
candidate did before the fix.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from bp_helpers import fake_builder, make_llm, mod

ROOT = Path(__file__).resolve().parent.parent


class Refused(PermissionError):
    """Stands in for the host's PluginLlmTrustError."""


class GatedBackend:
    """A host ctx.llm clone that refuses chosen models until the test grants them."""

    def __init__(self, script, refuse=("judge-x",)) -> None:
        self.script, self.refuse, self.calls = script, set(refuse), []

    def complete(self, *, messages, **kw):
        self.calls.append(dict(kw))
        model = kw.get("model")
        if model and model in self.refuse:
            raise Refused(f"plugin 'branchpoint' is not allowed to use model '{model}'")
        purpose = str(kw.get("purpose") or "").replace("branchpoint:", "")
        handler = self.script.get(purpose)
        if handler is None:
            raise RuntimeError(f"no script for purpose {purpose!r}")
        system, user = messages[0]["content"], messages[1]["content"]
        raw = handler(system, user) if callable(handler) else handler
        if isinstance(raw, list):
            idx = sum(1 for c in self.calls if str(c.get("purpose") or "").endswith(purpose)) - 1
            raw = raw[min(idx, len(raw) - 1)]
        return SimpleNamespace(text=str(raw), model=model or "your-default",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5, cost_usd=None))

    def grant(self, model: str) -> None:
        self.refuse.discard(model)


def build_run(repo, data, *, llm=None, agent=None, execute="none", extra_settings=None):
    A, R, AG = mod("anchor"), mod("run"), mod("agent")
    anchor = A.resolve_fork(cwd=repo, goal="add caching to the slow /report endpoint")
    settings = {"n": 3, "execute": execute, "budget_tokens": 5_000_000, "max_parallel": 2,
                "approach": "assigned", **(extra_settings or {})}
    return R.Run.create(data, anchor=anchor, mode="fork", goal=anchor.goal, settings=settings,
                        llm=llm or make_llm(),
                        agent=agent or AG.CallableRunner(lambda job: None))


def ledgers(run) -> dict:
    return {l.route: l for l in run.ledgers()}


def ledger_json(run, rid: str) -> dict:
    return json.loads((run.dir / "routes" / rid / "ledger.json").read_text())


# --------------------------------------------------------------------------- D1: unfinished builds


def failing_builder(job, *, rc=1, timed_out=False, write=True):
    """The build process dies (this is what `hermes chat -m <bad model>` looks like to the plugin)."""
    from hermes_branchpoint.agent import AgentResult
    if write:
        fake_builder(job)
    return AgentResult(rc, "hermes chat failed: model refused by host", "", 0.4, timed_out=timed_out)


def converge_builder(job):
    """A build that actually writes something (so converge's route Z is a real, finished build)."""
    from hermes_branchpoint.agent import AgentResult
    fake_builder(job)
    (job.worktree / "tests" / "test_converged.py").write_text("def test_converged():\n    assert True\n")
    return AgentResult(0, "converged the winner with the borrowed ideas", "", 1.0)


def test_a_build_agent_that_exits_nonzero_cannot_finish_a_route(repo, data):
    """RED: route.py only appended a note, so a dead build agent still produced a green `done` route."""
    run = build_run(repo, data, agent=mod("agent").CallableRunner(failing_builder), execute="all")
    run.prepare()
    run.go(background=False)

    executed = run.meta["executed"]
    assert executed, run.meta
    for rid in executed:
        led = ledgers(run)[rid]
        assert led.status == "failed", (rid, led.status, led.notes)
        assert not led.is_complete, (rid, led.completeness())
        stop = [i for i in led.issues if i.kind == "runtime" and "exited with code 1" in i.detail]
        assert stop, (rid, [(i.kind, i.detail) for i in led.issues])
        assert stop[0].severity == "high", stop[0]
        evidence = mod("ledger").RunEvidence(run.dir)
        assert evidence.exists(stop[0].evidence_ref), stop[0].evidence_ref
        # the checks and cost measured on the failed build are kept, not thrown away with the status
        refs = mod("ledger").EvidenceStore(run.dir / "routes" / rid, rid).refs()
        assert any(r.endswith(":stat:cost") for r in refs), refs
        assert any(r.endswith(":test:final") for r in refs), refs
        assert led.cost.tokens > 0, led.cost

    # nothing finished, so there is no council verdict and no winner to adopt
    assert run.meta["status"] == "failed", run.meta
    assert not (run.dir / "decision.json").exists()
    assert not (run.dir / "decision.md").exists()
    assert run.meta.get("winner") is None


def test_a_timed_out_build_cannot_finish_a_route(repo, data):
    """RED: a wall-clock timeout added a note and then let the route finish as if the build had ended."""
    run = build_run(repo, data, agent=mod("agent").CallableRunner(lambda job: failing_builder(job, rc=0, timed_out=True)),
                    execute="top1")
    run.prepare()
    run.go(background=False)

    rid = run.meta["executed"][0]
    led = ledgers(run)[rid]
    assert led.status == "failed", (led.status, led.notes)
    assert led.diff_stats["files"] >= 1            # the partial work is still measured, not thrown away
    stop = [i for i in led.issues if i.kind == "runtime" and "wall-clock cap" in i.detail]
    assert stop and stop[0].severity == "high", [(i.kind, i.detail) for i in led.issues]
    # the simulated routes can still be judged, but the timed-out build is not one of the candidates
    decision = json.loads((run.dir / "decision.json").read_text())
    assert rid not in {r["route"] for r in decision["ranking"]}, decision["ranking"]
    assert decision["winner"] != rid


def test_an_executed_route_that_changed_nothing_cannot_be_done(repo, data):
    """An execute route with an empty diff delivered nothing; it must not be a candidate the council can pick."""
    run = build_run(repo, data, agent=mod("agent").CallableRunner(lambda job: None), execute="top1")
    run.prepare()
    run.go(background=False)

    rid = run.meta["executed"][0]
    led = ledgers(run)[rid]
    assert led.status == "failed", (led.status, led.summary)
    assert led.diff_stats["files"] == 0
    assert any(i.kind == "runtime" and "without changing any file" in i.detail for i in led.issues), \
        [(i.kind, i.detail) for i in led.issues]
    decision = json.loads((run.dir / "decision.json").read_text())
    assert rid not in {r["route"] for r in decision["ranking"]} and decision["winner"] != rid


def test_a_failed_route_is_never_ranked_or_chosen(repo, data):
    """One route's build dies; the others finish, so the council must judge only what finished."""
    def builder(job):
        from hermes_branchpoint.agent import AgentResult
        if "ROUTE C STRATEGY" in job.prompt:
            return AgentResult(1, "hermes chat failed: model refused by host", "", 0.4)
        return fake_builder(job)

    run = build_run(repo, data, agent=mod("agent").CallableRunner(builder), execute="all")
    run.prepare()
    run.go(background=False)

    leds = ledgers(run)
    assert leds["C"].status == "failed", leds["C"].status
    assert {l.route for l in leds.values() if l.status == "done"} == {"A", "B"}
    assert run.meta["status"] == "decided", run.meta
    decision = json.loads((run.dir / "decision.json").read_text())
    ranked = {r["route"] for r in decision["ranking"]}
    assert ranked == {"A", "B"}, ranked
    assert decision["winner"] in {"A", "B"} and run.meta["winner"] in {"A", "B"}
    md = (run.dir / "decision.md").read_text()
    assert "Routes that did not finish" in md and "Route C" in md


def test_a_route_cost_is_marked_estimated_when_the_child_reported_no_usage(repo, data):
    """RED: the fallback token estimate was labelled measured (`estimated=False`) despite the note."""
    run = build_run(repo, data, agent=mod("agent").CallableRunner(lambda job: fake_builder(job)), execute="top1")
    run.prepare()
    run.go(background=False)

    led = ledgers(run)[run.meta["executed"][0]]
    assert led.status == "done", (led.status, led.notes)
    assert any("tokens estimated" in n for n in led.notes), led.notes
    assert led.cost.estimated is True, led.cost


def test_a_route_cost_is_measured_when_the_child_reports_usage(repo, data):
    """The honest label cuts both ways: real reported usage stays measured."""
    def builder(job):
        marker = mod("protocol").read_marker(job.worktree) or {}
        fake_builder(job)
        events = Path(marker["events_path"])
        with events.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "usage", "tokens": 4_321}) + "\n")
            fh.write(json.dumps({"type": "tool", "tool": "write_file", "path": "service.py"}) + "\n")
        from hermes_branchpoint.agent import AgentResult
        return AgentResult(0, "built it", "", 1.0)

    run = build_run(repo, data, agent=mod("agent").CallableRunner(builder), execute="top1")
    run.prepare()
    run.go(background=False)

    led = ledgers(run)[run.meta["executed"][0]]
    assert led.cost.tokens == 4_321 and led.cost.estimated is False, led.cost
    assert led.cost.tool_calls == 1, led.cost


def test_a_missing_exit_code_is_not_treated_as_a_failure(repo, data):
    """Some AgentResult producers leave ``rc`` unset; an absent code is not evidence that the build crashed."""
    from hermes_branchpoint.agent import AgentResult

    def builder(job):
        fake_builder(job)
        return AgentResult(None, "the runner reported no exit code", "", 0.5)   # type: ignore[arg-type]

    run = build_run(repo, data, agent=mod("agent").CallableRunner(builder), execute="all")
    run.prepare()
    run.go(background=False)

    assert {l.route for l in run.ledgers() if l.status == "done"} == set(run.meta["executed"])
    assert run.meta["status"] == "decided", run.meta


# --------------------------------------------------------------------------- D2: converge fail-closed


def test_a_refused_model_during_converge_fails_the_run_and_keeps_the_previous_decision(repo, data):
    """RED: the refusal escaped with status "council", no error and no warnings recorded."""
    L = mod("llm")
    backend = GatedBackend(make_llm().script, refuse=("judge-x",))
    run = build_run(repo, data, llm=L.CtxLLM(SimpleNamespace(llm=backend)), execute="none",
                    agent=mod("agent").CallableRunner(converge_builder))
    run.prepare()
    run.go(background=False)
    assert run.meta["status"] == "decided", run.meta
    before = (run.dir / "decision.json").read_text()
    assert run.meta["winner"]

    run.settings["judge_model"] = "judge-x"
    run.settings["reviewer_model"] = ""
    run.save()
    with pytest.raises(L.ModelOverrideRefused) as caught:
        run.converge()
    assert "judge-x" in str(caught.value)

    assert run.meta["status"] == "failed", run.meta["status"]
    assert "judge-x" in str(run.meta.get("error") or ""), run.meta.get("error")
    assert any("judge-x" in w for w in run.meta.get("warnings") or []), run.meta.get("warnings")
    # the verdict that existed before the attempt is untouched, and its own backup exists
    assert (run.dir / "decision.json").read_text() == before
    assert (run.dir / "decision.pre-converge.json").exists()
    assert "Z" in run.meta["routes"]
    assert not (run.meta.get("converged") or {}).get("judged")      # no fake verdict for the converged route
    zled = mod("ledger").load_ledger(run.dir / "routes" / "Z")
    assert zled is not None and zled.status in ("done", "failed"), (zled.status if zled else None)


def test_a_converge_that_stopped_during_the_build_marks_route_z_failed(repo, data, monkeypatch):
    """A Z that never finished must not sit there as a queued route pretending the converge is still running."""
    L, R = mod("llm"), mod("run")
    run = build_run(repo, data, execute="none", agent=mod("agent").CallableRunner(converge_builder))
    run.prepare()
    run.go(background=False)
    assert run.meta["status"] == "decided"

    def refuse(*_a, **_k):
        raise L.ModelOverrideRefused("Hermes refused the model override 'judge-x' (host refused it)")

    monkeypatch.setattr(R, "execute_route", refuse)
    with pytest.raises(L.ModelOverrideRefused):
        run.converge()

    zled = mod("ledger").load_ledger(run.dir / "routes" / "Z")
    assert zled is not None and zled.status == "failed", (zled.status if zled else None)
    assert "converge stopped" in zled.summary, zled.summary
    assert run.meta["status"] == "failed" and "judge-x" in str(run.meta.get("error") or "")
    assert (run.dir / "decision.pre-converge.json").exists()


def test_a_failed_converge_can_be_retried_once_the_model_is_granted(repo, data):
    """RED: Z stayed in `routes`, so every retry hit "already has a converged route" and the run was wedged."""
    L, R, AG = mod("llm"), mod("run"), mod("agent")
    backend = GatedBackend(make_llm().script, refuse=("judge-x",))
    run = build_run(repo, data, llm=L.CtxLLM(SimpleNamespace(llm=backend)), execute="none",
                    agent=AG.CallableRunner(converge_builder))
    run.prepare()
    run.go(background=False)
    run.settings["judge_model"] = "judge-x"
    run.settings["reviewer_model"] = ""
    run.save()
    with pytest.raises(L.ModelOverrideRefused):
        run.converge()
    assert run.meta["status"] == "failed"

    # a second attempt in the same process is a new attempt, so it asks the host again instead of reading a
    # stale refusal; the failed Z is cleared, so this stops for the refusal and nothing else.
    with pytest.raises(L.ModelOverrideRefused) as again:
        run.converge()
    assert "already has a converged route" not in str(again.value), again.value

    backend.grant("judge-x")
    call_before = sum(1 for c in backend.calls if c.get("model") == "judge-x")

    # the real retry path: a new command loads the run again, so it gets a fresh refusal record
    retried = R.Run.load(data, run.run_id, llm=L.CtxLLM(SimpleNamespace(llm=backend)),
                         agent=AG.CallableRunner(converge_builder))
    led = retried.converge()
    assert led.route == "Z" and led.status == "done", (led.status, led.summary, led.notes)
    assert retried.meta["status"] == "decided", retried.meta
    assert sum(1 for c in backend.calls if c.get("model") == "judge-x") > call_before   # the host was asked again
    decision = json.loads((retried.dir / "decision.json").read_text())
    assert "Z" in {r["route"] for r in decision["ranking"]}
    assert retried.meta["routes"].count("Z") == 1          # the failed attempts did not leave duplicate entries


def test_a_failed_converge_can_be_retried_in_the_same_service_once_the_model_is_granted(repo, data):
    """RED: get_run caches the Run, so the retry reused the first attempt's refusal record: the grant was never
    tried (0 host calls), the same "already refused earlier in this run" came back, and a full Z was rebuilt first.
    The plugin's service lives as long as the gateway, so the documented "run again" was unreachable exactly where
    it was aimed (set a judge model, get refused, grant, retry).
    """
    L, C, P, AG = mod("llm"), mod("commands"), mod("paths"), mod("agent")
    P.set_data_dir_override(data)
    try:
        backend = GatedBackend(make_llm().script, refuse=())
        svc = C.BranchpointService(None, llm=L.CtxLLM(SimpleNamespace(llm=backend)),
                                   agent=AG.CallableRunner(converge_builder))
        svc.cwd = lambda: str(repo)
        svc.handle('fork "add caching to the slow /report endpoint" --n 2 --execute none --wait')
        rid = svc.latest_run_id()
        run = svc.get_run(rid)
        assert run.meta["status"] == "decided", run.meta
        before = (run.dir / "decision.json").read_text(encoding="utf-8")
        run.settings["judge_model"] = "judge-x"
        run.settings["reviewer_model"] = ""
        run.save()

        backend.refuse.add("judge-x")               # the host has not granted the override yet
        svc.handle("converge --wait")
        assert run.meta["status"] == "failed", run.meta
        assert "judge-x" in str(run.meta.get("error") or ""), run.meta.get("error")
        asked = sum(1 for c in backend.calls if c.get("model") == "judge-x")
        assert asked >= 1, "the refused model was never tried at all"
        # the failure is honest: the earlier verdict is untouched and no verdict was invented for Z
        assert (run.dir / "decision.json").read_text(encoding="utf-8") == before
        assert not (run.meta.get("converged") or {}).get("judged")
        # the reset is scoped to the attempt, not a hole in the guard: inside it the refusal is still remembered
        denials = sum(1 for c in backend.calls if c.get("model") == "judge-x")
        with pytest.raises(L.ModelOverrideRefused):
            run.llm.complete("s", "u", model="judge-x", purpose="review:Maintainer")
        assert sum(1 for c in backend.calls if c.get("model") == "judge-x") == denials

        backend.grant("judge-x")                    # the user follows the error's advice
        before_calls = len(backend.calls)
        svc.handle("converge --wait")
        assert svc.get_run(rid) is run               # same service, cached Run: the seam this bug lived on
        assert run.meta["status"] == "decided", (run.meta.get("status"), run.meta.get("error"))
        assert not run.meta.get("error"), run.meta.get("error")   # the retry that worked no longer reads as failed
        retry = backend.calls[before_calls:]
        assert sum(1 for c in retry if c.get("model") == "judge-x") > 0, "the grant was never tried"
        # fail-closed in the other direction too: nothing answered on the default model instead of judge-x
        assert retry and all(c.get("model") == "judge-x" for c in retry), [c.get("model") for c in retry]
        assert (run.meta.get("converged") or {}).get("judged") is True
        decision = json.loads((run.dir / "decision.json").read_text(encoding="utf-8"))
        assert "Z" in {r["route"] for r in decision["ranking"]}
        assert run.meta["routes"].count("Z") == 1    # repeated attempts do not stack Z entries
    finally:
        P.set_data_dir_override(None)


# --------------------------------------------------------------------------- D3: refusal scope


def test_a_new_run_in_the_same_session_asks_the_host_again_after_the_grant(repo, data):
    """RED: the refusal cache lived as long as the CtxLLM, so a granted override was never re-tried."""
    L, C, P = mod("llm"), mod("commands"), mod("paths")
    P.set_data_dir_override(data)
    try:
        backend = GatedBackend(make_llm().script, refuse=("judge-x",))
        svc = C.BranchpointService(None, llm=L.CtxLLM(SimpleNamespace(llm=backend)))
        svc.cwd = lambda: str(repo)

        first = svc.handle('fork "add caching to the slow /report endpoint" --n 2 --execute none '
                           '--judge-model judge-x --reviewer-model judge-x --wait')
        assert "Branchpoint run" in first, first
        failed_run = svc.get_run(svc.latest_run_id())
        assert failed_run.meta["status"] == "failed", (failed_run.meta.get("status"), failed_run.meta.get("warnings"))
        assert "judge-x" in str(failed_run.meta.get("error") or ""), failed_run.meta.get("error")
        assert svc.last_run_failed() is True
        calls_at_failure = sum(1 for c in backend.calls if c.get("model") == "judge-x")
        assert calls_at_failure >= 1

        backend.grant("judge-x")            # the user grants the setting; the host re-reads it per call
        second = svc.handle('fork "add caching to the slow /report endpoint" --n 2 --execute none '
                            '--judge-model judge-x --reviewer-model judge-x --wait')
        assert "Branchpoint run" in second, second
        latest = svc.get_run(svc.latest_run_id())
        assert latest.run_id != failed_run.run_id
        assert latest.meta["status"] == "decided", (latest.meta.get("status"), latest.meta.get("warnings"))
        assert latest.meta["winner"]
        assert sum(1 for c in backend.calls if c.get("model") == "judge-x") > calls_at_failure
    finally:
        P.set_data_dir_override(None)


def test_one_run_still_refuses_a_denied_model_without_asking_the_host_twice(repo, data):
    """The other half of the contract: inside a single run the refusal is remembered, never retried."""
    L = mod("llm")

    class Deny(GatedBackend):
        pass

    backend = Deny(make_llm().script, refuse=("judge-x",))
    llm = L.CtxLLM(SimpleNamespace(llm=backend))
    run = build_run(repo, data, llm=llm, execute="none", extra_settings={"judge_model": "judge-x"})
    run.prepare()
    run.go(background=False)

    assert run.meta["status"] == "failed"
    denials = sum(1 for c in backend.calls if c.get("model") == "judge-x")
    assert denials <= 3, denials                      # three blind reviewers run in parallel, then stop
    with pytest.raises(L.ModelOverrideRefused):
        run.llm.complete("s", "u", model="judge-x", purpose="review:Maintainer")
    assert sum(1 for c in backend.calls if c.get("model") == "judge-x") == denials


# --------------------------------------------------------------------------- D4: README truth


def test_the_readme_does_not_promise_a_whole_run_token_hard_stop():
    """RED: the options line said `--budget` is a hard stop for the whole run; Cost says otherwise."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "hard stop for the whole run" not in readme
    options = readme[readme.index("Options: `--repo"):]
    options = options[: options.index("\n\n")]
    assert "--budget" in options and "per-build cap" in options, options
