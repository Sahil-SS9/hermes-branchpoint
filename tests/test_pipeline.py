"""The whole pipeline offline: anchor -> diverge -> simulate -> execute -> council -> adopt."""
from __future__ import annotations

import re

import pytest
from bp_helpers import fake_builder, git, make_llm, mod, write


def build_run(repo, data, *, mode="fork", n=3, execute="top2", llm=None, base=None, extra_settings=None):
    A, R, AG = mod("anchor"), mod("run"), mod("agent")
    llm = llm or make_llm()
    if mode == "fork":
        anchor = A.resolve_fork(cwd=repo, goal="add caching to the slow /report endpoint")
    else:
        anchor = A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="s1", n=1, base_override=base,
                                  goal_override="add caching to the slow /report endpoint", messages=[])
    settings = {"n": n, "execute": execute, "budget_tokens": 5_000_000, "max_parallel": 3, "approach": "assigned", **(extra_settings or {})}
    run = R.Run.create(data, anchor=anchor, mode=mode, goal=anchor.goal, settings=settings, llm=llm,
                       agent=AG.CallableRunner(fake_builder), baseline_stats={"tool_calls": 14, "tokens": 52_000, "seconds": 240.0})
    return run


def clean(repo):
    return git(repo, "status", "--porcelain")


def test_fork_pipeline_end_to_end(repo, data):
    head_before = git(repo, "rev-parse", "HEAD")
    run = build_run(repo, data)
    strategies = run.prepare()
    assert [s["id"] for s in strategies] == ["A", "B", "C"]
    assert len({s["axis"] for s in strategies}) == 3
    assert (run.wt_root / "base").is_dir()
    run.go(background=False)

    assert run.meta["status"] == "decided", run.meta.get("warnings")
    ledgers = {l.route: l for l in run.ledgers()}
    executed = run.meta["executed"]
    assert len(executed) == 2
    for rid, led in ledgers.items():
        assert led.status == "done" and led.is_complete, (rid, led.completeness())
        assert led.fidelity == ("execute" if rid in executed else "simulate")
    # --- isolation: nothing leaked into the user's tree
    assert clean(repo) == ""
    assert git(repo, "rev-parse", "HEAD") == head_before
    # --- executed ledgers carry real evidence
    for rid in executed:
        led = ledgers[rid]
        assert led.diff_stats["predicted"] is False and led.diff_stats["files"] >= 1
        assert led.head_sha and led.branch == f"branchpoint/{run.run_id}/{rid}"
        assert any(r.startswith(f"{rid}:diff:h") for r in mod("ledger").EvidenceStore(run.dir / "routes" / rid, rid).refs())
        assert led.cost.tool_calls == 0 or led.cost.estimated is False
    # --- redis route: new dep + config + failing test are captured as issues
    by_axis = {l.strategy["axis"]: l for l in ledgers.values()}
    if by_axis["infrastructure"].fidelity == "execute":
        redis = by_axis["infrastructure"]
        assert redis.deps_added == ["redis"] and any("REDIS_URL" in c for c in redis.blast.config)
        assert any(i.kind == "test" and i.severity == "high" for i in redis.issues)
        assert redis.reversibility <= 3
    # --- decision: every cited ref resolves, bogus claims were dropped
    md = (run.dir / "decision.md").read_text()
    L = mod("ledger")
    rev = L.RunEvidence(run.dir)
    refs = set(re.findall(r"\[`([A-Z]\d?:[a-z]+:[\w.\-/#]+)`\]", md))
    assert refs and all(rev.exists(r) for r in refs), [r for r in refs if not rev.exists(r)]
    assert "40x" not in md and "No evidence at all" not in md
    assert "withheld" in md                                  # integrity note about dropped sentences
    assert run.meta["winner"] in ledgers


def test_blind_review_never_shows_route_letters(repo, data):
    llm = make_llm()
    run = build_run(repo, data)
    run.prepare()
    run.go(background=False)
    reviews = [c for c in llm.calls if c["purpose"].startswith("review:")]
    # make_llm() above is not the run's llm; use the run's calls instead
    calls = [c for c in run.llm.calls if c["purpose"].startswith("review:")]
    assert len(calls) == 3
    for c in calls:
        body = c["user"]
        assert re.search(r"=== R\d", body)
        assert not re.search(r"(?m)^=== [A-E] ", body)
        assert not re.search(r"\b[A-E]:(trace|diff|test|plan|issue):", body)   # refs are relabelled too
    orders = [tuple(re.findall(r"(?m)^=== (R\d)", c["user"])) for c in calls]
    assert all(sorted(o) == ["R1", "R2", "R3"] for o in orders)
    del reviews


def test_aggregate_weights_and_tie_break():
    C, L = mod("council"), mod("ledger")

    def led(rid, blast, files):
        l = L.Ledger(route=rid, strategy={"name": rid}, status="done", fidelity="execute")
        l.blast.score, l.diff_stats["files"] = blast, files
        return l

    def rv(scores):
        return C.ReviewerResult("x", scores, {}, {})

    flat = {c: 5 for c in C.CRITERIA}
    reviews = [rv({"A": dict(flat), "B": dict(flat)})]
    rows = C.aggregate([led("A", 6, 3), led("B", 2, 9)], reviews)
    assert rows[0]["route"] == "B"                     # equal score: smaller blast radius wins
    assert rows[0]["score"] == 5.0
    better = dict(flat, goal_fit=10)
    rows = C.aggregate([led("A", 1, 1), led("B", 9, 9)], [rv({"A": dict(flat), "B": better})])
    assert rows[0]["route"] == "B" and rows[0]["score"] == round(5 * 0.7 + 10 * 0.3, 2)   # goal fit weighs 30%


def test_simulated_correctness_is_capped():
    C, L = mod("council"), mod("ledger")
    from bp_helpers import make_llm as ml
    sim = L.Ledger(route="B", strategy={"name": "s"}, status="done", fidelity="simulate")
    sim.blast.basis = "x"
    import json
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    (d / "routes" / "B").mkdir(parents=True)
    llm = ml()
    llm.script["review:Skeptic"] = lambda s, u: json.dumps({"scores": {"R1": {c: 10 for c in C.CRITERIA} | {"worst_flaw": {"text": "x", "refs": []}}}})
    r = C.run_reviewer(llm, "Skeptic", [sim], L.RunEvidence(d), "goal", __import__("random").Random(1))
    assert r.scores["B"]["correctness"] == C.SIMULATED_CORRECTNESS_CAP


def test_adopt_merge_commits_and_undo_restores(repo, data):
    run = build_run(repo, data, execute="all")
    run.prepare()
    run.go(background=False)
    ad = mod("adopt")
    led = {l.route: l for l in run.ledgers()}
    green = next(r for r, l in led.items() if not [i for i in l.issues if i.kind == "test" and i.severity == "high"])
    before = git(repo, "rev-parse", "HEAD")
    msg = ad.adopt_merge(repo, run.dir, run.meta, green)
    assert "Committed as one commit" in msg and "PASS" in msg
    assert git(repo, "rev-parse", "HEAD") != before and clean(repo) == ""
    assert run.meta["adopted"] == green
    assert "restored" in ad.undo_adopt(repo, run.dir, run.meta)
    assert git(repo, "rev-parse", "HEAD") == before and clean(repo) == ""      # commit taken back out, files restored
    from bp_helpers import SERVICE
    assert (repo / "service.py").read_text() == SERVICE


def test_adopt_refuses_unbuilt_route_and_cleanup(repo, data):
    run = build_run(repo, data, execute="none")
    run.prepare()
    run.go(background=False)
    ad = mod("adopt")
    with pytest.raises(mod("util").BranchpointError, match="nothing to merge|did not finish a build"):
        ad.adopt_merge(repo, run.dir, run.meta, "B")
    text, path = ad.replay_prompt(run.dir, run.meta, "B")
    assert "GOAL:" in text and path.exists() and "Plan:" in text
    assert "docs/decisions/0001-" in ad.adopt_adr(repo, run.dir, run.meta, run.meta["winner"])
    assert (repo / "docs" / "decisions").is_dir()
    run.drop_refs()
    removed = run.cleanup()
    assert "base" in removed and not run.wt_root.exists()
    assert "branchpoint/" not in git(repo, "branch", "--list")


def test_retrospective_baseline_and_transform(repo, data):
    """The user already built the in-memory cache (uncommitted): that's route A; others are alternatives."""
    from bp_helpers import ETAG, IN_MEMORY, STUB2
    write(repo, "service.py", IN_MEMORY)
    write(repo, "tests/test_cache.py", "from service import get_report\n\n\ndef test_cached():\n    assert get_report(2) is get_report(2)\n")
    base = git(repo, "rev-parse", "HEAD")
    llm = make_llm({"strategies": make_llm.__globals__["STRATS"]["strategies"][1:]})
    run = build_run(repo, data, mode="rewind", n=3, execute="all", llm=llm, base=base)
    assert run.meta["has_baseline"] is True
    strategies = run.prepare()
    assert strategies[0]["baseline"] and strategies[0]["id"] == "A"
    run.go(background=False)
    ledgers = {l.route: l for l in run.ledgers()}
    a = ledgers["A"]
    assert a.fidelity == "baseline" and a.status == "done" and a.diff_stats["files"] == 2
    assert a.cost.tool_calls == 14 and a.cost.estimated            # taken from the transcript, labelled as an estimate
    assert not [i for i in a.issues if i.kind == "test" and i.severity == "high"]
    # the user's tree is still exactly route A, untouched by the run
    assert (repo / "service.py").read_text() == IN_MEMORY
    # adopt another executed route: working tree is transformed from A to that route's tree
    ad = mod("adopt")
    target = next(r for r in ("C", "B") if ledgers[r].status == "done" and not [i for i in ledgers[r].issues if i.severity == "high" and i.kind == "test"])
    msg = ad.adopt_merge(repo, run.dir, run.meta, target)
    assert "uncommitted" in msg                                       # dirty tree: left for review, not committed
    tree_target = git(repo, "rev-parse", f"{ledgers[target].head_sha}^{{tree}}")
    snap = mod("util").snapshot_tree(repo)
    assert git(repo, "rev-parse", f"{snap}^{{tree}}") == tree_target
    del ETAG, STUB2


def test_budget_hard_stop(repo, data):
    """Once the meter is exhausted no model call is made at all, and the run ends cleanly with what it had."""
    L = mod("llm")
    llm = make_llm()
    run = build_run(repo, data, execute="all", llm=llm, extra_settings={"budget_tokens": 2500})
    try:
        run.prepare()
        run.go(background=False)
    except L.BudgetExceeded:
        pass
    assert run.meter.exhausted
    spent_at_stop = run.meter.used
    calls_at_stop = len(llm.calls)
    # nothing more can be asked of the model; every attempt is refused before it reaches the backend
    with pytest.raises(L.BudgetExceeded):
        run.llm.complete("s", "u", purpose="diverge")
    assert len(llm.calls) == calls_at_stop and run.meter.used == spent_at_stop
    assert run.meta["status"] in ("failed", "created", "diverging", "decided")
    assert not run.meta.get("winner") or run.meta["warnings"]          # a partial decision must say it was cut short
    assert clean(repo) == ""


def test_abort_marks_run(repo, data):
    run = build_run(repo, data)
    run.prepare()
    run.abort_run()
    run.go(background=False)
    assert run.meta["status"] in ("aborted", "decided", "failed")
    assert clean(repo) == ""


def test_inline_refs_are_stripped_and_replaced_by_one_link():
    C = mod("council")
    t = C.clean_text("B has the smallest blast radius [B:trace:blast], so it wins B:plan:md. Also (A:issue:p2) matters.")
    assert t == "B has the smallest blast radius, so it wins. Also matters."
    assert C.find_refs(t) == []


def test_confidence_is_computed_from_evidence_not_tone():
    C, L = mod("council"), mod("ledger")

    def led(rid, fid):
        return L.Ledger(route=rid, strategy={"name": rid}, status="done", fidelity=fid)

    flat = {c: 5 for c in C.CRITERIA}
    win = dict(flat, goal_fit=10)
    rv = lambda p, a, b: C.ReviewerResult(p, {"A": a, "B": b}, {}, {})   # noqa: E731
    reviews = [rv("m", win, flat), rv("s", win, flat), rv("u", win, flat)]
    ranking = C.aggregate([led("A", "execute"), led("B", "execute")], reviews)
    by_id = {"A": led("A", "execute"), "B": led("B", "execute")}
    assert C.compute_confidence(ranking, by_id, reviews)[0] == "high"            # clear margin, unanimous, built
    by_id["A"] = led("A", "simulate")
    assert C.compute_confidence(ranking, by_id, reviews)[0] == "medium"          # winner was only simulated
    # three routes, three reviewers, three different favourites: no consensus -> low
    def rv3(p, fav):
        return C.ReviewerResult(p, {r: (dict(flat, goal_fit=9) if r == fav else dict(flat)) for r in "ABC"}, {}, {})

    split = [rv3("m", "A"), rv3("s", "B"), rv3("u", "C")]
    leds = {r: led(r, "execute") for r in "ABC"}
    ranking3 = C.aggregate(list(leds.values()), split)
    assert C.compute_confidence(ranking3, leds, split)[0] == "low"
