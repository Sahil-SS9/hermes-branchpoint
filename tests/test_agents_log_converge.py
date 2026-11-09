"""Independent agents, the 2-8 route range, separate judge/reviewer models, the decision log and converge."""
from __future__ import annotations

import json

import pytest
from bp_helpers import INDEPENDENT_AXIS, fake_builder, independent_plan, make_llm, mod
from test_pipeline import build_run, clean

from hermes_branchpoint.declog import DecisionLog, render_markdown, summarise
from hermes_branchpoint.util import BranchpointError


def indep_llm(axis_for=None):
    llm = make_llm()
    axis_for = axis_for or {}
    for rid in "ABCDEFGH":
        llm.script[f"simulate:{rid}"] = (lambda r: lambda system, user: json.dumps(independent_plan(r, axis_for.get(r))))(rid)
    return llm


def run_independent(repo, data, **kw):
    return build_run(repo, data, n=3, llm=indep_llm(kw.pop("axis_for", None)), extra_settings={"approach": "independent", **kw.pop("extra", {})})


# ------------------------------------------------------------------ independent agents

def test_independent_agents_are_told_only_the_task_and_choose_their_own_approach(repo, data):
    run = run_independent(repo, data)
    strategies = run.prepare()
    assert [s["id"] for s in strategies] == ["A", "B", "C"]
    assert all(s["independent"] and not s["thesis"] for s in strategies)          # nothing assigned
    run.go(background=False)
    sims = [c for c in run.llm.calls if c["purpose"].startswith("simulate:")]
    assert sims and all("no assigned strategy" in c["user"] and "STRATEGY" not in c["user"] for c in sims)
    names = {l.route: l.strategy["name"] for l in run.ledgers()}
    assert names == {"A": "In-process TTL cache", "B": "Shared Redis cache", "C": "HTTP ETag caching"}   # the agents' own words
    assert {s["id"]: s["axis"] for s in run.meta["strategies"]} == INDEPENDENT_AXIS    # absorbed into the run
    assert run.meta["status"] == "decided" and run.meta["winner"]
    assert clean(repo) == ""


def test_build_prompt_carries_the_agents_own_plan(repo, data):
    seen = []
    A = mod("agent")

    def builder(job):
        seen.append(job.prompt)
        return fake_builder(job)

    run = run_independent(repo, data)
    run.agent = A.CallableRunner(builder)
    run.prepare()
    run.go(background=False)
    assert seen and all("YOU planned this approach yourself" in p and "YOUR PLAN" in p for p in seen)


def test_two_agents_choosing_the_same_thing_are_built_once(repo, data):
    run = run_independent(repo, data, axis_for={"B": "api-contract", "C": "api-contract"})   # B and C pick the same approach
    run.prepare()
    run.go(background=False)
    c = next(l for l in run.ledgers() if l.route == "C")
    assert c.strategy.get("converged_with") == "B" and "almost the same approach" in " ".join(c.notes)
    assert "C" not in run.meta["executed"]
    dup = [e for e in DecisionLog(run.dir).entries() if e["kind"] == "duplicate"]
    assert dup and dup[0]["route"] == "C" and "independently chose almost the same approach" in dup[0]["what"]


def test_an_agent_matching_what_you_built_is_not_built_again(repo, data):
    from bp_helpers import IN_MEMORY, git, write
    write(repo, "service.py", IN_MEMORY)                      # the user already built the in-memory cache: route A
    base = git(repo, "rev-parse", "HEAD")
    run = build_run(repo, data, mode="rewind", n=3, llm=indep_llm({"B": "library-choice"}), extra_settings={"approach": "independent"}, base=base)
    assert run.meta["has_baseline"] is True
    run.prepare()
    run.go(background=False)
    b = next(l for l in run.ledgers() if l.route == "B")
    assert b.strategy.get("converged_with") == "A"            # agent B landed on what you built
    assert "B" not in run.meta["executed"]
    assert any("(what you built)" in e["what"] for e in DecisionLog(run.dir).entries() if e["kind"] == "duplicate")


def test_assigned_mode_still_hands_out_distinct_strategies(repo, data):
    run = build_run(repo, data)
    strategies = run.prepare()
    assert len({s["axis"] for s in strategies}) == 3 and not any(s.get("independent") for s in strategies)


# ------------------------------------------------------------------ settings

def test_route_count_runs_from_two_to_eight_with_a_default_of_four():
    C = mod("commands")
    svc = C.BranchpointService(None, llm=make_llm())
    assert svc.settings()["n"] == 4
    assert svc.settings({"n": "2"})["n"] == 2 and svc.settings({"n": "8"})["n"] == 8
    for bad in ("1", "9"):
        with pytest.raises(BranchpointError, match="between 2 and 8"):
            svc.settings({"n": bad})


def test_models_judge_reviewer_and_approach_are_settable():
    C = mod("commands")
    svc = C.BranchpointService(None, llm=make_llm(), get_setting=lambda k, d: {"models": "m1, m2", "judge_model": "j", "approach": "assigned"}.get(k, d))
    s = svc.settings()
    assert s["models"] == ["m1", "m2"] and s["judge_model"] == "j" and s["approach"] == "assigned"
    s = svc.settings({"models": "x,y,z", "judge_model": "jj", "reviewer_model": "rr", "approach": "independent"})
    assert s["models"] == ["x", "y", "z"] and s["judge_model"] == "jj" and s["reviewer_model"] == "rr" and s["approach"] == "independent"
    with pytest.raises(BranchpointError, match="independent"):
        svc.settings({"approach": "chaos"})


def test_each_agent_the_reviewers_and_the_judge_get_their_own_model(repo, data):
    run = run_independent(repo, data, extra={"models": ["agent-a", "agent-b", "agent-c"], "judge_model": "big-judge", "reviewer_model": "rev-model"})
    run.prepare()
    run.go(background=False)
    by = {}
    for c in run.llm.calls:
        by.setdefault(c["purpose"], set()).add(c["model"])
    assert by["simulate:A"] == {"agent-a"} and by["simulate:B"] == {"agent-b"} and by["simulate:C"] == {"agent-c"}
    assert all(by[f"review:{p}"] == {"rev-model"} for p in ("Maintainer", "Skeptic", "User Advocate"))
    assert by["judge"] == {"big-judge"}
    agents = next(e for e in DecisionLog(run.dir).entries() if e["kind"] == "agents")
    assert "A = agent-a" in agents["why"] and "big-judge" in agents["why"] and "rev-model" in agents["why"]


def test_reviewers_fall_back_to_the_judge_model(repo, data):
    run = run_independent(repo, data, extra={"judge_model": "big-judge"})
    run.prepare()
    run.go(background=False)
    assert {c["model"] for c in run.llm.calls if c["purpose"].startswith("review:")} == {"big-judge"}


# ------------------------------------------------------------------ decision log

def test_the_decision_log_records_what_was_decided_and_why(repo, data):
    run = build_run(repo, data, mode="rewind", n=3, base="HEAD")
    run.prepare()
    run.go(background=False)
    ents = DecisionLog(run.dir).entries()
    kinds = [e["kind"] for e in ents]
    for k in ("anchor", "agents", "plan", "select", "build", "council"):
        assert k in kinds, k
    assert all(e["what"] and e["why"] for e in ents)                                   # nothing logged without a reason
    plan = next(e for e in ents if e["kind"] == "plan")
    assert plan["route"] in ("A", "B", "C") and plan["thinking"] and plan["refs"]            # the agent's own reasoning is kept
    build = [e for e in ents if e["kind"] == "build"]
    assert {e["route"] for e in build} == set(run.meta["executed"])
    verdict = next(e for e in ents if e["kind"] == "council")
    assert f"Route {run.meta['winner']}" in verdict["what"] and "confidence" in verdict["why"]
    md = (run.dir / "decision-log.md").read_text()
    assert md.startswith("# Decision log") and "## In short" in md and "**Because:**" in md
    brief = summarise(ents)
    assert f"Route {run.meta['winner']}" in brief and "won with" in brief and "built" in brief and "confidence" in brief


def test_a_failed_build_is_logged_with_its_reason(repo, data):
    run = build_run(repo, data, n=3, execute="all")
    run.prepare()
    run.go(background=False)
    bad = [e for e in DecisionLog(run.dir).entries() if e["kind"] == "build" and e["data"].get("failing")]
    assert bad and "failed" in bad[0]["why"]            # the redis route cannot import redis


def test_log_rendering_never_invents_a_reason(tmp_path):
    log = DecisionLog(tmp_path)
    log.record("note", "A thing", "Did a thing.", "Because the test said so.")
    text = render_markdown(log.entries(), {"goal": "g"})
    assert "Because the test said so." in text and "Task:** g" in text
    assert summarise([]) == ""


# ------------------------------------------------------------------ converge

def decided_run(repo, data, builder=None):
    run = build_run(repo, data, n=3, execute="top2")
    if builder:
        run.agent = mod("agent").CallableRunner(builder)
    run.prepare()
    run.go(background=False)
    return run


def converge_builder(prompts):
    def builder(job):
        prompts.append(job.prompt)
        if "Axis: converged" in job.prompt:
            (job.worktree / "tests" / "test_converged.py").write_text("def test_converged():\n    assert True\n")
        return fake_builder(job)
    return builder


def test_converge_builds_from_the_winner_plus_borrowed_ideas_and_is_judged_blind(repo, data):
    prompts: list[str] = []
    run = decided_run(repo, data, converge_builder(prompts))
    winner, cap_before = run.meta["winner"], run.meter.cap
    led = run.converge()
    assert led.route == "Z" and led.status == "done" and "Z" in run.meta["routes"]
    assert led.strategy["converged_from"] == winner and led.strategy["axis"] == "converged"
    z_prompt = next(p for p in prompts if "Axis: converged" in p)
    assert "CONVERGE" in z_prompt and "its trace shows what to watch." in z_prompt and f"Route {winner}" in z_prompt
    assert (run.dir / "decision.pre-converge.json").exists()                           # the first verdict is kept
    final = json.loads((run.dir / "decision.json").read_text())
    assert "Z" in {r["route"] for r in final["ranking"]}                                # Z was judged with everyone else
    assert run.meter.cap == cap_before + 150_000                                        # bounded, announced extra budget
    assert run.meta["converged"]["outcome"]
    ents = [e for e in DecisionLog(run.dir).entries() if e["kind"] == "converge"]
    assert len(ents) == 2 and "Result of converging" in ents[-1]["title"]
    assert clean(repo) == ""


def test_the_converged_route_diffs_against_the_original_base(repo, data):
    prompts: list[str] = []
    run = decided_run(repo, data, converge_builder(prompts))
    winner = run.meta["winner"]
    wled = next(l for l in run.ledgers() if l.route == winner)
    z = run.converge()
    assert z.diff_stats["files"] >= wled.diff_stats["files"]          # the winner's changes are inside Z's diff, plus the new test
    parent = mod("util").git(["rev-parse", f"branchpoint/{run.run_id}/Z^"], repo).strip()
    assert parent == run.meta["base_sha"]


def test_converge_needs_a_verdict_and_only_runs_once(repo, data):
    run = build_run(repo, data, n=3)
    run.prepare()
    with pytest.raises(BranchpointError, match="run the council first"):
        run.converge()
    run.go(background=False)
    run.converge()
    with pytest.raises(BranchpointError, match="already has a converged route"):
        run.converge()


def test_converge_with_nothing_to_borrow_says_so(repo, data):
    run = decided_run(repo, data)
    dec = json.loads((run.dir / "decision.json").read_text())
    dec["narrative"]["borrow"], dec["narrative"]["risks_carried"] = [], []
    (run.dir / "decision.json").write_text(json.dumps(dec))
    with pytest.raises(BranchpointError, match="nothing to converge"):
        run.converge()
    assert "Z" not in run.meta["routes"]


def test_the_converged_route_can_be_adopted_like_any_other(repo, data):
    prompts: list[str] = []
    run = decided_run(repo, data, converge_builder(prompts))
    run.converge()
    msg = mod("adopt").adopt_merge(run.repo, run.dir, run.meta, "Z")
    assert "Route Z" in msg or "Z" in msg
