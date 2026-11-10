"""Run one route: simulate it (plan + static analysis) or execute it (real build in its own worktree).

Both paths end in the same place: a complete ``Ledger`` whose every claim has an ``evidence_ref``.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import blast as blastmod
from .agent import AgentJob, AgentRunner
from .checks import CheckResult, CheckSpec, detect_checks, run_checks
from .diverge import read_excerpts, repo_map
from .events import EventBus
from .guard import HooksNotConfirmed, verify_handshake
from .ledger import (Cost, EvidenceStore, Issue, Ledger, reversibility_from_facts, save_ledger)
from .llm import LLM, BudgetExceeded, Meter, ModelOverrideRefused, redact_for_model
from .protocol import write_marker
from .util import (GIT_IDENTITY, BranchpointError, extract_json, git, read_json, short, tail, worktree_lock)

SIM_SYSTEM = (
    "You are a senior engineer producing an implementation plan for ONE specific route of a design "
    "exploration. Do not write the code. Be concrete (real file paths and symbols from the repo map), "
    "honest about risks, and do not drift to a different approach than the route's strategy. Reply with JSON only."
)

SIM_SYSTEM_INDEPENDENT = (
    "You are one of several engineers who were each given the SAME task, independently. Nobody has told you how to "
    "approach it and you cannot see anyone else's plan. Choose the approach you judge best, state it, and plan it "
    "concretely (real file paths and symbols from the repo map), honest about risks. Do not write the code. Reply with JSON only."
)

SIM_SCHEMA = (
    '{"summary":"<2 sentences: what this route builds>",'
    '"plan_md":"<markdown, numbered implementation steps>",'
    '"files":[{"path":"<repo path>","action":"create|modify|delete","symbols":["<function/class touched>"],"est_lines":<int>,"contract_change":<true if clients of this symbol/route would see a different contract>}],'
    '"risks":[{"detail":"<specific bug/edge case/operational risk>","severity":"low|medium|high"}],'
    '"new_deps":["<package>"],"new_service":<true if a new runtime service must be operated>,'
    '"config_keys":["<ENV_VAR or config key introduced>"],"schema_changes":["<migration/schema change>"],'
    '"public_api_changes":["<externally visible API change>"],'
    '"effort":{"tool_calls":<int>,"minutes":<int>,"tokens":<int>},'
    '"reversibility":{"score":<1-5, 5 = trivially undone>,"why":"<one sentence>"},'
    '"test_plan":"<how this would be verified>"}'
)


APPROACH_SCHEMA = (
    '"approach":{"name":"<3-5 words>","axis":"<short label for the kind of approach, e.g. in-process cache, external service, '
    'client-side, precomputed>","thesis":"<one sentence: the bet your approach makes>","key_decisions":["<concrete decision>"],'
    '"trade_off":"<what your approach gives up>"},'
)
SIM_SCHEMA_INDEPENDENT = SIM_SCHEMA.replace('{"summary"', "{" + APPROACH_SCHEMA + '"summary"', 1)


def _log(rc: Any, kind: str, title: str, what: str, why: str, **kw: Any) -> None:
    dl = getattr(rc, "declog", None)
    if dl is not None:
        try:
            dl.record(kind, title, what, why, **kw)
        except Exception:   # the log must never break a run
            pass


@dataclass
class RouteSpec:
    id: str
    strategy: dict[str, Any]
    model: str | None = None
    start_sha: str | None = None   # build on top of this commit instead of the base (used by converge)
    guide: str = ""               # extra instructions for the build agent (used by converge)


@dataclass
class RunContext:
    run_id: str
    repo: Path
    run_dir: Path
    wt_root: Path
    base_sha: str
    goal: str
    bus: EventBus
    abort: threading.Event
    llm: LLM
    agent: AgentRunner
    meter: Meter
    settings: dict[str, Any] = field(default_factory=dict)
    followups: list[str] = field(default_factory=list)
    context: str = ""
    route_token_cap: int | None = None
    declog: Any = None
    _graph: blastmod.ImportGraph | None = None
    _base_checks: list[CheckResult] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def base_root(self) -> Path:
        return self.wt_root / "base"

    def route_dir(self, rid: str) -> Path:
        d = self.run_dir / "routes" / rid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def worktree_path(self, rid: str) -> Path:
        return self.wt_root / rid

    def branch(self, rid: str) -> str:
        return f"branchpoint/{self.run_id}/{rid}"

    def graph(self) -> blastmod.ImportGraph:
        with self._lock:
            if self._graph is None:
                self._graph = blastmod.build_graph(self.base_root)
            return self._graph

    def check_specs(self, root: Path | None = None) -> list[CheckSpec]:
        return detect_checks(root or self.base_root, self.settings)

    def base_checks(self) -> list[CheckResult]:
        with self._lock:
            if self._base_checks is None:
                self._base_checks = run_checks(self.base_root, self.check_specs(self.base_root))
                self.bus.emit("base_checks", results=[r.to_dict() for r in self._base_checks])
            return self._base_checks


# --------------------------------------------------------------------------- git worktree helpers

def worktree_add(repo: Path, path: Path, rev: str, branch: str | None = None) -> None:
    with worktree_lock(repo):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _worktree_remove_unlocked(repo, path)
        args = ["worktree", "add", "--quiet"]
        args += ["-b", branch, str(path), rev] if branch else ["--detach", str(path), rev]
        for attempt in range(4):        # external Git processes may still hold ordinary ref locks
            try:
                git(args, repo, timeout=180)
                return
            except BranchpointError as exc:
                if attempt == 3 or not re.search(r"\.lock|lock file|Unable to create|another git process", str(exc), re.I):
                    raise
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                time.sleep(0.2 * (attempt + 1))


def _worktree_remove_unlocked(repo: Path, path: Path) -> None:
    git(["worktree", "remove", "--force", str(path)], repo, check=False, timeout=120)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    git(["worktree", "prune"], repo, check=False)


def worktree_remove(repo: Path, path: Path) -> None:
    with worktree_lock(repo):
        _worktree_remove_unlocked(repo, path)


def delete_branch(repo: Path, branch: str) -> None:
    with worktree_lock(repo):
        git(["branch", "-D", branch], repo, check=False)


def link_ignored(repo: Path, wt: Path, dirs: list[str]) -> list[str]:
    """Symlink untracked dependency dirs (.venv, node_modules) into a worktree so checks can run."""
    linked: list[str] = []
    for d in dirs:
        src, dst = repo / d, wt / d
        if src.exists() and not dst.exists() and not dst.is_symlink():
            try:
                os.symlink(src, dst)
                linked.append(d)
            except OSError:
                pass
    return linked


def commit_worktree(wt: Path, message: str, exclude: list[str]) -> str:
    specs = ["."] + [f":(exclude){e}" for e in exclude]
    git(["add", "-A", "--", *specs], wt, timeout=120)
    git(["commit", "-q", "--allow-empty", "--no-verify", "--no-gpg-sign", "-m", message], wt, env=GIT_IDENTITY, timeout=120)
    return git(["rev-parse", "HEAD"], wt).strip()


# --------------------------------------------------------------------------- diff evidence

def diff_numstat(repo: Path, base: str, head: str) -> tuple[list[dict[str, Any]], int, int]:
    out = git(["diff", "--numstat", "--no-renames", base, head], repo, check=False)
    files: list[dict[str, Any]] = []
    ins = dels = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        a = int(parts[0]) if parts[0].isdigit() else 0
        d = int(parts[1]) if parts[1].isdigit() else 0
        files.append({"path": parts[2], "insertions": a, "deletions": d, "source": "diff"})
        ins += a
        dels += d
    st = git(["diff", "--name-status", "--no-renames", base, head], repo, check=False)
    actions = {l.split("\t")[-1]: {"A": "create", "D": "delete"}.get(l[0], "modify") for l in st.splitlines() if "\t" in l}
    for f in files:
        f["action"] = actions.get(f["path"], "modify")
    return files, ins, dels


def diff_hunks(repo: Path, base: str, head: str, max_hunks: int = 40) -> list[tuple[str, str, str]]:
    text = git(["diff", "-U2", "--no-color", "--no-ext-diff", "--no-renames", base, head], repo, check=False, timeout=120)
    hunks: list[tuple[str, str, str]] = []
    cur_file = ""
    cur: list[str] = []
    header = ""

    def flush() -> None:
        nonlocal cur, header
        if cur and header:
            hunks.append((cur_file, header, "\n".join(cur)))
        cur, header = [], ""

    for line in text.splitlines():
        if line.startswith("diff --git"):
            flush()
            m = re.search(r" b/(.+)$", line)
            cur_file = m.group(1) if m else ""
        elif line.startswith("@@"):
            flush()
            header = line.split("@@")[1].strip() if line.count("@@") >= 2 else line
            cur = [line]
        elif cur:
            cur.append(line)
    flush()
    return hunks[:max_hunks]


# --------------------------------------------------------------------------- publishing

def publish(rc: RunContext, led: Ledger, **extra: Any) -> None:
    counts: dict[str, int] = {}
    for i in led.issues:
        counts[i.kind] = counts.get(i.kind, 0) + 1
    rc.bus.emit(
        "route", route=led.route, name=led.strategy.get("name", ""), axis=led.strategy.get("axis", ""),
        thesis=led.strategy.get("thesis", ""), status=led.status, fidelity=led.fidelity,
        blast=led.blast.score, blast_parts={"public_api": len(led.blast.public_api), "schema": len(led.blast.schema),
                                              "config": len(led.blast.config), "dependents": led.blast.dependents,
                                              "deps": len(led.deps_added)},
        issues=counts, issue_total=len(led.issues), cost=asdict(led.cost), files=led.diff_stats,
        budget_used=rc.meter.used, budget_cap=rc.meter.cap,
        failing=len([i for i in led.issues if i.kind in ("test", "lint", "type") and i.severity == "high"]),
        reversibility=led.reversibility, summary=short(led.summary, 240),
        model=led.strategy.get("model") or "", independent=bool(led.strategy.get("independent")),
        converged_with=led.strategy.get("converged_with") or "", converged_from=led.strategy.get("converged_from") or "",
        borrowed_from=led.strategy.get("borrowed_from") or [], **extra)


def _blank_ledger(rc: RunContext, spec: RouteSpec, fidelity: str) -> Ledger:
    return Ledger(route=spec.id, strategy={**spec.strategy, "id": spec.id}, fidelity=fidelity, base_sha=rc.base_sha,
                  branch=rc.branch(spec.id) if fidelity == "execute" else "")


# --------------------------------------------------------------------------- simulate

def _coerce_plan(data: dict[str, Any]) -> dict[str, Any]:
    def lst(k: str) -> list[Any]:
        v = data.get(k)
        return v if isinstance(v, list) else []

    files = []
    for f in lst("files"):
        if isinstance(f, dict) and f.get("path"):
            files.append({"path": str(f["path"]).lstrip("./"), "action": str(f.get("action") or "modify").lower(),
                          "symbols": [str(s) for s in (f.get("symbols") or [])][:8],
                          "est_lines": int(f.get("est_lines") or 0) if str(f.get("est_lines") or 0).lstrip("-").isdigit() else 0,
                          "contract_change": bool(f.get("contract_change"))})
    risks = []
    for r in lst("risks"):
        if isinstance(r, dict) and r.get("detail"):
            sev = str(r.get("severity") or "medium").lower()
            risks.append({"detail": short(r["detail"], 300), "severity": sev if sev in ("low", "medium", "high") else "medium"})
        elif isinstance(r, str) and r.strip():
            risks.append({"detail": short(r, 300), "severity": "medium"})
    eff = data.get("effort") if isinstance(data.get("effort"), dict) else {}
    rev = data.get("reversibility") if isinstance(data.get("reversibility"), dict) else {}
    ap = data.get("approach") if isinstance(data.get("approach"), dict) else {}
    return {
        "approach": ap,
        "summary": short(data.get("summary") or "", 500), "plan_md": str(data.get("plan_md") or ""), "files": files,
        "risks": risks, "new_deps": [str(x) for x in lst("new_deps")], "new_service": bool(data.get("new_service")),
        "config_keys": [str(x) for x in lst("config_keys")], "schema_changes": [str(x) for x in lst("schema_changes")],
        "public_api_changes": [str(x) for x in lst("public_api_changes")], "test_plan": short(data.get("test_plan") or "", 400),
        "effort": {"tool_calls": int(eff.get("tool_calls") or 0) if str(eff.get("tool_calls") or 0).isdigit() else 0,
                   "minutes": int(eff.get("minutes") or 0) if str(eff.get("minutes") or 0).isdigit() else 0,
                   "tokens": int(eff.get("tokens") or 0) if str(eff.get("tokens") or 0).isdigit() else 0},
        "reversibility": {"score": int(rev.get("score") or 3) if str(rev.get("score") or 3).isdigit() else 3, "why": short(rev.get("why") or "", 240)},
    }


def simulate_route(rc: RunContext, spec: RouteSpec) -> Ledger:
    t0 = time.monotonic()
    rdir = rc.route_dir(spec.id)
    store = EvidenceStore(rdir, spec.id)
    led = _blank_ledger(rc, spec, "simulate")
    led.status = "simulating"
    save_ledger(rdir, led)
    publish(rc, led)
    independent = bool(spec.strategy.get("independent"))
    route_meter = rc.meter.child(f"route {spec.id}", rc.route_token_cap)
    llm = rc.llm.with_meter(route_meter) if hasattr(rc.llm, "with_meter") else rc.llm  # type: ignore[attr-defined]
    try:
        graph = rc.graph()
        pf = [p for p in spec.strategy.get("predicted_files") or []]
        existing = [p for p in pf if p.lstrip("./") in graph.files]
        excerpt_paths = existing or sorted(f for f in graph.files if f.endswith((".py", ".js", ".ts")) and not blastmod.is_test_path(f))[:6 if independent else 4]
        if independent:
            user = "\n\n".join([
                f"GOAL:\n{redact_for_model(rc.goal)}",
                f"YOU ARE AGENT {spec.id}. Decide your own approach to the goal; there is no assigned strategy.",
                f"REPO MAP:\n{repo_map(rc.base_root, 5500)}",
                f"RELEVANT FILES:\n{redact_for_model(read_excerpts(rc.base_root, excerpt_paths, 9000))}",
                f"State your approach and plan it. Reply with JSON only: {SIM_SCHEMA_INDEPENDENT}",
            ])
        else:
            user = "\n\n".join([
                f"GOAL:\n{redact_for_model(rc.goal)}",
                f"ROUTE {spec.id} STRATEGY (axis: {spec.strategy.get('axis')}):\n{spec.strategy.get('thesis')}\nKey decisions:\n- " + "\n- ".join(spec.strategy.get("key_decisions") or []),
                f"REPO MAP:\n{repo_map(rc.base_root, 5500)}",
                f"RELEVANT FILES:\n{redact_for_model(read_excerpts(rc.base_root, excerpt_paths, 9000))}",
                f"Produce the implementation plan for this route only. Reply with JSON only: {SIM_SCHEMA}",
            ])
        reply = llm.complete(SIM_SYSTEM_INDEPENDENT if independent else SIM_SYSTEM, user, max_tokens=4000, model=spec.model, purpose=f"simulate:{spec.id}")
        plan = _coerce_plan(extract_json(reply.text))
        if not plan["files"]:
            raise BranchpointError("the plan listed no files")
    except ModelOverrideRefused:
        raise                      # a refused model is final for the run; marking one route failed would hide it
    except BudgetExceeded as exc:
        led.status, led.summary = "failed", f"stopped: {exc}"
        led.notes.append(str(exc))
        save_ledger(rdir, led)
        publish(rc, led)
        return led
    except Exception as exc:
        led.status, led.summary = "failed", f"simulation failed: {short(exc, 200)}"
        save_ledger(rdir, led)
        publish(rc, led)
        return led

    if independent:
        ap = plan["approach"]
        led.strategy.update({
            "name": short(ap.get("name") or f"Agent {spec.id}'s approach", 60),
            "axis": short(str(ap.get("axis") or "own approach").lower(), 40),
            "thesis": short(ap.get("thesis") or plan["summary"], 300),
            "key_decisions": [short(x, 200) for x in (ap.get("key_decisions") or [])][:8],
            "trade_off": short(ap.get("trade_off") or "", 200),
            "predicted_files": [f["path"] for f in plan["files"]][:15],
            "plan_md": short(plan["plan_md"], 3500)})
    store.put("plan", "Implementation plan", plan["plan_md"] or plan["summary"], key="md")
    store.put("plan", "Files the route would touch", json.dumps(plan["files"], indent=1), key="files")
    facts = blastmod.analyze_plan(rc.base_root, plan, graph)
    trace_ref = store.put("trace", "Blast radius trace (static, on the base checkout)", facts.trace_text(), key="blast")
    led.blast = facts.to_blast(trace_ref)
    led.deps_added = facts.deps_added
    led.files = [{"path": f["path"], "action": f["action"], "insertions": f["est_lines"], "deletions": 0,
                  "symbols": f["symbols"], "source": "predicted"} for f in plan["files"]]
    led.diff_stats = {"files": len(plan["files"]), "insertions": sum(f["est_lines"] for f in plan["files"]),
                      "deletions": 0, "predicted": True}
    for i, risk in enumerate(plan["risks"], 1):
        ref = store.put("issue", f"Predicted risk ({risk['severity']})", risk["detail"], key=f"p{i}")
        led.issues.append(Issue("predicted", risk["detail"], ref, risk["severity"]))
    score, why = reversibility_from_facts(schema=bool(facts.schema), public_api=bool(facts.public_api),
                                          new_deps=len(facts.deps_added), config=len(facts.config), infra=facts.infra,
                                          files=len(plan["files"]))
    judged = max(1, min(5, plan["reversibility"]["score"]))
    led.reversibility = (score + judged) // 2
    led.reversibility_notes = f"measured from plan facts: {why}. model-judged {judged}/5: {plan['reversibility']['why']}"
    eff = plan["effort"]
    led.cost = Cost(tokens=eff["tokens"] or 40_000, seconds=float(eff["minutes"] * 60 or 600),
                    tool_calls=eff["tool_calls"] or 25, estimated=True)
    led.summary = plan["summary"] or short(spec.strategy.get("thesis", ""), 300)
    top = next((r for r in plan["risks"] if r["severity"] == "high"), plan["risks"][0] if plan["risks"] else None)
    who = f"Agent {spec.id}" + (f" ({spec.model})" if spec.model else "")
    _log(rc, "plan", f"{who} chose its approach" if independent else f"{who} plans its assigned approach",
         f"{who}: {led.strategy.get('name')}. {led.strategy.get('thesis')}",
         ("Nobody told it how to do this; it judged this approach best. " if independent else f"Assigned the '{led.strategy.get('axis')}' direction. ")
         + f"Plan touches {len(plan['files'])} file(s); impact {led.blast.score}/10; reversibility {led.reversibility}/5"
         + (f"; biggest risk it named: {top['detail']}" if top else "") + ".",
         route=spec.id, thinking=" ".join(filter(None, [("Key decisions: " + "; ".join(led.strategy.get("key_decisions") or []) + ".") if led.strategy.get("key_decisions") else "",
                                                       ("Gives up: " + led.strategy["trade_off"] + ".") if led.strategy.get("trade_off") else "", plan["summary"]])),
         refs=[f"{spec.id}:plan:md", trace_ref], data={"model": spec.model or "", "independent": independent})
    led.notes.append(f"plan produced with {route_meter.used} tokens in {time.monotonic() - t0:.0f}s (not part of the estimated build cost)")
    led.status = "done"
    save_ledger(rdir, led)
    publish(rc, led, planning_tokens=route_meter.used)
    return led


def child_token_cap(share: int | None, remaining: int | None) -> int | None:
    """The token cap written into a route's marker.

    A route gets its share of the run budget, but never more than the run actually has left when the build
    starts: a share is only real while the run can still pay for it. ``None`` means nothing is capped.
    """
    if share is None:
        return remaining
    if remaining is None:
        return int(share)
    return max(0, min(int(share), int(remaining)))


def rank_for_execution(ledgers: list[Ledger]) -> list[Ledger]:
    """Which simulated routes deserve a real build? Lower blast, fewer/lighter predicted risks, easier undo."""
    sev = {"low": 0.5, "medium": 1.0, "high": 2.0}

    def score(l: Ledger) -> float:
        risk = sum(sev.get(i.severity, 1.0) for i in l.issues if i.kind == "predicted")
        return (10 - l.blast.score) * 1.0 + l.reversibility * 0.8 - risk * 0.6

    return sorted((l for l in ledgers if l.status == "done"), key=score, reverse=True)


# --------------------------------------------------------------------------- shared: diff -> ledger

_FAIL_ID = re.compile(r"\s+-\s+.*$")


def _norm_fail(kind: str, f: str) -> str:
    return _FAIL_ID.sub("", f).strip() if kind == "test" else f.strip()


def apply_diff_facts(rc: RunContext, led: Ledger, store: EvidenceStore, head_sha: str, extra_notes: list[str] | None = None) -> None:
    files, ins, dels = diff_numstat(rc.repo, rc.base_sha, head_sha)
    led.files = files
    led.diff_stats = {"files": len(files), "insertions": ins, "deletions": dels, "predicted": False}
    stat_ref = store.put("stat", "Diff stat vs base", "\n".join(f"{f['path']}  +{f['insertions']} -{f['deletions']} ({f['action']})" for f in files) or "(no changes)", key="diff")
    for n, (path, header, text) in enumerate(diff_hunks(rc.repo, rc.base_sha, head_sha), 1):
        store.put("diff", f"{path} @@ {header}", text, key=f"h{n}")
    facts = blastmod.analyze_diff(rc.repo, rc.base_sha, head_sha, rc.graph())
    trace_ref = store.put("trace", "Blast radius trace (static, on the real diff)", facts.trace_text(), key="blast")
    led.blast = facts.to_blast(trace_ref)
    led.deps_added = facts.deps_added
    if facts.deps_added:
        store.put("dep", "New dependencies", "\n".join(facts.deps_added), key="added")
    score, why = reversibility_from_facts(schema=bool(facts.schema), public_api=bool(facts.public_api),
                                          new_deps=len(facts.deps_added), config=len(facts.config), infra=facts.infra,
                                          files=len(files))
    led.reversibility, led.reversibility_notes = score, f"measured on the real diff: {why}"
    led.notes.extend(extra_notes or [])
    _ = stat_ref


def apply_check_issues(led: Ledger, store: EvidenceStore, results: list[CheckResult], base_results: list[CheckResult]) -> None:
    base_fail = {(r.kind, _norm_fail(r.kind, f)) for r in base_results if not r.passed and not r.skipped for f in r.failures}
    for r in results:
        ref = store.put(r.kind, f"{r.kind}: {r.cmd} → {r.summary}", f"$ {r.cmd}\n(exit {r.rc}, {r.seconds:.1f}s) {r.summary}\n\n{r.output}", key="final")
        if r.skipped:
            led.notes.append(f"{r.kind} evidence unavailable: {r.summary} [{ref}]")
            continue
        if r.passed:
            continue
        new = [f for f in r.failures if (r.kind, _norm_fail(r.kind, f)) not in base_fail]
        for f in (new or r.failures)[:10]:
            led.issues.append(Issue(r.kind, f, ref, "high" if new else "low"))
        if not new:
            led.notes.append(f"{r.kind}: all {len(r.failures)} failure(s) already exist on the base checkout")


def apply_child_checks(led: Ledger, store: EvidenceStore, log: list[dict[str, Any]], final_failing: set[tuple[str, str]]) -> None:
    """Failures the build agent hit mid-run (and later fixed) are bugs hit along the way."""
    seen: set[tuple[str, str]] = set()
    for n, entry in enumerate(log, 1):
        for r in entry.get("results", []):
            if r.get("passed") or r.get("skipped"):
                continue
            ref = store.put(r["kind"], f"{r['kind']} after {entry.get('calls')} tool calls: {r.get('summary')}", r.get("output") or "\n".join(r.get("failures", [])), key=f"run{n}")
            for f in (r.get("failures") or [])[:5]:
                key = (r["kind"], _norm_fail(r["kind"], f))
                if key in seen or key in final_failing:
                    continue
                seen.add(key)
                led.issues.append(Issue(r["kind"], f"mid-build, after {entry.get('calls')} tool calls: {f}", ref, "low"))


def apply_event_issues(led: Ledger, store: EvidenceStore, events: list[dict[str, Any]]) -> None:
    errors = [e for e in events if e.get("status") == "error" or e.get("error")]
    blocked = [e for e in events if e.get("blocked")]
    for i, e in enumerate(blocked[:5], 1):
        ref = store.put("event", f"blocked {e.get('tool')}", json.dumps(e), key=f"b{i}")
        led.issues.append(Issue("runtime", f"blocked by Branchpoint: {e.get('blocked')}", ref, "medium"))
    for i, e in enumerate([e for e in errors if not e.get("blocked")][:8], 1):
        ref = store.put("event", f"{e.get('tool')} error", json.dumps(e), key=f"e{i}")
        led.issues.append(Issue("runtime", f"{e.get('tool')} {e.get('path') or ''}: {short(e.get('error') or 'error', 140)}", ref, "low"))
    writes: dict[str, int] = {}
    for e in events:
        if e.get("tool") in ("write_file", "patch") and e.get("path") and not e.get("blocked"):
            writes[e["path"]] = writes.get(e["path"], 0) + 1
    for path, n in writes.items():
        if n >= 3:
            ref = store.put("event", f"{path} rewritten {n}x", json.dumps([e for e in events if e.get("path") == path]), key=f"r-{abs(hash(path)) % 10_000}")
            led.issues.append(Issue("retry", f"{path} was rewritten {n} times", ref, "low"))


# --------------------------------------------------------------------------- baseline (route A)

def baseline_ledger(rc: RunContext, spec: RouteSpec, head_sha: str, stats: dict[str, Any]) -> Ledger:
    """Route A: what the user actually built, judged on the same ledger as everything else."""
    t0 = time.monotonic()
    rdir = rc.route_dir(spec.id)
    store = EvidenceStore(rdir, spec.id)
    led = _blank_ledger(rc, spec, "baseline")
    led.status, led.head_sha = "checking", head_sha
    save_ledger(rdir, led)
    publish(rc, led)
    wt = rc.worktree_path(spec.id)
    worktree_add(rc.repo, wt, head_sha)
    link_ignored(rc.repo, wt, rc.settings.get("link_dirs") or [])
    led.worktree = str(wt)
    apply_diff_facts(rc, led, store, head_sha, ["baseline route: the work already in your tree, measured from the transcript and diff"])
    results = run_checks(wt, rc.check_specs(wt))
    apply_check_issues(led, store, results, rc.base_checks())
    led.cost = Cost(tokens=int(stats.get("tokens") or 0), seconds=float(stats.get("seconds") or 0.0),
                    tool_calls=int(stats.get("tool_calls") or 0), estimated=True)
    if led.cost.seconds <= 0:
        led.cost.seconds = time.monotonic() - t0
    led.summary = short(spec.strategy.get("thesis") or "What you built.", 400)
    led.status = "done"
    save_ledger(rdir, led)
    publish(rc, led)
    return led


# --------------------------------------------------------------------------- execute

BUILD_PROMPT = """You are building ONE route of a multi-route design exploration (Branchpoint). Work only inside the
current directory, which is a private git worktree. You only have file tools: you cannot run commands.
Branchpoint runs the project's checks for you and appends their results to your tool outputs.

GOAL
{goal}
{followups}
{heading}
Axis: {axis}
Thesis: {thesis}
Key decisions:
{decisions}
Likely files: {files}
{plan_block}{guide}
RULES
- Read the code you need first, then make the smallest complete change that delivers this strategy.
- Follow the repo's existing test style; add or update tests so the new behaviour is covered.
- Do not touch unrelated files. Never write outside the current directory.
- Stop when the goal is met and checks look green. Finish with a 3-sentence summary: what you built,
  the main trade-off, and anything you left undone.
"""


def _stopped_building(result: Any, job: AgentJob) -> str:
    """Why this build did not finish, or ``""`` when it did.

    A build process that exits nonzero or is killed at its wall-clock cap has not delivered a route. Saying so here
    (and failing the ledger) is the difference between "the check suite happens to be green" and "an agent built
    this". ``rc is None`` is not a failure: some ``AgentResult`` producers leave the exit code unset, and an absent
    code is not evidence of a crash. An abort is reported separately by the caller, and a build that changed nothing
    is judged from the diff once it is measured.
    """
    if result.aborted:
        return ""
    if result.timed_out:
        return f"build agent hit the {job.wall_seconds}s wall-clock cap and was stopped"
    if result.rc not in (0, None):
        detail = short(result.error or result.final_text, 200)
        return f"build agent exited with code {result.rc}" + (f": {detail}" if detail else "")
    return ""


def execute_route(rc: RunContext, spec: RouteSpec) -> Ledger:
    t0 = time.monotonic()
    # Nothing is built until the child that would build it is proven to load this plugin's hooks: without them
    # nothing confines the build to this worktree and nothing measures it (see guard.py). This runs before the
    # route's worktree exists, so a refused run cannot have written a single build file.
    proof = rc.agent.confirm_hooks(rc.repo)
    rdir = rc.route_dir(spec.id)
    store = EvidenceStore(rdir, spec.id)
    led = _blank_ledger(rc, spec, "execute")
    seed = read_json(rdir / "ledger.sim.json")
    if seed:   # keep showing the simulation's prediction until the real numbers arrive
        sim = Ledger.from_dict(seed)
        for attr in ("files", "diff_stats", "blast", "deps_added", "cost", "reversibility", "reversibility_notes", "issues", "summary"):
            setattr(led, attr, getattr(sim, attr))
    led.status = "executing"
    save_ledger(rdir, led)
    publish(rc, led)
    wt = rc.worktree_path(spec.id)
    link_dirs = list(rc.settings.get("link_dirs") or [])
    try:
        worktree_add(rc.repo, wt, spec.start_sha or rc.base_sha, branch=rc.branch(spec.id))
        if spec.start_sha:      # keep the starting code in place but diff against the original base
            git(["reset", "--soft", rc.base_sha], wt)
        link_ignored(rc.repo, wt, link_dirs)
        led.worktree = str(wt)
        specs = rc.check_specs(wt)
        base_results = rc.base_checks()
        events_path = rdir / "tool_events.jsonl"
        checks_log = rdir / "checks_log.jsonl"
        handshake = rdir / "hook_handshake.json"
        nonce = secrets.token_hex(16)
        for p in (events_path, checks_log):
            p.write_text("", encoding="utf-8")
        marker = {
            "route": spec.id, "run_id": rc.run_id, "strategy": led.strategy, "goal": rc.goal,
            "confine_root": str(wt), "events_path": str(events_path), "checks_log": str(checks_log),
            "check_every": int(rc.settings.get("check_every") or 5),
            "max_tool_calls": int(rc.settings.get("route_max_tool_calls") or 60),
            "token_cap": child_token_cap(rc.route_token_cap, rc.meter.remaining()),
            "checks": [{"kind": s.kind, "argv": s.argv, "timeout": s.timeout} for s in specs],
            "base_failures": sorted({f"{r.kind}|{_norm_fail(r.kind, f)}" for r in base_results if not r.passed and not r.skipped for f in r.failures}),
            # The build child reports from ``pre_llm_call`` (before it can call a tool) that these hooks are
            # live and that its confinement decision refuses a path outside this worktree. Checked below.
            "handshake": {"path": str(handshake), "nonce": nonce},
            "probe_escape": str(rc.run_dir / f"outside-worktree-{spec.id}.txt"),
        }
        write_marker(wt, marker)
        s = spec.strategy
        own = bool(s.get("independent"))
        heading = (f"ROUTE {spec.id} — YOU planned this approach yourself; build it as planned and do not switch to a different approach" if own
                   else f"ROUTE {spec.id} STRATEGY — a hard constraint; do not drift to a different approach")
        plan_block = f"\nYOUR PLAN\n{s['plan_md']}\n" if own and s.get("plan_md") else ""
        prompt = BUILD_PROMPT.format(
            heading=heading, plan_block=plan_block, guide=(("\n" + spec.guide + "\n") if spec.guide else ""),
            goal=redact_for_model(rc.goal), rid=spec.id, axis=s.get("axis", ""), thesis=s.get("thesis", ""),
            decisions="\n".join(f"- {d}" for d in s.get("key_decisions") or []) or "- (none specified)",
            files=", ".join(s.get("predicted_files") or []) or "(your call)",
            followups=("\nLATER REQUESTS TO ALSO SATISFY\n" + "\n".join(f"- {short(redact_for_model(f), 240)}" for f in rc.followups[:5]) + "\n") if rc.followups else "")
        job = AgentJob(spec.id, wt, prompt, model=spec.model or rc.settings.get("child_model") or None,
                       max_turns=int(rc.settings.get("route_max_turns") or 40),
                       wall_seconds=int(rc.settings.get("route_seconds") or 900), log_path=rdir / "agent.log",
                       child_rules=bool(rc.settings.get("child_rules")))
        rc.bus.emit("route_log", route=spec.id, message=f"build agent started in {wt.name} (file tools only)")
        result = rc.agent.run(job, rc.abort)
        stopped = _stopped_building(result, job)
        if not stopped and proof.is_child_process:
            # Defence in depth: the child was proven before the build started (guard.py), so this only fires if
            # the hooks died between the proof and the build. A build nobody measured is not a finished route.
            try:
                verify_handshake(handshake, nonce=nonce, route=spec.id)
            except HooksNotConfirmed as exc:
                stopped = f"the build child did not prove this plugin's hooks ({short(exc, 180)}), so the build was not confined or measured"
        if result.aborted:
            led.status, led.summary = "aborted", "aborted by the user"
        led.status = "checking" if led.status == "executing" else led.status
        publish(rc, led)
        head = commit_worktree(wt, f"branchpoint route {spec.id}: {s.get('name', '')}", link_dirs)
        led.head_sha = head
        led.issues = []     # the simulation's predicted risks stay in ledger.sim.json; from here on only what actually happened
        git(["update-ref", f"refs/branchpoint/{rc.run_id}/{spec.id}", head], rc.repo, check=False)
        apply_diff_facts(rc, led, store, head)
        if not stopped and not result.aborted and led.diff_stats.get("files", 0) == 0:
            stopped = "build agent finished without changing any file: the route delivered nothing"
        stop_ref = ""
        if stopped:         # a build that did not finish is not a finished route, whatever its checks say
            led.notes.append(stopped)
            stop_ref = store.put("runtime", "the build did not finish", f"{stopped}\n\n{tail(result.error or result.final_text, 1500)}", key="stop")
            led.issues.append(Issue("runtime", stopped, stop_ref, "high"))
        results = run_checks(wt, specs)
        apply_check_issues(led, store, results, base_results)
        events = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()] if events_path.exists() else []
        clog = [json.loads(l) for l in checks_log.read_text(encoding="utf-8").splitlines() if l.strip()] if checks_log.exists() else []
        final_failing = {(r.kind, _norm_fail(r.kind, f)) for r in results if not r.passed and not r.skipped for f in r.failures}
        apply_child_checks(led, store, clog, final_failing)
        apply_event_issues(led, store, events)
        tokens = sum(int(e.get("tokens") or 0) for e in events if e.get("type") == "usage") or 0
        estimated = not tokens
        if estimated:
            tokens = max(1, len(result.final_text) // 4) + 1500 * len([e for e in events if e.get("tool")])
            led.notes.append("tokens estimated: the child process did not report usage")
        led.cost = Cost(tokens=tokens, seconds=time.monotonic() - t0, tool_calls=len([e for e in events if e.get("tool")]),
                        estimated=estimated)
        stat_ref = store.put("stat", "Cost measured for this route", json.dumps(asdict(led.cost)), key="cost")
        _ = stat_ref
        if rc.route_token_cap and tokens > rc.route_token_cap:
            led.notes.append(f"over its token share ({tokens} > {rc.route_token_cap})")
        failing = [i for i in led.issues if i.kind in ("test", "lint", "type") and i.severity == "high"]
        if stopped:
            led.summary = short(f"stopped: {stopped}", 420)
        else:
            led.summary = short(result.final_text, 420) or (
                f"{led.diff_stats['files']} file(s), +{led.diff_stats['insertions']} -{led.diff_stats['deletions']}; "
                + ("checks failing" if failing else "checks green"))
        if led.status == "checking":
            led.status = "failed" if stopped else "done"
        first_fail = failing[0].detail if failing else ""
        if stopped:
            _log(rc, "build", f"Route {spec.id} did not finish",
                 f"Route {spec.id} ({s.get('name', '')}) did not finish: {short(stopped, 200)}",
                 f"{len(failing)} check(s) failed beside it." if failing else "Its checks below are not evidence of a build.",
                 route=spec.id, thinking=short(result.final_text, 600), refs=[stop_ref] if stop_ref else [],
                 data={"failing": True, "stopped": True, "tokens": tokens, "files": led.diff_stats["files"]})
        else:
            _log(rc, "build", f"Route {spec.id} built",
                 f"Route {spec.id} ({s.get('name', '')}) built: {led.diff_stats['files']} file(s) +{led.diff_stats['insertions']} -{led.diff_stats['deletions']}; "
                 + ("checks failing." if failing else "checks passing."),
                 (f"{len(failing)} check(s) failed, first: {short(first_fail, 160)}" if failing else
                  f"Every check the project has passed. Impact {led.blast.score}/10 ({led.blast.dependents} dependent file(s))."),
                 route=spec.id, thinking=short(result.final_text, 600), refs=[f"{spec.id}:stat:diff", f"{spec.id}:trace:blast"] + ([failing[0].evidence_ref] if failing else []),
                 data={"failing": bool(failing), "tokens": tokens, "files": led.diff_stats["files"]})
    except ModelOverrideRefused:
        raise                      # a refused model is final for the run; "execution failed" would hide the cause
    except BudgetExceeded as exc:
        led.status, led.summary = "failed", f"stopped: {exc}"
        _log(rc, "build", f"Route {spec.id} stopped", f"Route {spec.id} was stopped before it finished.", f"The token budget ran out: {exc}", route=spec.id, data={"failing": True})
    except Exception as exc:
        led.status = "failed"
        led.summary = f"execution failed: {short(exc, 240)}"
        led.notes.append(tail(str(exc), 600))
        _log(rc, "build", f"Route {spec.id} failed", f"Route {spec.id} could not be built.", short(exc, 240), route=spec.id, data={"failing": True})
    save_ledger(rdir, led)
    publish(rc, led)
    return led
