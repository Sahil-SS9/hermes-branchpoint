"""Run one route: simulate it (plan + static analysis) or execute it (real build in its own worktree).

Both paths end in the same place: a complete ``Ledger`` whose every claim has an ``evidence_ref``.
"""
from __future__ import annotations

import json
import os
import re
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
from .ledger import (Cost, EvidenceStore, Issue, Ledger, reversibility_from_facts, save_ledger)
from .llm import LLM, BudgetExceeded, Meter, redact_for_model
from .protocol import write_marker
from .util import (GIT_IDENTITY, BranchpointError, extract_json, git, short, tail)

SIM_SYSTEM = (
    "You are a senior engineer producing an implementation plan for ONE specific route of a design "
    "exploration. Do not write the code. Be concrete (real file paths and symbols from the repo map), "
    "honest about risks, and do not drift to a different approach than the route's strategy. Reply with JSON only."
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


@dataclass
class RouteSpec:
    id: str
    strategy: dict[str, Any]
    model: str | None = None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        worktree_remove(repo, path)
    args = ["worktree", "add", "--quiet"]
    args += ["-b", branch, str(path), rev] if branch else ["--detach", str(path), rev]
    git(args, repo, timeout=180)


def worktree_remove(repo: Path, path: Path) -> None:
    git(["worktree", "remove", "--force", str(path)], repo, check=False, timeout=120)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    git(["worktree", "prune"], repo, check=False)


def delete_branch(repo: Path, branch: str) -> None:
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
        reversibility=led.reversibility, summary=short(led.summary, 240), **extra)


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
    return {
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
    route_meter = rc.meter.child(f"route {spec.id}", rc.route_token_cap)
    llm = rc.llm.with_meter(route_meter) if hasattr(rc.llm, "with_meter") else rc.llm  # type: ignore[attr-defined]
    try:
        graph = rc.graph()
        pf = [p for p in spec.strategy.get("predicted_files") or []]
        existing = [p for p in pf if p.lstrip("./") in graph.files]
        excerpt_paths = existing or sorted(f for f in graph.files if f.endswith((".py", ".js", ".ts")) and not blastmod.is_test_path(f))[:4]
        user = "\n\n".join([
            f"GOAL:\n{redact_for_model(rc.goal)}",
            f"ROUTE {spec.id} STRATEGY (axis: {spec.strategy.get('axis')}):\n{spec.strategy.get('thesis')}\nKey decisions:\n- " + "\n- ".join(spec.strategy.get("key_decisions") or []),
            f"REPO MAP:\n{repo_map(rc.base_root, 5500)}",
            f"RELEVANT FILES:\n{redact_for_model(read_excerpts(rc.base_root, excerpt_paths, 9000))}",
            f"Produce the implementation plan for this route only. Reply with JSON only: {SIM_SCHEMA}",
        ])
        reply = llm.complete(SIM_SYSTEM, user, max_tokens=4000, model=spec.model, purpose=f"simulate:{spec.id}")
        plan = _coerce_plan(extract_json(reply.text))
        if not plan["files"]:
            raise BranchpointError("the plan listed no files")
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
    led.notes.append(f"plan produced with {route_meter.used} tokens in {time.monotonic() - t0:.0f}s (not part of the estimated build cost)")
    led.status = "done"
    save_ledger(rdir, led)
    publish(rc, led, planning_tokens=route_meter.used)
    return led


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
    base_fail = {(r.kind, _norm_fail(r.kind, f)) for r in base_results if not r.passed for f in r.failures}
    for r in results:
        ref = store.put(r.kind, f"{r.kind}: {r.cmd} → {r.summary}", f"$ {r.cmd}\n(exit {r.rc}, {r.seconds:.1f}s) {r.summary}\n\n{r.output}", key="final")
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
            if r.get("passed"):
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
ROUTE {rid} STRATEGY — a hard constraint; do not drift to a different approach
Axis: {axis}
Thesis: {thesis}
Key decisions:
{decisions}
Likely files: {files}

RULES
- Read the code you need first, then make the smallest complete change that delivers this strategy.
- Follow the repo's existing test style; add or update tests so the new behaviour is covered.
- Do not touch unrelated files. Never write outside the current directory.
- Stop when the goal is met and checks look green. Finish with a 3-sentence summary: what you built,
  the main trade-off, and anything you left undone.
"""


def execute_route(rc: RunContext, spec: RouteSpec) -> Ledger:
    t0 = time.monotonic()
    rdir = rc.route_dir(spec.id)
    store = EvidenceStore(rdir, spec.id)
    led = _blank_ledger(rc, spec, "execute")
    led.status = "executing"
    save_ledger(rdir, led)
    publish(rc, led)
    wt = rc.worktree_path(spec.id)
    link_dirs = list(rc.settings.get("link_dirs") or [])
    try:
        worktree_add(rc.repo, wt, rc.base_sha, branch=rc.branch(spec.id))
        link_ignored(rc.repo, wt, link_dirs)
        led.worktree = str(wt)
        specs = rc.check_specs(wt)
        base_results = rc.base_checks()
        events_path = rdir / "tool_events.jsonl"
        checks_log = rdir / "checks_log.jsonl"
        for p in (events_path, checks_log):
            p.write_text("", encoding="utf-8")
        marker = {
            "route": spec.id, "run_id": rc.run_id, "strategy": led.strategy, "goal": rc.goal,
            "confine_root": str(wt), "events_path": str(events_path), "checks_log": str(checks_log),
            "check_every": int(rc.settings.get("check_every") or 5),
            "max_tool_calls": int(rc.settings.get("route_max_tool_calls") or 60),
            "token_cap": rc.route_token_cap, "checks": [{"kind": s.kind, "argv": s.argv, "timeout": s.timeout} for s in specs],
            "base_failures": sorted({f"{r.kind}|{_norm_fail(r.kind, f)}" for r in base_results if not r.passed for f in r.failures}),
        }
        write_marker(wt, marker)
        s = spec.strategy
        prompt = BUILD_PROMPT.format(
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
        if result.aborted:
            led.status, led.summary = "aborted", "aborted by the user"
        if result.timed_out:
            led.notes.append(f"build agent hit the {job.wall_seconds}s wall-clock cap and was stopped")
        if result.rc not in (0, None) and not (result.timed_out or result.aborted):
            led.notes.append(f"build agent exited with code {result.rc}: {short(result.error or result.final_text, 200)}")
        led.status = "checking" if led.status == "executing" else led.status
        publish(rc, led)
        head = commit_worktree(wt, f"branchpoint route {spec.id}: {s.get('name', '')}", link_dirs)
        led.head_sha = head
        git(["update-ref", f"refs/branchpoint/{rc.run_id}/{spec.id}", head], rc.repo, check=False)
        apply_diff_facts(rc, led, store, head)
        results = run_checks(wt, specs)
        apply_check_issues(led, store, results, base_results)
        events = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()] if events_path.exists() else []
        clog = [json.loads(l) for l in checks_log.read_text(encoding="utf-8").splitlines() if l.strip()] if checks_log.exists() else []
        final_failing = {(r.kind, _norm_fail(r.kind, f)) for r in results if not r.passed for f in r.failures}
        apply_child_checks(led, store, clog, final_failing)
        apply_event_issues(led, store, events)
        tokens = sum(int(e.get("tokens") or 0) for e in events if e.get("type") == "usage") or 0
        if not tokens:
            tokens = max(1, len(result.final_text) // 4) + 1500 * len([e for e in events if e.get("tool")])
            led.notes.append("tokens estimated: the child process did not report usage")
        led.cost = Cost(tokens=tokens, seconds=time.monotonic() - t0, tool_calls=len([e for e in events if e.get("tool")]),
                        estimated=False)
        stat_ref = store.put("stat", "Cost measured for this route", json.dumps(asdict(led.cost)), key="cost")
        _ = stat_ref
        if rc.route_token_cap and tokens > rc.route_token_cap:
            led.notes.append(f"over its token share ({tokens} > {rc.route_token_cap})")
        failing = [i for i in led.issues if i.kind in ("test", "lint", "type") and i.severity == "high"]
        led.summary = short(result.final_text, 420) or (
            f"{led.diff_stats['files']} file(s), +{led.diff_stats['insertions']} -{led.diff_stats['deletions']}; "
            + ("checks failing" if failing else "checks green"))
        if led.status == "checking":
            led.status = "done"
    except BudgetExceeded as exc:
        led.status, led.summary = "failed", f"stopped: {exc}"
    except Exception as exc:
        led.status = "failed"
        led.summary = f"execution failed: {short(exc, 240)}"
        led.notes.append(tail(str(exc), 600))
    save_ledger(rdir, led)
    publish(rc, led)
    return led
