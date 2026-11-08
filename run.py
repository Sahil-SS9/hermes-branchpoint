"""A Branchpoint run: anchor -> diverge -> simulate -> execute top-k -> council.

State lives on disk (``run.json`` + per-route ledgers) so ``status``, ``council`` and ``adopt`` work from
a later command, a different process, or after a restart.
"""
from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import council as councilmod
from . import diverge as divergemod
from .adopt import cleanup_run
from .agent import AgentRunner, HermesAgentRunner
from .anchor import Anchor
from .events import EventBus
from .ledger import Ledger, RunEvidence, load_ledger, save_ledger
from .llm import LLM, BudgetExceeded, Meter
from .route import (RouteSpec, RunContext, baseline_ledger, diff_numstat, execute_route, link_ignored,
                    publish, rank_for_execution, simulate_route, worktree_add)
from .util import BranchpointError, git, iso, now, read_json, short, write_json

LETTERS = "ABCDEFGH"
DEFAULT_SETTINGS: dict[str, Any] = {
    "n": 4, "execute": "top2", "budget_tokens": 600_000, "max_parallel": 3, "route_seconds": 900, "route_max_turns": 40,
    "route_max_tool_calls": 60, "check_every": 5, "diversity_threshold": divergemod.DEFAULT_THRESHOLD,
    "diversity_llm_check": True, "link_dirs": [".venv", "venv", "node_modules"], "child_model": "", "child_rules": False,
    "test_command": "", "lint_command": "", "type_command": "", "models": [],
}


def new_run_id() -> str:
    return f"{time.strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class RunPaths:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def run_dir(self, run_id: str) -> Path:
        return self.data_dir / "runs" / run_id

    def wt_root(self, run_id: str) -> Path:
        return self.data_dir / "worktrees" / run_id


def list_runs(data_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((Path(data_dir) / "runs").glob("*/run.json"), reverse=True):
        meta = read_json(p)
        if meta:
            out.append(meta)
    return out


def exec_count(policy: str, n_alt: int) -> int:
    p = str(policy).strip().lower()
    if p in ("none", "0", "off", "simulate"):
        return 0
    if p == "all":
        return n_alt
    if p.startswith("top") and p[3:].isdigit():
        return min(int(p[3:]), n_alt)
    if p.isdigit():
        return min(int(p), n_alt)
    return min(2, n_alt)


class Run:
    def __init__(self, paths: RunPaths, run_id: str, meta: dict[str, Any], llm: LLM | None = None,
                 agent: AgentRunner | None = None, redact: Callable[[str], str] | None = None):
        self.paths = paths
        self.run_id = run_id
        self.meta = meta
        self.dir = paths.run_dir(run_id)
        self.wt_root = paths.wt_root(run_id)
        self.bus = EventBus(self.dir / "events.jsonl")
        self.abort = threading.Event()
        self.agent: AgentRunner = agent or HermesAgentRunner()
        self.meter = Meter(int(meta["settings"].get("budget_tokens") or 0) or None, label="run budget")
        self.meter.used = int(meta.get("tokens_used") or 0)
        self.llm = llm.with_meter(self.meter) if (llm is not None and hasattr(llm, "with_meter")) else llm
        self.rc: RunContext | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, data_dir: Path, *, anchor: Anchor, mode: str, goal: str, settings: dict[str, Any], llm: LLM,
               agent: AgentRunner | None = None, baseline_stats: dict[str, Any] | None = None) -> "Run":
        paths = RunPaths(data_dir)
        run_id = new_run_id()
        s = {**DEFAULT_SETTINGS, **{k: v for k, v in settings.items() if v is not None}}
        meta = {
            "run_id": run_id, "mode": mode, "goal": goal, "repo": anchor.repo, "base_sha": anchor.base_sha,
            "head_sha": anchor.head_sha, "anchor_source": anchor.source, "prompt_n": anchor.prompt_n,
            "session_id": anchor.session_id, "has_baseline": bool(anchor.has_baseline and mode == "rewind"),
            "followups": anchor.followups, "context": anchor.context, "anchor_notes": anchor.notes,
            "settings": s, "status": "created", "created": now(), "created_iso": iso(), "routes": [], "strategies": [],
            "baseline_stats": baseline_stats or {}, "tokens_used": 0, "warnings": [], "winner": None, "adopted": None,
            "view_url": "",
        }
        run = cls(paths, run_id, meta, llm, agent)
        run.dir.mkdir(parents=True, exist_ok=True)
        run.save()
        return run

    @classmethod
    def load(cls, data_dir: Path, run_id: str, llm: LLM | None = None, agent: AgentRunner | None = None) -> "Run":
        paths = RunPaths(data_dir)
        meta = read_json(paths.run_dir(run_id) / "run.json")
        if not meta:
            raise BranchpointError(f"No Branchpoint run called '{run_id}'. Try /branchpoint list.")
        return cls(paths, run_id, meta, llm, agent)

    def save(self) -> None:
        with self.lock:
            self.meta["tokens_used"] = self.meter.used
            write_json(self.dir / "run.json", self.meta)

    def set_status(self, status: str, **extra: Any) -> None:
        self.meta["status"] = status
        self.meta.update(extra)
        self.save()
        self.bus.emit("status", status=status, **extra)

    @property
    def repo(self) -> Path:
        return Path(self.meta["repo"])

    @property
    def settings(self) -> dict[str, Any]:
        return self.meta["settings"]

    def ledgers(self) -> list[Ledger]:
        out = []
        for rid in self.meta.get("routes", []):
            led = load_ledger(self.dir / "routes" / rid)
            if led:
                out.append(led)
        return out

    def context(self) -> RunContext:
        if self.rc is None:
            assert self.llm is not None, "LLM backend required"
            strategies = self.meta.get("strategies") or []
            n_alt = max(1, len([s for s in strategies if not s.get("baseline")]))
            cap = self.settings.get("budget_tokens")
            share = int(cap * 0.7 / n_alt) if (cap and strategies) else None
            self.rc = RunContext(
                run_id=self.run_id, repo=self.repo, run_dir=self.dir, wt_root=self.wt_root, base_sha=self.meta["base_sha"],
                goal=self.meta["goal"], bus=self.bus, abort=self.abort, llm=self.llm, agent=self.agent, meter=self.meter,
                settings=self.settings, followups=self.meta.get("followups") or [], context=self.meta.get("context") or "",
                route_token_cap=share)
        return self.rc

    # ------------------------------------------------------------------ phase 1: anchor + diverge (synchronous)
    def prepare(self) -> list[dict[str, Any]]:
        """Create the base worktree and generate distinct strategies. Fast; the caller prints the result."""
        self.set_status("diverging")
        base_wt = self.wt_root / "base"
        worktree_add(self.repo, base_wt, self.meta["base_sha"])
        link_ignored(self.repo, base_wt, self.settings.get("link_dirs") or [])
        git(["update-ref", f"refs/branchpoint/{self.run_id}/base", self.meta["base_sha"]], self.repo)
        if self.meta["has_baseline"]:
            git(["update-ref", f"refs/branchpoint/{self.run_id}/A", self.meta["head_sha"]], self.repo)
        rc = self.context()
        baseline = None
        diff_stat = ""
        if self.meta["has_baseline"]:
            files, ins, dels = diff_numstat(self.repo, self.meta["base_sha"], self.meta["head_sha"])
            diff_stat = "\n".join(f"{f['path']} +{f['insertions']} -{f['deletions']}" for f in files)
            diff_text = git(["diff", "-U2", "--no-color", self.meta["base_sha"], self.meta["head_sha"]], self.repo, check=False)
            try:
                baseline = divergemod.describe_baseline(self.llm, self.meta["goal"], diff_text, diff_stat)  # type: ignore[arg-type]
            except Exception as exc:
                self.meta["warnings"].append(f"could not describe the baseline route ({short(exc, 100)}); using a generic description")
                baseline = {"axis": "as-built", "name": "What you built", "thesis": f"The implementation already in the working tree ({len(files)} file(s) changed).",
                            "key_decisions": [], "predicted_files": [f["path"] for f in files][:10], "trade_off": ""}
        res = divergemod.generate_strategies(
            self.llm, goal=self.meta["goal"], n=int(self.settings["n"]), context=self.meta.get("context") or "",  # type: ignore[arg-type]
            repo_map_text=divergemod.repo_map(base_wt), baseline=baseline, followups=self.meta.get("followups") or None,
            threshold=float(self.settings["diversity_threshold"]), llm_check=bool(self.settings["diversity_llm_check"]))
        letters = list(LETTERS)
        strategies: list[dict[str, Any]] = []
        if baseline:
            strategies.append({**baseline, "id": "A", "baseline": True})
            letters.remove("A")
        for s in res.strategies:
            strategies.append({**s, "id": letters.pop(0), "baseline": False})
        models = list(self.settings.get("models") or [])
        for i, s in enumerate([s for s in strategies if not s["baseline"]]):
            if models:
                s["model"] = models[i % len(models)]
        self.meta["strategies"] = strategies
        self.meta["routes"] = [s["id"] for s in strategies]
        self.meta["warnings"] += res.warnings
        self.meta["rejected"] = res.rejected
        self.meta["similarity"] = res.similarity
        for s in strategies:
            rdir = rc.route_dir(s["id"])
            led = Ledger(route=s["id"], strategy=s, fidelity="baseline" if s["baseline"] else "simulate", status="queued", base_sha=self.meta["base_sha"])
            save_ledger(rdir, led)
        cap = self.settings.get("budget_tokens")
        n_alt = max(1, len([s for s in strategies if not s["baseline"]]))
        rc.route_token_cap = int(cap * 0.7 / n_alt) if cap else None   # 30% stays in reserve for the council
        self.save()
        self.bus.emit("strategies", strategies=strategies, rejected=res.rejected, warnings=res.warnings,
                      anchor={"base": self.meta["base_sha"][:10], "source": self.meta["anchor_source"], "mode": self.meta["mode"],
                              "prompt_n": self.meta["prompt_n"], "goal": self.meta["goal"]})
        for s in strategies:
            led = load_ledger(rc.route_dir(s["id"]))
            if led:
                publish(rc, led)
        self.set_status("planned")
        return strategies

    # ------------------------------------------------------------------ phase 2..5 (long; runs in a thread)
    def go(self, background: bool = True) -> None:
        if background:
            self.thread = threading.Thread(target=self._pipeline, name=f"branchpoint-{self.run_id}", daemon=True)
            self.thread.start()
        else:
            self._pipeline()

    def join(self, timeout: float | None = None) -> None:
        if self.thread:
            self.thread.join(timeout)

    def _pipeline(self) -> None:
        rc = self.context()
        try:
            strategies = self.meta["strategies"]
            alts = [s for s in strategies if not s["baseline"]]
            pool = ThreadPoolExecutor(max_workers=int(self.settings["max_parallel"]), thread_name_prefix="bp-route")
            # baseline (already built) ledger
            futures = []
            if self.meta["has_baseline"]:
                base_spec = next(s for s in strategies if s["baseline"])
                futures.append(pool.submit(baseline_ledger, rc, RouteSpec("A", base_spec), self.meta["head_sha"], self.meta.get("baseline_stats") or {}))
            self.set_status("simulating")
            sims = [pool.submit(simulate_route, rc, RouteSpec(s["id"], s, s.get("model"))) for s in alts]
            sim_ledgers = [f.result() for f in sims]
            for f in futures:
                f.result()
            self.save()
            if self.abort.is_set():
                return self._finish_aborted()
            k = exec_count(self.settings["execute"], len(alts))
            ranked = rank_for_execution(sim_ledgers)
            chosen = [l.route for l in ranked[:k]]
            self.meta["executed"] = chosen
            self.save()
            if chosen:
                self.set_status("executing", executing=chosen)
                jobs = []
                for rid in chosen:
                    spec = next(s for s in alts if s["id"] == rid)
                    jobs.append(pool.submit(self._execute_one, RouteSpec(rid, spec, spec.get("model"))))
                for j in jobs:
                    j.result()
            pool.shutdown(wait=True)
            if self.abort.is_set():
                return self._finish_aborted()
            self.council()
        except BudgetExceeded as exc:
            self.meta["warnings"].append(str(exc))
            self.set_status("failed", error=str(exc))
        except Exception as exc:  # pragma: no cover - last-resort guard
            self.meta["warnings"].append(f"run failed: {short(exc, 200)}")
            self.set_status("failed", error=str(exc))

    def _finish_aborted(self) -> None:
        self.set_status("aborted")

    def _execute_one(self, spec: RouteSpec) -> Ledger:
        rc = self.context()
        rdir = rc.route_dir(spec.id)
        prior = load_ledger(rdir)
        try:
            self.meter.check()
        except BudgetExceeded as exc:
            if prior:
                prior.notes.append(f"not executed: {exc}")
                save_ledger(rdir, prior)
                publish(rc, prior)
            return prior  # type: ignore[return-value]
        if prior and prior.status == "done":
            (rdir / "ledger.sim.json").write_text((rdir / "ledger.json").read_text(encoding="utf-8"), encoding="utf-8")
        led = execute_route(rc, spec)
        self.meter.charge(led.cost.tokens)
        if prior and prior.fidelity == "simulate" and prior.status == "done":
            led.notes.append(
                f"simulation predicted blast {prior.blast.score}/10 and {prior.diff_stats.get('files')} file(s); actual: "
                f"{led.blast.score}/10 and {led.diff_stats.get('files')} file(s)")
            save_ledger(rdir, led)
            publish(rc, led)
        self.save()
        return led

    # ------------------------------------------------------------------ council
    def council(self) -> "councilmod.CouncilResult":
        assert self.llm is not None
        self.set_status("council")
        ledgers = self.ledgers()
        rev = RunEvidence(self.dir)
        res = councilmod.run_council(self.llm, ledgers, rev, goal=self.meta["goal"], run_id=self.run_id,
                                     mode=self._mode_label(), run_dir=self.dir)
        self.meta["winner"] = res.winner
        self.meta["runner_up"] = res.runner_up
        self.meta["warnings"] += [w for w in res.warnings if w not in self.meta["warnings"]]
        self.save()
        self.bus.emit("decision", winner=res.winner, runner_up=res.runner_up, ranking=res.ranking, confidence=res.narrative.get("confidence"),
                      warnings=res.warnings)
        self.set_status("decided")
        return res

    def _mode_label(self) -> str:
        if self.meta["mode"] == "rewind":
            return f"retrospective, rewound to prompt {self.meta.get('prompt_n')}"
        return "prospective"

    # ------------------------------------------------------------------ housekeeping
    def cleanup(self, keep: bool = False, keep_routes: list[str] | None = None) -> list[str]:
        return cleanup_run(self.repo, self.dir, self.run_id, self.wt_root, list(self.meta.get("routes", [])), keep, keep_routes)

    def drop_refs(self) -> None:
        res = git(["for-each-ref", "--format=%(refname)", f"refs/branchpoint/{self.run_id}/"], self.repo, check=False)
        for ref in res.split():
            if not ref.endswith("/pre-adopt"):
                git(["update-ref", "-d", ref], self.repo, check=False)

    def abort_run(self) -> None:
        self.abort.set()
        self.bus.emit("log", message="abort requested")


# --------------------------------------------------------------------------- text status

_ICON = {"queued": "·", "planning": "…", "simulating": "…", "executing": "▶", "checking": "…", "done": "✓", "failed": "✗", "aborted": "■", "skipped": "–"}


def render_status(run: "Run") -> str:
    m = run.meta
    cap = m["settings"].get("budget_tokens")
    head = [f"Branchpoint run {m['run_id']}  —  {m['mode']}" + (f" at prompt {m['prompt_n']}" if m.get("prompt_n") else "") + f"  [{m['status']}]",
            f"Goal: {short(m['goal'], 120)}",
            f"Base: {m['base_sha'][:10]} ({m['anchor_source']})   Budget: {run.meter.used // 1000}k / {(cap or 0) // 1000}k tokens"]
    if m.get("view_url"):
        head.append(f"Live view: {m['view_url']}")
    lines = head + [""]
    leds = {l.route: l for l in run.ledgers()}
    ids = list(m.get("routes", []))
    winner = m.get("winner")
    for i, rid in enumerate(ids):
        l = leds.get(rid)
        branch = "└─" if i == len(ids) - 1 else "├─"
        if not l:
            lines.append(f"{branch} {rid}  (pending)")
            continue
        c = l.cost
        crown = "  ★ winner" if rid == winner else ""
        lines.append(f"{branch} {rid} {_ICON.get(l.status, '?')} {short(l.strategy.get('name'), 34):<34} {l.fidelity:<8} {l.status:<10} blast {l.blast.score}/10  "
                     f"issues {len(l.issues)}  {c.tokens // 1000}k tok {c.seconds:.0f}s{' est.' if c.estimated else ''}{crown}")
    if winner:
        lines += ["", f"Decision: Route {winner} — see {run.dir / 'decision.md'}", f"Next: /branchpoint adopt {winner} [merge|replay|adr]"]
    elif m["status"] == "planned":
        lines += ["", "Routes are queued; building starts automatically."]
    for w in m.get("warnings", [])[:4]:
        lines.append(f"! {w}")
    return "\n".join(lines)
