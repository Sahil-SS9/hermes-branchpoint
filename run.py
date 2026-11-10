"""A Branchpoint run: anchor -> diverge -> simulate -> execute top-k -> council.

State lives on disk (``run.json`` + per-route ledgers) so ``status``, ``council`` and ``adopt`` work from
a later command, a different process, or after a restart.
"""
from __future__ import annotations

import secrets
import shutil
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
from .declog import DecisionLog
from .events import EventBus
from .ledger import Ledger, RunEvidence, load_ledger, save_ledger
from .llm import LLM, BudgetExceeded, Meter, ModelOverrideRefused
from .route import (RouteSpec, RunContext, baseline_ledger, delete_branch, diff_numstat, execute_route,
                    link_ignored, publish, rank_for_execution, simulate_route, worktree_add, worktree_remove)
from .util import BranchpointError, git, iso, now, read_json, short, write_json

LETTERS = "ABCDEFGH"
DEFAULT_SETTINGS: dict[str, Any] = {
    "n": 4, "execute": "top2", "budget_tokens": 600_000, "max_parallel": 3, "route_seconds": 900, "route_max_turns": 40,
    "route_max_tool_calls": 60, "check_every": 5, "diversity_threshold": divergemod.DEFAULT_THRESHOLD,
    "diversity_llm_check": True, "link_dirs": [".venv", "venv", "node_modules"], "child_model": "", "child_rules": False,
    "test_command": "", "lint_command": "", "type_command": "", "models": [],
    # "independent": every agent gets only the task, plans and builds its own way. "assigned": strategies are handed out on distinct axes.
    "approach": "independent", "judge_model": "", "reviewer_model": "", "converge_tokens": 150_000,
}
MIN_ROUTES, MAX_ROUTES = 2, 8


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
        # One run gets one refusal record: a model the host refused last time is asked again (the grant may have
        # changed), while every route and the council inside this run share the record and stay fail-closed.
        self.llm: LLM | None = llm
        if llm is not None:
            start_run = getattr(llm, "for_run", None)
            if callable(start_run):
                self.llm = start_run(self.meter)  # type: ignore[assignment]
            elif hasattr(llm, "with_meter"):
                self.llm = llm.with_meter(self.meter)  # type: ignore[attr-defined]
        self.rc: RunContext | None = None
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.declog = DecisionLog(self.dir)

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
                route_token_cap=share, declog=self.declog)
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
            except ModelOverrideRefused:
                raise            # a refused model is not a missing description: stop rather than substitute one
            except Exception as exc:
                self.meta["warnings"].append(f"could not describe the baseline route ({short(exc, 100)}); using a generic description")
                baseline = {"axis": "as-built", "name": "What you built", "thesis": f"The implementation already in the working tree ({len(files)} file(s) changed).",
                            "key_decisions": [], "predicted_files": [f["path"] for f in files][:10], "trade_off": ""}
        independent = str(self.settings.get("approach") or "independent") == "independent"
        notes = self.meta.get("anchor_notes") or []
        self.declog.record(
            "anchor", "The code every route starts from",
            (f"Rewound to prompt {self.meta.get('prompt_n')}; code state taken from {self.meta['anchor_source']} ({self.meta['base_sha'][:10]})."
             if self.meta["mode"] == "rewind" else f"Started from the current code ({self.meta['base_sha'][:10]}); nothing built yet."),
            (str(notes[0]) if notes else "That is the exact code state at that point, so every route starts from the same place.")
            + (" What you built since then stays as route A, the baseline." if self.meta["has_baseline"] else ""),
            data={"base": self.meta["base_sha"][:10], "source": self.meta["anchor_source"]})
        if independent:
            want = int(self.settings["n"]) - (1 if baseline else 0)
            res = divergemod.DivergeResult([{"axis": "independent", "name": "Agent", "thesis": "", "key_decisions": [], "predicted_files": [],
                                             "trade_off": "", "independent": True} for _ in range(max(1, want))])
        else:
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
        for s in strategies:
            if s.get("independent"):
                s["name"] = f"Agent {s['id']}"
        alts_now = [s for s in strategies if not s["baseline"]]
        used_models = [s.get("model") or self.settings.get("child_model") or "your default model" for s in alts_now]
        jm = self.settings.get("judge_model") or "the council model (or your default)"
        rm = self.settings.get("reviewer_model") or self.settings.get("judge_model") or "the council model (or your default)"
        self.declog.record(
            "agents", "Agents, models and judge",
            f"{len(alts_now)} agent{'s' if len(alts_now) != 1 else ''} ({', '.join(s['id'] for s in alts_now)}) " + (
                "were each given the same task and told nothing else: each plans and builds it its own way." if independent else
                "were each assigned a different direction to build."),
            (("Models: " + ", ".join(s["id"] + " = " + m for s, m in zip(alts_now, used_models)) + ". ") if alts_now else "")
            + f"Blind reviewers use {rm}; the judge uses {jm}. " + ("You set the models." if (self.settings.get("models") or self.settings.get("judge_model")) else "No models were set, so the defaults are used."),
            data={"n": len(alts_now), "independent": independent})
        for r in res.rejected:
            self.declog.record("duplicate", "A direction was rejected before planning", f"Rejected '{r.get('name')}'.", r.get("rejected_because") or "too close to another route.", data={"axis": r.get("axis")})
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
        self.bus.emit("strategies", strategies=strategies, settings={k: self.settings.get(k) for k in ("n", "approach", "models", "judge_model", "reviewer_model")},
                       rejected=res.rejected, warnings=res.warnings,
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
            if str(self.settings.get("approach") or "independent") == "independent":
                self._absorb_plans(sim_ledgers)
            k = exec_count(self.settings["execute"], len(alts))
            ranked = [l for l in rank_for_execution(sim_ledgers) if not l.strategy.get("converged_with")]
            chosen = [l.route for l in ranked[:k]]
            self.meta["executed"] = chosen
            self.save()
            if chosen:
                detail = "; ".join(f"{l.route} (impact {l.blast.score}/10, undo {l.reversibility}/5, {len([i for i in l.issues if i.kind == 'predicted'])} predicted risk(s))" for l in ranked)
                self.declog.record("select", "Which plans get built", f"Building {', '.join(chosen)} for real.",
                                   f"Plans are ranked by predicted impact, predicted risk and ease of undo, best first: {detail}. Your setting builds the top {len(chosen)}.",
                                   data={"chosen": chosen})
            else:
                self.declog.record("select", "Which plans get built", "No real builds: plans only.", f"The execute setting is '{self.settings['execute']}', so routes are planned and compared but not built.")
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
        except ModelOverrideRefused as exc:
            # Final for the run: a refused model is not swapped for another, so there is no honest result to show.
            self.meta["warnings"].append(str(exc))
            self.set_status("failed", error=str(exc))
        except BudgetExceeded as exc:
            self.meta["warnings"].append(str(exc))
            self.set_status("failed", error=str(exc))
        except Exception as exc:  # pragma: no cover - last-resort guard
            self.meta["warnings"].append(f"run failed: {short(exc, 200)}")
            self.set_status("failed", error=str(exc))

    def _absorb_plans(self, sim_ledgers: list[Ledger]) -> None:
        """Independent agents named their own approach in their plan: copy it into the run, and flag plans that match."""
        strategies = self.meta["strategies"]
        by_id = {s["id"]: s for s in strategies}
        for led in sim_ledgers:
            if led.route in by_id:
                keep = {k: v for k, v in led.strategy.items() if k not in ("id", "baseline", "model")}
                by_id[led.route].update(keep)
        rc = self.context()
        pool = [s for s in strategies if s["baseline"]] + [by_id[l.route] for l in sim_ledgers if l.status == "done" and l.route in by_id]
        flat = [{**s, "axis": ""} for s in pool]          # free-text axis labels are not comparable; use the plans' wording only
        seen: set[str] = set()
        for i, j, score, _why in divergemod.find_duplicates(flat, float(self.settings["diversity_threshold"])):
            a, b = pool[i], pool[j]
            if b.get("baseline") or b["id"] in seen:
                continue
            seen.add(b["id"])
            b["converged_with"] = a["id"]
            led = next((x for x in sim_ledgers if x.route == b["id"]), None)
            if led:
                led.strategy["converged_with"] = a["id"]
                led.notes.append(f"independently chose almost the same approach as route {a['id']} (plans {score:.0%} alike), so it is not built again")
                save_ledger(rc.route_dir(led.route), led)
                publish(rc, led)
            same_as_you = bool(a.get("baseline"))
            self.declog.record(
                "duplicate", f"Agent {b['id']} matched {'what you built' if same_as_you else 'agent ' + a['id']}",
                f"Agent {b['id']} independently chose almost the same approach as route {a['id']}" + (" (what you built)" if same_as_you else "") + ".",
                f"Their plans are {score:.0%} alike (threshold {float(self.settings['diversity_threshold']):.0%}). Building it again would add cost and no new information, "
                "so it is kept as a plan only. Two agents landing on the same idea is itself a sign it is the obvious route.",
                route=b["id"], data={"with": a["id"], "similarity": round(score, 2)})
        self.save()

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
        judge = str(self.settings.get("judge_model") or "") or None
        reviewer = str(self.settings.get("reviewer_model") or "") or judge
        res = councilmod.run_council(self.llm, ledgers, rev, goal=self.meta["goal"], run_id=self.run_id,
                                     mode=self._mode_label(), run_dir=self.dir, reviewer_model=reviewer, judge_model=judge)
        self.meta["winner"] = res.winner
        self.meta["runner_up"] = res.runner_up
        self.meta["warnings"] += [w for w in res.warnings if w not in self.meta["warnings"]]
        self.save()
        nar = res.narrative or {}
        self.bus.emit("decision", winner=res.winner, runner_up=res.runner_up, ranking=res.ranking, confidence=nar.get("confidence"),
                      warnings=self.meta["warnings"],
                      judge_model=(judge or ""),
                      reviewer_model=(reviewer or ""),
                      can_converge=bool((nar.get("borrow") or nar.get("risks_carried")) and "Z" not in (self.meta.get("routes") or [])
                                        and res.winner != "Z"))
        self._log_verdict(res)
        self.set_status("decided")
        return res

    def _log_verdict(self, res: "councilmod.CouncilResult") -> None:
        top, nxt = res.ranking[0], (res.ranking[1] if len(res.ranking) > 1 else None)
        nar = res.narrative or {}
        why_won = " ".join(x.get("text", "") for x in (nar.get("winner_why") or [])[:2])
        conf = nar.get("confidence") or "medium"
        margin = (top["score"] - nxt["score"]) if nxt else None
        self.declog.record(
            "council", "The council's pick",
            f"Route {top['route']} ({top.get('name', '')}) won with {top['score']}" + (f" against {nxt['score']} for route {nxt['route']}." if nxt else "."),
            f"{conf.capitalize()} confidence" + (f": {nar['confidence_note']}" if nar.get("confidence_note") else (f", {margin:.2f} points clear" if margin is not None else ""))
            + ". Three reviewers scored every route blind on goal fit, correctness, blast radius, simplicity, reversibility and cost; the judge combined them with fixed weights.",
            route=top["route"], thinking=why_won,
            refs=[r for x in (nar.get("winner_why") or [])[:2] for r in (x.get("refs") or [])],
            data={"winner": top["route"], "score": top["score"], "margin": margin, "confidence": conf})
        for lid, items in (nar.get("losers") or {}).items():
            if items:
                self.declog.record("council", f"Why route {lid} lost", f"Route {lid} finished behind the winner.", " ".join(x.get("text", "") for x in items[:2]),
                                   route=lid, refs=[r for x in items[:2] for r in (x.get("refs") or [])])
        if nar.get("borrow"):
            self.declog.record("council", "Ideas worth borrowing", "Ideas from losing routes worth keeping: " + " ".join(f"({b.get('from')}) {b.get('text', '')}" for b in nar["borrow"][:3]),
                               "The judge named these as strengths of routes that did not win. /branchpoint converge builds them into the winner.",
                               refs=[r for b in nar["borrow"][:3] for r in (b.get("refs") or [])])
        for rk in (nar.get("risks_carried") or [])[:2]:
            self.declog.record("council", "Risk the winner still carries", rk.get("text", ""), "The judge flagged this and the winner does not remove it.", route=top["route"], refs=rk.get("refs") or [])
        self.declog.write_markdown(self.meta)

    def _mode_label(self) -> str:
        if self.meta["mode"] == "rewind":
            return f"retrospective, rewound to prompt {self.meta.get('prompt_n')}"
        return "prospective"

    # ------------------------------------------------------------------ converge
    def converge(self) -> Ledger:
        """Build one more route: the winner plus the ideas worth borrowing from the others, judged blind with the rest."""
        assert self.llm is not None
        dec = read_json(self.dir / "decision.json") or {}
        if not self.meta.get("winner") or not dec:
            raise BranchpointError("Converge builds on the council's winner, so run the council first (and converge before adopting).")
        if "Z" in (self.meta.get("routes") or []):
            if (self.meta.get("converged") or {}).get("judged") or dec.get("winner") == "Z":
                raise BranchpointError("This run already has a converged route (Z). Adopt Z or the original winner.")
            # An earlier attempt stopped before the converged verdict was recorded: it is retryable, so clear the
            # half-built route instead of refusing for ever on a Z that never produced a decision.
            self._drop_failed_converge()
        nar = dec.get("narrative") or {}
        borrow, risks = nar.get("borrow") or [], nar.get("risks_carried") or []
        if not borrow and not risks:
            raise BranchpointError("The council found nothing worth borrowing and no risk left to reduce, so there is nothing to converge. Adopt the winner.")
        win = self.meta["winner"]
        leds = {l.route: l for l in self.ledgers()}
        w = leds[win]
        rev = RunEvidence(self.dir)
        froms = sorted({str(b.get("from")) for b in borrow if b.get("from") in leds})
        parts = [f"CONVERGE: you are improving the council's winner, not starting over. Route {win} ({w.strategy.get('name')}) is the approach to keep."]
        why = " ".join(x.get("text", "") for x in (nar.get("winner_why") or [])[:2])
        if why:
            parts.append(f"Why it won: {why}")
        if w.head_sha:
            parts.append("Its finished code is already in this directory. Keep it and its passing tests.")
        else:
            plan = rev.resolve(f"{win}:plan:md")
            parts.append("It was only planned, not built. Build it from this plan:\n" + short((plan or {}).get("text") or w.strategy.get("plan_md") or w.strategy.get("thesis", ""), 3000))
        if borrow:
            lines = []
            for b in borrow[:4]:
                src = leds.get(str(b.get("from")))
                lines.append(f"- from route {b.get('from')}" + (f" ({src.strategy.get('name')}: {short(src.strategy.get('thesis'), 160)})" if src else "") + f": {b.get('text', '')}")
            parts.append("Ideas worth borrowing from the other routes (apply one only if it fits cleanly):\n" + "\n".join(lines))
        if risks:
            parts.append("Risks the winner still carries (reduce them without a redesign where you can):\n" + "\n".join(f"- {r.get('text', '')}" for r in risks[:3]))
        parts.append("Do not add scope. Every change must keep the project's checks green.")
        strat = {"id": "Z", "axis": "converged", "name": short(f"Converged {win}" + "".join(f" + {f}" for f in froms), 60),
                 "thesis": f"Start from route {win} and fold in the best ideas from " + (", ".join(froms) or "the other routes") + ", reducing the risks the council flagged.",
                 "key_decisions": [short(b.get("text", ""), 200) for b in borrow[:4]] + [short("reduce: " + r.get("text", ""), 200) for r in risks[:2]],
                 "predicted_files": list(w.strategy.get("predicted_files") or [f["path"] for f in w.files][:15]), "trade_off": "", "baseline": False,
                 "independent": False, "converged_from": win, "borrowed_from": froms, "model": w.strategy.get("model") or ""}
        if self.meter.cap is not None:
            extra = int(self.settings.get("converge_tokens") or 150_000)
            self.meter.cap += extra
            self.settings["budget_tokens"] = self.meter.cap
        rc = self.context()
        rc.route_token_cap = int(self.settings.get("converge_tokens") or 150_000)
        self.set_status("converging", converging="Z")
        (self.dir / "decision.pre-converge.json").write_text((self.dir / "decision.json").read_text(encoding="utf-8"), encoding="utf-8")
        if (self.dir / "decision.md").exists():
            (self.dir / "decision.pre-converge.md").write_text((self.dir / "decision.md").read_text(encoding="utf-8"), encoding="utf-8")
        self.meta["strategies"].append(strat)
        self.meta["routes"].append("Z")
        led = Ledger(route="Z", strategy=strat, fidelity="execute", status="queued", base_sha=self.meta["base_sha"])
        save_ledger(rc.route_dir("Z"), led)
        publish(rc, led)
        self.save()
        self.declog.record(
            "converge", "Building the converged route",
            f"Building route Z: winner {win} plus ideas from {', '.join(froms) or 'the other routes'}.",
            "You asked for the best parts of every route in one. It starts from the winner's own code, applies the ideas the judge named, and is then judged blind against all the other routes like any other, so it has to earn its place."
            + (f" The run's token budget was raised by {int(self.settings.get('converge_tokens') or 150_000):,} for this step." if self.meter.cap is not None else ""),
            route="Z", data={"from": win, "borrowed_from": froms}, refs=[r for b in borrow[:4] for r in (b.get("refs") or [])])
        before = next((r for r in (dec.get("ranking") or []) if r["route"] == win), {})
        try:
            led = self._execute_one(RouteSpec("Z", strat, strat.get("model") or None, start_sha=w.head_sha or None, guide="\n\n".join(parts)))
            self.meta["converged"] = {"from": win, "borrowed_from": froms}
            self.save()
            res = self.council()
            z = next((r for r in res.ranking if r["route"] == "Z"), None)
            wnow = next((r for r in res.ranking if r["route"] == win), before)
            if z and res.winner == "Z":
                outcome = f"The converged route won: {z['score']} against {wnow.get('score')} for the original winner {win}."
            elif z:
                outcome = f"The original winner {win} still scores higher ({wnow.get('score')} against {z['score']} for Z), so converging did not help here. Adopt {res.winner}."
            else:
                outcome = "The converged route did not finish, so the original verdict stands."
            self.declog.record("converge", "Result of converging", outcome, "The council re-scored every route blind, with the converged route included, using the same fixed weights.",
                               route="Z", data={"winner": res.winner, "z": (z or {}).get("score")})
            self.meta["converged"]["outcome"] = outcome
            self.meta["converged"]["judged"] = True      # only now has the converged route a verdict (retry guard)
            self.save()
            self.declog.write_markdown(self.meta)
        except Exception as exc:      # a refused model, a budget stop or any other error: record it, do not look decided
            self._fail_converge(exc)
            raise
        return led

    def _fail_converge(self, exc: BaseException) -> None:
        """Persist a converge that stopped, so the run is neither left mid-flight nor mistaken for decided.

        The previous Decision Record is never touched (the attempt only ever writes ``decision.pre-converge``), and
        the half-built route Z is marked failed only when its build did not finish: when the build succeeded and the
        council is what refused, the ledger keeps saying the build succeeded.
        """
        reason = str(exc) or type(exc).__name__
        zled = load_ledger(self.dir / "routes" / "Z")
        if zled is not None and zled.status not in ("done",):
            zled.status = "failed"
            zled.summary = short(f"converge stopped: {reason}", 420)
            zled.notes.append(f"converge stopped: {reason}")
            save_ledger(self.dir / "routes" / "Z", zled)
            if self.rc is not None:
                publish(self.rc, zled)
        if reason not in self.meta["warnings"]:
            self.meta["warnings"].append(reason)
        self.set_status("failed", error=reason)

    def _drop_failed_converge(self) -> None:
        """Clear a converge attempt that never reached a verdict, so the retry starts from the same place."""
        rc = self.context()
        worktree_remove(self.repo, rc.worktree_path("Z"))
        delete_branch(self.repo, rc.branch("Z"))
        zdir = self.dir / "routes" / "Z"
        if zdir.exists():
            shutil.rmtree(zdir, ignore_errors=True)
        self.meta["routes"] = [r for r in (self.meta.get("routes") or []) if r != "Z"]
        self.meta["strategies"] = [s for s in (self.meta.get("strategies") or []) if s.get("id") != "Z"]
        self.meta.pop("converged", None)
        self.save()

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
