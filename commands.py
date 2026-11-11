"""The ``/branchpoint`` command and the ``branchpoint_*`` tools."""
from __future__ import annotations

import json
import shlex
import threading
from pathlib import Path
from typing import Any, Callable

from . import adopt as adoptmod
from . import anchor as anchormod
from .llm import LLM, CtxLLM, redact_for_model, redaction_available
from .paths import data_dir, hermes_home
from .declog import read_entries, render_markdown
from .run import DEFAULT_SETTINGS, MAX_ROUTES, MIN_ROUTES, Run, list_runs, render_status
from .server import ViewServer
from .util import BranchpointError, read_json, short

USAGE = """Branchpoint — rewind, explore alternative builds, and let a blind council pick one.

  /branchpoint fork "<goal>" [--n 4] [--execute top2|top1|all|none]     give the same task to several agents before you build
  /branchpoint rewind <N> [--goal "<text>"] [--base <rev>] [...]        undo what you built since prompt N and try other ways
  /branchpoint status [run]        tree of routes, blast radius, issues, cost
  /branchpoint view [run]          open the live multiverse view (loopback only)
  /branchpoint council [run]       (re)run the council on finished routes
  /branchpoint converge [run]      build one more route: the winner plus the best ideas from the others, judged blind with the rest
  /branchpoint log [run]           the decision log: what was decided at each step, and why
  /branchpoint adopt <route> [merge|replay|adr|discard] [--keep] [--run id]
  /branchpoint undo [run]          reverse an adopt
  /branchpoint list | abort [run]

Options: --repo <path> (chat surfaces have no working directory of their own) · --n 2-8 routes, default 4 (A-D) · --execute how many routes get a real build (default top2) · --budget 600k tokens (a hard stop for Branchpoint's own calls, a per-build cap for build agents; see Cost in the README)
         --models a,b,c (one model per agent, in order) · --judge-model <m> · --reviewer-model <m> · --approach independent|assigned
         --test-cmd/--lint-cmd/--type-cmd "<cmd>" · --child-model <m> · --wait
         --unsafe-checks (run checks without an OS sandbox: only if no sandbox is available and you accept it) · --allow-sensitive (land execution-sensitive files on adopt)

Each agent is given the same task and plans and builds it its own way (--approach independent, the default). With --approach assigned,
each agent is instead handed a different direction to build."""

_FLAGS = {"--n": "n", "-n": "n", "--execute": "execute", "-x": "execute", "--budget": "budget", "--test-cmd": "test_command",
          "--lint-cmd": "lint_command", "--type-cmd": "type_command", "--base": "base", "--goal": "goal", "--models": "models",
          "--run": "run", "--child-model": "child_model", "--seconds": "route_seconds", "--repo": "repo",
          "--judge-model": "judge_model", "--reviewer-model": "reviewer_model", "--approach": "approach"}
_BOOLS = {"--keep": "keep", "--wait": "wait", "--no-view": "no_view", "--inject": "inject",
          "--unsafe-checks": "unsafe_checks", "--allow-sensitive": "allow_sensitive"}


def parse_args(raw: str) -> tuple[str, list[str], dict[str, Any]]:
    raw = (raw or "").replace("\u2014", "--").replace("\u2013", "-")
    try:
        toks = shlex.split(raw)
    except ValueError as exc:
        raise BranchpointError(f"Could not parse the command: {exc}")
    if not toks:
        return "help", [], {}
    cmd, rest = toks[0].lower(), toks[1:]
    pos: list[str] = []
    opts: dict[str, Any] = {}
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in _BOOLS:
            opts[_BOOLS[t]] = True
        elif t in _FLAGS:
            if i + 1 >= len(rest):
                raise BranchpointError(f"{t} needs a value.\n\n{USAGE}")
            opts[_FLAGS[t]] = rest[i + 1]
            i += 1
        elif t.startswith("--") and "=" in t and t.split("=", 1)[0] in _FLAGS:
            k, v = t.split("=", 1)
            opts[_FLAGS[k]] = v
        elif t.startswith("-") and len(t) > 1 and not t[1:].isdigit():
            raise BranchpointError(f"Unknown option {t}.\n\n{USAGE}")
        else:
            pos.append(t)
        i += 1
    return cmd, pos, opts


def parse_tokens(v: str) -> int:
    s = str(v).strip().lower().replace(",", "")
    try:
        if s.endswith("m"):
            return int(float(s[:-1]) * 1_000_000)
        if s.endswith("k"):
            return int(float(s[:-1]) * 1_000)
        return int(float(s))
    except ValueError:
        raise BranchpointError(f"--budget expects a token count like 600k, got '{v}'.")


class BranchpointService:
    def __init__(self, ctx: Any = None, llm: LLM | None = None, agent: Any = None,
                 get_setting: Callable[[str, Any], Any] | None = None):
        self.ctx = ctx
        self._llm = llm
        self.agent = agent
        self._get = get_setting or (lambda k, d: d)
        self.runs: dict[str, Run] = {}
        self.announce: Callable[[str], None] | None = None   # standalone CLI: print the live-view URL as soon as it exists
        self.server: ViewServer | None = None
        self._lock = threading.Lock()
        self.last_error: str = ""   # set by handle() so a script can exit non-zero instead of printing and passing

    # ------------------------------------------------------------------ plumbing
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            if self.ctx is None:
                raise BranchpointError("No LLM backend is configured.")
            self._llm = CtxLLM(self.ctx, default_model=str(self._get("council_model", "") or "") or None)
        return self._llm

    def settings(self, opts: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = opts or {}
        s = dict(DEFAULT_SETTINGS)
        s["n"] = int(self._get("default_routes", s["n"]) or s["n"])
        s["execute"] = str(self._get("execute", s["execute"]) or s["execute"])
        s["budget_tokens"] = int(self._get("budget_tokens", s["budget_tokens"]) or s["budget_tokens"])
        s["max_parallel"] = max(1, min(3, int(self._get("max_parallel", s["max_parallel"]) or 3)))
        s["route_seconds"] = int(float(self._get("route_minutes", s["route_seconds"] / 60) or 15) * 60)
        for k in ("test_command", "lint_command", "type_command", "child_model", "judge_model", "reviewer_model"):
            v = self._get(k, "")
            if v:
                s[k] = str(v)
        cfg_models = self._get("models", "")
        if cfg_models:
            s["models"] = [m.strip() for m in str(cfg_models).split(",") if m.strip()]
        ap = str(self._get("approach", "") or "").strip().lower()
        if ap in ("independent", "assigned"):
            s["approach"] = ap
        s["converge_tokens"] = int(self._get("converge_tokens", s["converge_tokens"]) or s["converge_tokens"])
        links = self._get("link_dirs", "")
        if links:
            s["link_dirs"] = [x.strip() for x in str(links).split(",") if x.strip()]
        if opts.get("n"):
            s["n"] = int(opts["n"])
        if not MIN_ROUTES <= s["n"] <= MAX_ROUTES:
            raise BranchpointError(f"--n must be between {MIN_ROUTES} and {MAX_ROUTES} routes (default 4: A, B, C, D).")
        if opts.get("execute"):
            s["execute"] = str(opts["execute"])
        # Accepting unsandboxed checks is a per-run choice with no config default: without an OS sandbox a
        # route's checks would execute agent-written code as you (see sandbox.py), so --execute refuses
        # unless the user asked for exactly that.
        s["unsafe_checks"] = bool(opts.get("unsafe_checks"))
        s["allow_sensitive"] = bool(opts.get("allow_sensitive"))
        if opts.get("budget"):
            s["budget_tokens"] = parse_tokens(opts["budget"])
        for k in ("test_command", "lint_command", "type_command", "child_model", "judge_model", "reviewer_model"):
            if opts.get(k):
                s[k] = opts[k]
        if opts.get("approach"):
            a = str(opts["approach"]).lower()
            if a not in ("independent", "assigned"):
                raise BranchpointError("--approach must be 'independent' or 'assigned'.")
            s["approach"] = a
        if opts.get("route_seconds"):
            s["route_seconds"] = int(opts["route_seconds"])
        if opts.get("models"):
            s["models"] = [m.strip() for m in str(opts["models"]).split(",") if m.strip()]
        return s

    def cwd(self) -> str:
        if self.ctx is not None and hasattr(self.ctx, "session_cwd"):
            try:
                c = self.ctx.session_cwd()
                if c:
                    return c
            except Exception:
                pass
        import os
        return os.getcwd()

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            if run_id in self.runs:
                return self.runs[run_id]
        run = Run.load(data_dir(), run_id, self.llm if self._llm or self.ctx else None, self.agent)
        with self._lock:
            self.runs[run_id] = run
        return run

    def last_run_failed(self) -> bool:
        """True when the newest run exists and ended ``failed`` (used for the standalone CLI exit code)."""
        try:
            rid = self.latest_run_id()
            return bool(rid) and self.get_run(rid).meta.get("status") == "failed"     # type: ignore[union-attr]
        except Exception:
            return False

    def latest_run_id(self) -> str | None:
        with self._lock:
            live = [r for r in self.runs.values()]
        if live:
            return max(live, key=lambda r: r.meta["created"]).run_id
        runs = list_runs(data_dir())
        return runs[0]["run_id"] if runs else None

    def pick_run(self, pos: list[str], opts: dict[str, Any]) -> Run:
        rid = opts.get("run") or (pos[0] if pos and len(pos[0]) > 6 and "-" in pos[0] else None) or self.latest_run_id()
        if not rid:
            raise BranchpointError("No Branchpoint runs yet. Start one with /branchpoint fork \"<goal>\".")
        return self.get_run(rid)

    def ensure_view(self) -> ViewServer:
        if self.server is None:
            self.server = ViewServer(self.get_run, self.latest_run_id)
            self.server.start()
        return self.server

    # ------------------------------------------------------------------ commands
    def handle(self, raw: str, session_id: str = "", **_kw: Any) -> str:
        self.last_error = ""
        try:
            cmd, pos, opts = parse_args(raw)
            if cmd in ("help", "-h", "--help", "?"):
                return USAGE
            if cmd == "fork":
                return self.start("fork", pos, opts, session_id)
            if cmd == "rewind":
                return self.start("rewind", pos, opts, session_id)
            if cmd == "status":
                return render_status(self.pick_run(pos, opts))
            if cmd == "view":
                run = self.pick_run(pos, opts)
                url = self.ensure_view().run_url(run.run_id)
                run.meta["view_url"] = url
                run.save()
                return f"Live view (loopback only, token in the URL):\n{url}"
            if cmd == "council":
                return self.council(self.pick_run(pos, opts))
            if cmd == "converge":
                return self.converge(self.pick_run(pos, opts), bool(opts.get("wait", True)))
            if cmd == "log":
                run = self.pick_run(pos, opts)
                ents = read_entries(run.dir)
                return (render_markdown(ents, run.meta) if ents else "No decisions were logged for this run.") + f"\nFull log: {run.dir / 'decision-log.md'}"
            if cmd == "adopt":
                return self.adopt(pos, opts)
            if cmd == "undo":
                run = self.pick_run(pos, opts)
                return adoptmod.undo_adopt(run.repo, run.dir, run.meta)
            if cmd == "list":
                return self.list_text()
            if cmd == "abort":
                run = self.pick_run(pos, opts)
                run.abort_run()
                return f"Abort requested for run {run.run_id}; build agents are being stopped."
            self.last_error = f"unknown subcommand '{cmd}'"
            return f"Unknown subcommand '{cmd}'.\n\n{USAGE}"
        except BranchpointError as exc:
            self.last_error = str(exc) or type(exc).__name__
            return f"Branchpoint: {exc}"
        except Exception as exc:  # never raise into the host turn
            self.last_error = f"{type(exc).__name__}: {short(exc, 240)}"
            return f"Branchpoint hit an unexpected error: {type(exc).__name__}: {short(exc, 240)}"

    def start(self, mode: str, pos: list[str], opts: dict[str, Any], session_id: str) -> str:
        settings = self.settings(opts)
        cwd = str(Path(opts["repo"]).expanduser()) if opts.get("repo") else self.cwd()
        redact = redact_for_model if redaction_available() else None
        if mode == "fork":
            goal = " ".join(pos).strip()
            if not goal:
                raise BranchpointError('Give the goal to explore, e.g. /branchpoint fork "add caching to /report" --n 4')
            anchor = anchormod.resolve_fork(cwd=cwd, goal=goal, session_id=session_id, redact=redact)
            stats = None
        else:
            if not pos or not pos[0].isdigit():
                raise BranchpointError("Usage: /branchpoint rewind <prompt number> (the number /rewind shows).")
            n = int(pos[0])
            messages = anchormod.load_session_messages(session_id)
            anchor = anchormod.resolve_rewind(cwd=cwd, data_dir=data_dir(), hermes_home=hermes_home(), session_id=session_id, n=n,
                                              base_override=opts.get("base"), goal_override=opts.get("goal"), messages=messages, redact=redact)
            goal = anchor.goal
            stats = anchormod.baseline_stats(messages, n) if messages else None
            if not goal:
                raise BranchpointError("Could not recover the goal from prompt %d; pass --goal \"...\"." % n)
        if self.ctx is not None and not redaction_available():
            anchor.notes.append("host redaction unavailable: transcript text is sent as-is")
        run = Run.create(data_dir(), anchor=anchor, mode=mode, goal=goal, settings=settings, llm=self.llm, agent=self.agent, baseline_stats=stats)
        with self._lock:
            self.runs[run.run_id] = run
        self._attach_notifier(run)
        try:
            strategies = run.prepare()
        except Exception:
            run.set_status("failed")
            raise
        url = ""
        if not opts.get("no_view"):
            try:
                url = self.ensure_view().run_url(run.run_id)
                run.meta["view_url"] = url
                run.save()
                if self.announce:
                    self.announce(f"Live view: {url}")
            except Exception:
                url = ""
        run.go(background=not opts.get("wait"))
        if opts.get("wait"):
            run.join()
        return self._start_text(run, strategies, anchor, url, settings)

    def _start_text(self, run: Run, strategies: list[dict[str, Any]], anchor: anchormod.Anchor, url: str, settings: dict[str, Any]) -> str:
        m = run.meta
        head = (f"Branchpoint run {run.run_id} — " + (f"rewound to prompt {anchor.prompt_n}" if m["mode"] == "rewind" else "forked from your working tree now")
                + f"  (base {anchor.base_sha[:10]} via {anchor.source})")
        lines = [head, f"Goal: {short(m['goal'], 140)}", ""]
        if str(settings.get("approach")) == "independent":
            names = [s["id"] for s in strategies if not s.get("baseline")]
            lines.append(f"The same task goes to {len(names)} agents ({', '.join(names)}). Each plans and builds it its own way; their approaches appear as they finish planning."
                         + (" Route A is what you built." if m.get("has_baseline") else ""))
            for s in strategies:
                if s.get("baseline"):
                    lines.append(f"  {s['id']}  {s['name']} ← what you built\n       {short(s['thesis'], 150)}")
        else:
            lines.append(f"{len(strategies)} distinct routes:")
            for s in strategies:
                tag = " ← what you built" if s.get("baseline") else ""
                lines.append(f"  {s['id']}  [{s['axis']}] {s['name']}{tag}\n       {short(s['thesis'], 150)}")
        for r in m.get("rejected", [])[:3]:
            lines.append(f"  ✗ rejected: {short(r['name'], 40)} — {r.get('rejected_because')}")
        for w in m.get("warnings", []):
            lines.append(f"  ! {w}")
        for n in anchor.notes:
            lines.append(f"  note: {n}")
        n_alt = len([s for s in strategies if not s.get("baseline")])
        alts = [s for s in strategies if not s.get("baseline")]
        who = ", ".join(f"{s['id']} = {s.get('model') or settings.get('child_model') or 'your default model'}" for s in alts)
        council_model = settings.get("reviewer_model") or settings.get("judge_model") or ""
        shown = council_model or "your default model"
        if settings.get("judge_model") and settings.get("reviewer_model") and settings["reviewer_model"] != settings["judge_model"]:
            shown += f" / {settings['judge_model']}"
        lines.append("")
        lines.append(f"Models: {who}. Reviewers and judge: {shown}.")
        from .run import exec_count
        k = exec_count(settings["execute"], n_alt)
        lines += ["", f"Plan: simulate {n_alt} route(s)" + (f", then really build the top {k} (file tools only; your project's checks are run by Branchpoint)" if k else " (no builds)")
                  + f". Budget {settings['budget_tokens'] // 1000}k tokens (a hard stop for Branchpoint's own calls, a per-build cap for build agents)."]
        lines.append("Your working tree is not touched until you run /branchpoint adopt.")
        if url:
            lines.append(f"Live view: {url}")
        if m["status"] in ("decided", "failed", "aborted"):
            lines += ["", render_status(run)]
        else:
            lines.append("Follow along: /branchpoint status   (the council verdict is posted when all routes finish)")
        return "\n".join(lines)

    def _attach_notifier(self, run: Run) -> None:
        if self.ctx is None or not hasattr(self.ctx, "post_message"):
            return
        try:
            handle = self.ctx.post_message(f"Branchpoint run {run.run_id} started…")
        except Exception:
            return

        def listener(ev: dict[str, Any]) -> None:
            if ev["type"] == "decision":
                text = f"Branchpoint {run.run_id}: council picked Route {ev['winner']}. /branchpoint status · decision: {run.dir / 'decision.md'}"
            elif ev["type"] == "status" and ev["status"] in ("failed", "aborted"):
                text = f"Branchpoint {run.run_id}: {ev['status']}."
            else:
                return
            try:
                handle.edit(text)  # type: ignore[attr-defined]
            except Exception:
                pass

        run.bus.listeners.append(listener)

    def council(self, run: Run) -> str:
        if run.thread and run.thread.is_alive():
            return "The routes are still running; the council starts automatically when they finish. Check /branchpoint status."
        res = run.council()
        head = f"Council decided: Route {res.winner}" + (f" over {res.runner_up}" if res.runner_up else "")
        rows = "\n".join(f"  #{r['rank']} {r['route']} {short(r['name'], 34):<34} {r['score']}" for r in res.ranking)
        return f"{head}\n{rows}\n\nDecision Record: {run.dir / 'decision.md'}\nNext: /branchpoint adopt {res.winner}"

    def converge(self, run: Run, wait: bool = True) -> str:
        if run.thread and run.thread.is_alive():
            return "The run is still working. Converge starts from the council's winner, so wait for the verdict."
        if wait:
            led = run.converge()
            res = read_json(run.dir / "decision.json") or {}
            rows = "\n".join(f"  #{r['rank']} {r['route']} {short(r['name'], 34):<34} {r['score']}" for r in res.get("ranking", []))
            return (f"Converge finished. {run.meta.get('converged', {}).get('outcome', '')}\n{rows}\n\n"
                    f"Route Z: {led.diff_stats.get('files', 0)} file(s), checks {'failing' if any(i.kind in ('test', 'lint', 'type') and i.severity == 'high' for i in led.issues) else 'passing'}.\n"
                    f"Next: /branchpoint adopt {res.get('winner')}")
        run.thread = threading.Thread(target=run.converge, name=f"branchpoint-converge-{run.run_id}", daemon=True)
        run.thread.start()
        return "Converging in the background. Watch /branchpoint status; the council re-judges when route Z finishes."

    def adopt(self, pos: list[str], opts: dict[str, Any]) -> str:
        words = [p for p in pos]
        mode = next((w.lower() for w in words if w.lower() in ("merge", "replay", "adr", "discard")), None)
        rest = [w for w in words if w.lower() not in ("merge", "replay", "adr", "discard")]
        route = next((w.upper() for w in rest if len(w) <= 2 and w[0].isalpha()), None)
        run = self.pick_run([w for w in rest if w.upper() != route], opts)
        if run.thread and run.thread.is_alive():
            raise BranchpointError("The run is still working. Wait for the council, or /branchpoint abort first.")
        if mode == "discard":
            run.drop_refs()
            removed = run.cleanup(keep=False)
            run.set_status("discarded")
            return f"Discarded run {run.run_id}: removed worktrees {', '.join(removed) or '(none)'}, branches and refs. Ledgers stay in {run.dir}."
        route = route or run.meta.get("winner")
        if not route:
            raise BranchpointError("Name a route, e.g. /branchpoint adopt C merge")
        if route not in run.meta.get("routes", []):
            raise BranchpointError(f"Run {run.run_id} has routes {', '.join(run.meta.get('routes', []))}, not {route}.")
        mode = mode or "merge"
        if mode == "merge":
            msg = adoptmod.adopt_merge(run.repo, run.dir, run.meta, route)
            run.meta["adopted"] = route
        elif mode == "replay":
            inject = None
            if opts.get("inject") and self.ctx is not None and hasattr(self.ctx, "inject_message"):
                inject = lambda text: self.ctx.inject_message(text, "user")  # noqa: E731
            msg = adoptmod.adopt_replay(run.dir, run.meta, route, inject)
        elif mode == "adr":
            msg = adoptmod.adopt_adr(run.repo, run.dir, run.meta, route)
        else:
            raise BranchpointError("adopt modes: merge, replay, adr, discard")
        led_a = next((x for x in run.ledgers() if x.route == route), None)
        run.declog.record("adopt", f"Adopted route {route}" if mode == "merge" else f"Route {route}: {mode}",
                          f"Route {route} ({led_a.strategy.get('name') if led_a else ''}) was {'merged into your working tree' if mode == 'merge' else 'written out as ' + mode}.",
                          ("It is the council's pick. " if route == run.meta.get("winner") else f"You chose it over the council's pick ({run.meta.get('winner')}). ")
                          + (msg.splitlines()[-1] if msg else ""), route=route, data={"mode": mode})
        run.declog.write_markdown(run.meta)
        removed = run.cleanup(keep=bool(opts.get("keep")))
        run.set_status("adopted" if mode == "merge" else run.meta["status"])
        tail = "Worktrees kept (--keep)." if opts.get("keep") else f"Removed worktrees: {', '.join(removed) or '(none)'}."
        return f"{msg}\n{tail}"

    def list_text(self) -> str:
        runs = list_runs(data_dir())
        if not runs:
            return "No Branchpoint runs yet."
        return "\n".join(f"{m['run_id']}  {m['status']:<10} {m['mode']:<6} {short(m['goal'], 70)}" for m in runs[:15])

    # ------------------------------------------------------------------ tools (model-facing, read-mostly)
    def tool_status(self, args: dict[str, Any] | None = None, **_kw: Any) -> str:
        try:
            run = self.pick_run([], {"run": (args or {}).get("run_id")})
            leds = [{"route": l.route, "name": l.strategy.get("name"), "status": l.status, "fidelity": l.fidelity, "blast": l.blast.score,
                     "issues": len(l.issues), "tokens": l.cost.tokens} for l in run.ledgers()]
            return json.dumps({"run_id": run.run_id, "status": run.meta["status"], "winner": run.meta.get("winner"), "routes": leds})
        except BranchpointError as exc:
            return json.dumps({"error": str(exc)})

    def tool_decision(self, args: dict[str, Any] | None = None, **_kw: Any) -> str:
        try:
            run = self.pick_run([], {"run": (args or {}).get("run_id")})
            p = run.dir / "decision.md"
            return json.dumps({"run_id": run.run_id, "decision_markdown": p.read_text(encoding="utf-8") if p.exists() else None,
                               "note": None if p.exists() else "no decision yet"})
        except BranchpointError as exc:
            return json.dumps({"error": str(exc)})

    def tool_fork(self, args: dict[str, Any] | None = None, **_kw: Any) -> str:
        """Model-facing fork: simulate-only (no builds, small budget). Builds need the user's slash command."""
        a = args or {}
        goal = str(a.get("goal") or "").strip()
        if not goal:
            return json.dumps({"error": "goal is required"})
        opts = {"n": a.get("n") or 4, "execute": "none", "budget": "200k", "no_view": True}
        text = self.start("fork", [goal], opts, session_id="")
        return json.dumps({"summary": text})


def tool_schemas() -> dict[str, dict[str, Any]]:
    return {
        "branchpoint_status": {
            "name": "branchpoint_status",
            "description": "Show the status of a Branchpoint run: its routes, blast radius, issue counts and the council winner. Read-only.",
            "parameters": {"type": "object", "properties": {"run_id": {"type": "string", "description": "Run id; defaults to the latest run."}}, "required": []},
        },
        "branchpoint_decision": {
            "name": "branchpoint_decision",
            "description": "Return the Decision Record (markdown with evidence-cited reasoning) of a Branchpoint run. Read-only.",
            "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": []},
        },
        "branchpoint_fork": {
            "name": "branchpoint_fork",
            "description": ("Explore 3-5 genuinely different ways to build a goal BEFORE building it: returns distinct strategies and starts a "
                            "simulate-only analysis (plans + static blast-radius tracing, no builds, 200k token cap). Use when the user is at an "
                            "architecture fork (e.g. Redis vs in-process cache). Real builds and adoption are only available through the user's /branchpoint command."),
            "parameters": {"type": "object", "properties": {
                "goal": {"type": "string", "description": "What the user wants to build, in one or two sentences."},
                "n": {"type": "integer", "minimum": 2, "maximum": 5, "description": "Number of routes (default 4)."}}, "required": ["goal"]},
        },
    }
