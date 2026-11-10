"""Hook implementations. One set of hooks serves two roles, chosen by where the process runs:

* In the user's own session: record a throw-away snapshot of the working tree as each prompt is
  submitted (the anchor that ``/branchpoint rewind N`` later uses). Everything is a no-op otherwise.
* Inside a route's headless build agent (found via the worktree marker): inject the strategy as a hard
  constraint, confine file tools to the worktree, cap tool calls/tokens, log every tool event for the
  ledger, and run the project's checks every few tool calls, feeding results back to the agent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from . import anchor as anchormod
from .checks import CheckSpec, run_check
from .paths import data_dir
from .protocol import ALLOWED_TOOLS, WRITE_TOOLS, extract_paths, outside_root, read_marker, tokens_from_usage
from .util import is_secret_path, repo_root, short, now


log = logging.getLogger("branchpoint")


@dataclass
class ChildState:
    marker: dict[str, Any]
    calls: int = 0
    tokens: int = 0
    dirty: bool = False
    running: bool = False
    pending: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_failures: set[str] = field(default_factory=set)

    @property
    def over_budget(self) -> str:
        if self.calls >= int(self.marker.get("max_tool_calls") or 60):
            return f"the tool-call budget ({self.marker.get('max_tool_calls')}) is used up"
        cap = self.marker.get("token_cap")
        if cap is not None and self.tokens >= int(cap):
            return f"the token budget ({cap}) is used up"
        return ""


class Hooks:
    def __init__(self, ctx: Any = None, settings: Callable[[str, Any], Any] | None = None):
        self._ctx = ctx
        self._settings = settings or (lambda k, d: d)
        self._cache: dict[str, dict[str, Any] | None] = {}
        self._child: dict[str, ChildState] = {}
        self._last_prompt: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _marker(self) -> dict[str, Any] | None:
        """The route marker for the session running this hook, else None.

        Hermes may run a headless chat turn inside a gateway process whose own cwd is unrelated, so the
        *session's* working directory (``ctx.session_cwd()``) is consulted before the process cwd.
        """
        candidates: list[str] = []
        if self._ctx is not None and hasattr(self._ctx, "session_cwd"):
            try:
                c = self._ctx.session_cwd()
                if c:
                    candidates.append(str(c))
            except Exception:
                pass
        try:
            candidates.append(os.getcwd())
        except OSError:
            pass
        for cwd in dict.fromkeys(candidates):
            if cwd not in self._cache:
                self._cache[cwd] = read_marker(cwd)
                log.info("branchpoint: hook cwd %s -> %s", cwd, "route " + self._cache[cwd]["route"] if self._cache[cwd] else "no route marker")
            if self._cache[cwd]:
                return self._cache[cwd]
        return None

    def _state(self, marker: dict[str, Any]) -> ChildState:
        key = marker.get("events_path", "")
        with self._lock:
            st = self._child.get(key)
            if st is None:
                st = self._child[key] = ChildState(marker)
            return st

    def _log(self, marker: dict[str, Any], event: dict[str, Any]) -> None:
        try:
            with open(marker["events_path"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": round(now(), 3), **event}, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ pre_llm_call
    def pre_llm_call(self, session_id: str = "", user_message: str = "", conversation_history: list | None = None,
                     **_kw: Any) -> dict[str, str] | None:
        marker = self._marker()
        if marker:
            st = self._state(marker)
            s = marker.get("strategy") or {}
            lines = [f"[Branchpoint — route {marker['route']}] HARD CONSTRAINT: build exactly this strategy, nothing else.",
                     f"Axis: {s.get('axis')}. Thesis: {s.get('thesis')}"]
            lines += [f"- {d}" for d in s.get("key_decisions") or []]
            lines.append("You have file tools only; project checks run automatically and their results are appended to tool outputs. Stay inside this directory.")
            if st.over_budget:
                lines.append(f"STOP: {st.over_budget}. Reply now with your 3-sentence summary.")
            return {"context": "\n".join(lines)}
        if not self._settings("anchor_on_turn", True):
            return None
        self._record_anchor(session_id, user_message, conversation_history or [])
        return None

    def _record_anchor(self, session_id: str, prompt: str, history: list) -> None:
        if not session_id or not prompt or prompt.lstrip().startswith("/"):
            return
        try:
            cwd = ""
            if self._ctx is not None and hasattr(self._ctx, "session_cwd"):
                cwd = self._ctx.session_cwd() or ""
            cwd = cwd or os.getcwd()
            repo = repo_root(cwd)
        except Exception:
            return
        h = hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if self._last_prompt.get(session_id) == h:
            return
        self._last_prompt[session_id] = h
        prior = [m for m in history if isinstance(m, dict) and m.get("role") == "user"]
        n = len(prior) if prior and prior[-1].get("content") == prompt else len(prior) + 1
        try:
            dd = data_dir()
        except Exception:
            return
        t = threading.Thread(target=anchormod.record_turn, args=(dd, repo, session_id, n, prompt), daemon=True, name="bp-anchor")
        t.start()
        t.join(1.5)   # usually done well inside this; a slow repo finishes in the background

    # ------------------------------------------------------------------ pre_tool_call
    def pre_tool_call(self, tool_name: str = "", args: dict | None = None, **_kw: Any) -> dict[str, Any] | None:
        marker = self._marker()
        if not marker:
            return None
        st = self._state(marker)
        reason = ""
        if tool_name not in ALLOWED_TOOLS:
            reason = f"'{tool_name}' is not available inside a Branchpoint route (file tools only)."
        else:
            root = marker.get("confine_root") or marker.get("worktree")
            bad = outside_root(extract_paths(tool_name, args), root, root)
            if bad:
                reason = f"path '{short(bad, 80)}' is outside this route's worktree. Use paths relative to the current directory."
            elif tool_name in WRITE_TOOLS:
                paths = extract_paths(tool_name, args)
                if any(is_secret_path(p) for p in paths):
                    reason = "writing credential/secret files is not allowed in a route."
        if not reason and st.over_budget:
            reason = f"{st.over_budget}. Stop and reply with your 3-sentence summary."
        if reason:
            self._log(marker, {"tool": tool_name, "blocked": reason, "path": (extract_paths(tool_name, args) or [""])[0]})
            return {"action": "block", "message": f"Branchpoint: {reason}"}
        return None

    # ------------------------------------------------------------------ post_tool_call
    def post_tool_call(self, tool_name: str = "", args: dict | None = None, result: Any = None, duration_ms: int = 0,
                       status: str = "", error_type: str = "", error_message: str = "", **_kw: Any) -> None:
        marker = self._marker()
        if not marker:
            return
        st = self._state(marker)
        paths = extract_paths(tool_name, args)
        err = error_message or ""
        if not err and isinstance(result, str) and '"error"' in result[:200]:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and parsed.get("error"):
                    err = str(parsed["error"])
            except ValueError:
                pass
        failed = bool(err) or status in ("error", "failed")
        self._log(marker, {"tool": tool_name, "path": paths[0] if paths else "", "duration_ms": duration_ms,
                           "status": "error" if failed else "ok", "error": short(err, 240) if err else ""})
        with st.lock:
            st.calls += 1
            if tool_name in WRITE_TOOLS and not failed:
                st.dirty = True
            due = (st.calls % int(marker.get("check_every") or 5) == 0) and st.dirty and not st.running
            if due:
                st.running = True
                st.dirty = False
        if due:
            threading.Thread(target=self._run_checks, args=(st,), daemon=True, name="bp-checks").start()

    def _run_checks(self, st: ChildState) -> None:
        m = st.marker
        calls = st.calls
        results = []
        try:
            for c in m.get("checks") or []:
                spec = CheckSpec(c["kind"], list(c["argv"]), float(c.get("timeout") or 240))
                results.append(run_check(m.get("confine_root") or m["worktree"], spec))
            entry = {"calls": calls, "ts": round(now(), 3), "results": [{**r.to_dict(), "output": r.output[-3000:]} for r in results]}
            with open(m["checks_log"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            base = set(m.get("base_failures") or [])
            lines = []
            for r in results:
                if r.passed or r.skipped:
                    lines.append(f"{r.kind}: {r.summary}")
                else:
                    new = [f for f in r.failures if f"{r.kind}|{f.split(' - ')[0] if r.kind == 'test' else f}" not in base]
                    lines.append(f"{r.kind}: {r.summary}" + ("".join(f"\n    - {short(f, 160)}" for f in new[:4]) if new else " (all pre-existing)"))
            with st.lock:
                st.pending = f"[Branchpoint checks after {calls} tool calls]\n" + "\n".join(lines)
        except Exception as exc:
            with st.lock:
                st.pending = f"[Branchpoint checks could not run: {short(exc, 120)}]"
        finally:
            with st.lock:
                st.running = False

    # ------------------------------------------------------------------ transform_tool_result
    def transform_tool_result(self, tool_name: str = "", result: Any = None, **_kw: Any) -> str | None:
        marker = self._marker()
        if not marker or not isinstance(result, str):
            return None
        st = self._state(marker)
        with st.lock:
            note, st.pending = st.pending, ""
        return f"{result}\n\n{note}" if note else None

    # ------------------------------------------------------------------ post_api_request
    def post_api_request(self, usage: Any = None, **_kw: Any) -> None:
        marker = self._marker()
        if not marker:
            return
        n = tokens_from_usage(usage)
        if n:
            st = self._state(marker)
            with st.lock:
                st.tokens += n
            self._log(marker, {"type": "usage", "tokens": n})
