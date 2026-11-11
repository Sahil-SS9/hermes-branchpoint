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
from pathlib import Path
from typing import Any, Callable

from . import anchor as anchormod
from .checks import CheckSpec, run_check
from .paths import data_dir
from .protocol import (ALLOWED_TOOLS, WRITE_TOOLS, PatchUnparsed, extract_paths, is_protected, outside_root,
                       read_marker, tokens_from_usage, unknown_keys)
from .util import atomic_write, in_skipped_dir, is_secret_path, repo_root, short, now


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
        self._handshaken: set[str] = set()
        self._extra: dict[str, tuple[str, ...]] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _marker(self) -> dict[str, Any] | None:
        """The route marker for the session running this hook, else None.

        Cached per cwd for the life of the process on purpose: the marker lives in the worktree's private git
        dir, which the agent can rewrite, so re-reading it every call would let a later rewrite change the
        confinement root mid-build. A cached marker is the one that was proven before the first tool call.

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

    def _extra_args(self) -> dict[str, tuple[str, ...]]:
        """Extra argument keys the guard should accept, from the declared ``guard_extra_args`` setting.

        Format: ``tool=key,tool=key`` (for example ``write_file=content_sha``). Only needed when a host adds an
        argument to a file tool: without this the guard refuses the call rather than letting an unchecked key through.
        """
        if self._extra is None:
            self._extra = {}
            for entry in str(self._settings("guard_extra_args", "") or "").split(","):
                tool, _, key = entry.partition("=")
                tool, key = tool.strip(), key.strip()
                if tool and key:
                    self._extra[tool] = (*self._extra.get(tool, ()), key)
        return self._extra

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

    def _report_loaded(self, marker: dict[str, Any], *, reported_by: str = "pre_llm_call") -> None:
        """Tell the parent that these hooks are live in *this* process, before the model can ask for a tool.

        The native preflight reports the installed code before a build starts. The actual child reports
        again from its earliest model-turn hook. Both exercise the same confinement decision on an outside
        path; this is plugin-level verification, not an OS sandbox or a claim of immunity to host changes.
        """
        target = marker.get("handshake") or {}
        path, nonce = str(target.get("path") or ""), str(target.get("nonce") or "")
        if not path or not nonce:
            return
        with self._lock:
            if path in self._handshaken:
                return
            self._handshaken.add(path)
        root = marker.get("confine_root") or marker.get("worktree") or ""
        from .guard import hook_fingerprint
        report: dict[str, Any] = {
            "fingerprint": hook_fingerprint(),
            "nonce": nonce, "route": str(marker.get("route") or ""), "run_id": str(marker.get("run_id") or ""),
            "pid": os.getpid(), "confine_root": str(root), "reported_by": reported_by,
            # every hook this plugin registers, so a partially loaded plugin is visible rather than assumed
            "hooks": ["pre_llm_call", "pre_tool_call", "post_tool_call", "transform_tool_result", "post_api_request"],
        }
        escape = marker.get("probe_escape")
        if escape:
            verdict = self._guard_tool(marker, "write_file", {"path": str(escape), "content": "probe"})
            report["confinement"] = "blocked" if verdict and verdict.get("action") == "block" and "outside" in verdict.get("message", "") else "unconfined"
        try:
            atomic_write(Path(path), json.dumps(report, default=str) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ pre_llm_call
    def pre_llm_call(self, session_id: str = "", user_message: str = "", conversation_history: list | None = None,
                     **_kw: Any) -> dict[str, str] | None:
        marker = self._marker()
        if marker:
            self._report_loaded(marker)
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
        return self._guard_tool(marker, tool_name, args)

    def _guard_tool(self, marker: dict[str, Any], tool_name: str, args: dict | None) -> dict[str, Any] | None:
        """One confinement decision, shared by the live hook and the native preflight.

        Fail-closed: a call is allowed only when this method can fully account for it — a known tool, every
        argument key named by that tool's schema, every path it names inside the worktree and outside the
        protected set, and a patch body whose headers parse. Anything it cannot read is refused, because an
        argument the guard cannot parse is an argument it cannot confine.
        """
        st = self._state(marker)
        reason = ""
        root = str(marker.get("confine_root") or marker.get("worktree") or "")
        protected_extra = tuple(marker.get("protected_paths") or ())
        if tool_name not in ALLOWED_TOOLS:
            reason = f"'{tool_name}' is not available inside a Branchpoint route (file tools only)."
        elif not root:
            return {"action": "block", "message": "Branchpoint: route has no confinement root."}
        else:
            unknown = unknown_keys(tool_name, args, self._extra_args().get(tool_name, ()))
            if unknown:
                reason = ("this call carries argument(s) the route guard cannot read "
                          f"({', '.join(sorted(unknown))}), so it cannot be confined.")
            else:
                try:
                    paths = extract_paths(tool_name, args)
                except PatchUnparsed as exc:
                    paths, reason = [], f"{exc}."
                bad = outside_root(paths, root, root) if not reason else None
                if reason:
                    pass          # already refused: an unparsed patch cannot be confined
                elif bad:
                    reason = f"path '{short(bad, 80)}' is outside this route's worktree. Use paths relative to the current directory."
                else:
                    # Check the path *and* what it resolves to: a symlink the agent created (``innocent.txt`` ->
                    # ``.env``) would otherwise read or write a secret or a protected git file through a safe name.
                    resolved = [(Path(p) if Path(p).is_absolute() else Path(root) / p).resolve() for p in paths]
                    protected = next((p for p in [*paths, *resolved] if is_protected(str(p), protected_extra)), None)
                    if protected:
                        reason = (f"'{short(protected, 80)}' is a protected path: it is git or harness configuration, "
                                  "so editing it here would change what runs, not what the code does. "
                                  "Land it by hand if the change is genuinely wanted.")
                    elif any(is_secret_path(str(p)) for p in [*paths, *resolved]):
                        reason = "reading or writing credential/secret files is not allowed in a route."
                    elif not paths and tool_name in WRITE_TOOLS:
                        reason = "a write with no readable target path cannot be confined."
                    elif tool_name == "search_files" and (args or {}).get("target", "content") != "files":
                        search = Path((args or {}).get("path") or ".")
                        search = search if search.is_absolute() else Path(root) / search
                        # The host search tool can return matching content from tracked secret files.
                        # Do not rely on .gitignore or on a model-supplied glob to protect them.
                        count = 0
                        for folder, dirs, names in os.walk(search):
                            dirs[:] = [d for d in dirs if not in_skipped_dir(str(Path(d) / "_"))]
                            count += len(names)
                            if any(is_secret_path(str(Path(folder) / n)) for n in names):
                                reason = "directory content search could expose secret files; search specific non-secret files instead."
                                break
                            if count > 10000:
                                reason = "secret-file search safety scan is too large; search a smaller directory or specific file."
                                break
        if not reason and st.over_budget:
            reason = f"{st.over_budget}. Stop and reply with your 3-sentence summary."
        if reason:
            try:
                first = (extract_paths(tool_name, args) or [""])[0]
            except PatchUnparsed:
                first = ""
            self._log(marker, {"tool": tool_name, "blocked": reason, "path": first})
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
                results.append(run_check(m.get("confine_root") or m["worktree"], spec,
                                         unsafe=bool(m.get("unsafe_checks"))))
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
