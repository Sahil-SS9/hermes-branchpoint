"""Runs the headless build agent for one route.

``HermesAgentRunner`` starts ``hermes chat -Q -q`` in the route's worktree with **only the file toolset**
(read_file, write_file, patch, search_files). It gets no shell and no network tools, and it does not
inherit YOLO / gateway / non-interactive approval state: nothing runs that nobody approved. The plugin
(not the agent) runs the project's tests, lint and type checks.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .guard import HookGuard, HookProof, InProcessGuard
from .llm import hermes_argv
from .util import hermes_child_env

_SESSION_RE = re.compile(r"session_id:\s*(\S+)\s*$")


@dataclass
class AgentJob:
    route_id: str
    worktree: Path
    prompt: str
    model: str | None = None
    max_turns: int = 40
    wall_seconds: int = 900
    log_path: Path | None = None
    child_rules: bool = False


@dataclass
class AgentResult:
    rc: int
    final_text: str = ""
    session_id: str = ""
    seconds: float = 0.0
    timed_out: bool = False
    aborted: bool = False
    error: str = ""


class AgentRunner(Protocol):
    def run(self, job: AgentJob, abort: threading.Event) -> AgentResult: ...

    def confirm_hooks(self, repo: Path) -> HookProof:
        """Proof that the process this runner builds with loads Branchpoint's hooks (``guard.py``)."""
        ...


child_env = hermes_child_env


class HermesAgentRunner:
    """A real ``hermes chat`` child per route, behind a proof that the child loads this plugin's hooks.

    The hooks are what confine the build to its worktree and measure it, so ``confirm_hooks`` runs before a
    single child is started: a runner that could not prove them must not build at all.
    """

    def __init__(self, guard: HookGuard | None = None):
        self.guard = guard or HookGuard()

    def confirm_hooks(self, repo: Path) -> HookProof:
        return self.guard.confirm(repo)

    def run(self, job: AgentJob, abort: threading.Event) -> AgentResult:
        t0 = time.monotonic()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(job.prompt)
            qf = fh.name
        argv = hermes_argv() + [
            "chat", "-Q", "--query-file", qf, "--in", str(job.worktree), "-t", "file",
            "--max-turns", str(job.max_turns), "--run-budget", str(max(60, job.wall_seconds - 30)), "--source", "tool",
        ]
        if not job.child_rules:
            argv.append("--ignore-rules")
        if job.model:
            argv += ["-m", job.model]
        log = job.log_path
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(log, "wb") if log else subprocess.DEVNULL
        timed_out = aborted = False
        try:
            proc = subprocess.Popen(argv, cwd=str(job.worktree), env=child_env(job.worktree), stdout=out_fh,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            return AgentResult(127, error=f"could not start hermes: {exc}", seconds=time.monotonic() - t0)
        try:
            while proc.poll() is None:
                if abort.is_set():
                    aborted = True
                    break
                if time.monotonic() - t0 > job.wall_seconds:
                    timed_out = True
                    break
                time.sleep(0.4)
            if proc.poll() is None:
                self._kill(proc)
        finally:
            if log and hasattr(out_fh, "close"):
                out_fh.close()  # type: ignore[union-attr]
            try:
                os.unlink(qf)
            except OSError:
                pass
        text = ""
        if log and log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
        sid = ""
        m = _SESSION_RE.search(text.strip())
        if m:
            sid = m.group(1)
            text = text[: text.rfind("session_id:")].strip()
        return AgentResult(proc.returncode if proc.returncode is not None else -1, text.strip(), sid,
                           time.monotonic() - t0, timed_out, aborted)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                proc.wait(timeout=8)
                return
            except subprocess.TimeoutExpired:
                continue


class CallableRunner:
    """Test double: ``fn(job) -> AgentResult`` may edit ``job.worktree`` directly.

    A build here happens in this process, so there is no child that could load the hooks and nothing to prove
    (``InProcessGuard``). Real builds use ``HermesAgentRunner``, whose builds are proven first.
    """

    def __init__(self, fn: Callable[[AgentJob], AgentResult | str | None]):
        self.fn = fn
        self.guard = InProcessGuard()

    def confirm_hooks(self, repo: Path) -> HookProof:
        return self.guard.confirm(repo)

    def run(self, job: AgentJob, abort: threading.Event) -> AgentResult:
        t0 = time.monotonic()
        res = self.fn(job)
        if isinstance(res, AgentResult):
            return res
        return AgentResult(0, res or "", "", time.monotonic() - t0)
