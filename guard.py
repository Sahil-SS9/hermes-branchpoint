"""Prove this plugin's hooks are live in the child that builds a route, before it can write anything.

A route is built by a separate ``hermes chat`` child. What confines that child to its worktree, caps its
tool calls and tokens, records ``tool_events.jsonl``/``checks_log.jsonl`` and feeds the project's check
results back to the model is *this plugin's hooks, loaded in that child*. A child started without them is
an ordinary Hermes session with file tools: it builds unconfined, measures nothing, and a run that then
reported "checks green" would be reporting a build nobody guarded.

So every real build is preceded by a proof that the child loads these hooks. The proof has two parts:

1. ``hermes plugins list --json`` says ``branchpoint`` is enabled in the same Hermes home the child will
   use (fast, and it produces the exact command to fix it when it is not);
2. a native ``/branchpoint __guard__ <worktree>`` command writes a nonce-bound report from the
   installed hook instance, tests its confinement decision, and fingerprints the installed guard code.
   This dispatches no model call. An explicit worktree is required because Hermes resolves command cwd
   differently from model-turn cwd. Each real build must also report its own hooks from ``pre_llm_call``;
   a missing or stale report fails that route, even if its final checks pass.

This is plugin-level confinement, not an OS sandbox. Use only trusted repositories and check commands.

Branchpoint never installs or enables a plugin to make the proof pass, and there is no switch to skip it:
it says exactly what to run and stops. ``--accept-hooks`` is not part of this: that flag approves
config-declared shell hooks, not the plugin hooks a build depends on.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .llm import hermes_argv
from .protocol import write_marker
from .util import BranchpointError, git, hermes_child_env, run, short, worktree_lock

PLUGIN = "branchpoint"
PROBE_ROUTE = "hooks-probe"
PROBE_PROMPT = "/branchpoint __guard__"


def hook_fingerprint() -> str:
    """Reject a stale installed plugin rather than silently building under different guards."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("hooks.py", "protocol.py", "__init__.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class HooksNotConfirmed(BranchpointError):
    """The child that would build a route does not run this plugin's hooks, so nothing may be built."""


@dataclass(frozen=True)
class HookProof:
    """What was proven about the process that builds a route.

    ``source`` is ``"host-child"`` for a real ``hermes chat`` build (proven by handshake) and
    ``"in-process"`` for a runner that does not start one (test doubles), where there is no second process
    that could silently miss the hooks.
    """

    home: str = ""
    status: str = ""
    route: str = ""
    confinement: str = ""
    pid: int = 0
    source: str = "host-child"

    @property
    def is_child_process(self) -> bool:
        return self.source == "host-child"


# --------------------------------------------------------------------------- refusals

def _enable_help(home: str) -> str:
    return ("  hermes plugins install <path to this checkout> --enable   # if it is not installed\n"
            "  hermes plugins enable branchpoint                        # if it is installed but not enabled\n"
            f"  (equivalently `plugins.enabled: [branchpoint]` in {home}/config.yaml)")


def _refusal(home: str, state: str, detail: str = "") -> HooksNotConfirmed:
    """One refusal, with the state that caused it and the commands that fix it."""
    lines = [
        "Branchpoint stopped before building: its hooks are not loaded in the `hermes chat` child that would do the build.",
        "Those hooks are the only thing that confines a build's file writes to its route worktree, caps its tool calls and",
        "tokens, records tool_events.jsonl/checks_log.jsonl and feeds check results back. Without them the build is an",
        "ordinary, unconfined Hermes session and this run's evidence would be empty, so no route is built.",
        "",
        f"This Hermes home ({home or 'unknown'}) {state}.",
        "",
        "Fix it in that home, then run again:",
        _enable_help(home or "<hermes home>"),
        "",
        "Branchpoint does not install or enable plugins itself, and it will not build without them.",
        "To compare plans without real builds, use `--execute none`.",
    ]
    if detail:
        lines += ["", f"Detail: {detail}"]
    return HooksNotConfirmed("\n".join(lines))


# --------------------------------------------------------------------------- the handshake

def verify_handshake(path: str | Path, *, nonce: str, route: str) -> dict[str, Any]:
    """Check a build child's proof of load: right build, right process, confinement actually in force."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HooksNotConfirmed(
            "the build child never reported this plugin's hooks "
            f"({short(exc, 120)}): it ran without them, so the build was neither confined nor measured"
        ) from exc
    if not isinstance(data, dict):
        raise HooksNotConfirmed("the build child reported hooks but the report was not an object, so the build was not guarded")
    problem = ""
    if str(data.get("nonce") or "") != nonce:
        problem = "the report belongs to a different build (nonce mismatch)"
    elif str(data.get("route") or "") != route:
        problem = f"the report was for route {data.get('route')!r}, not {route!r}"
    elif data.get("confinement") != "blocked":
        problem = f"the loaded hooks did not refuse a path outside the worktree (confinement: {data.get('confinement')!r})"
    if not problem and data.get("fingerprint") != hook_fingerprint():
        problem = "the installed hook code differs from this checkout; update the installed plugin"
    if problem:
        raise HooksNotConfirmed(f"the build child reported hooks but {problem}, so the build was not guarded")
    return data


# --------------------------------------------------------------------------- the host

class HostHooks(Protocol):
    """What the guard needs from the host around one Hermes home."""

    home: str

    def plugin_status(self) -> str | None:
        """``"enabled"``, the reported status, or ``None`` when the plugin is not installed."""

    def probe(self) -> dict[str, Any]:
        """Run one real child and return its verified handshake (raises ``HooksNotConfirmed``)."""


def _plugin_entry(stdout: str) -> dict[str, Any] | None:
    """The ``branchpoint`` row of ``hermes plugins list --json`` (tolerating any banner before the JSON)."""
    start = stdout.find("[")
    if start < 0:
        raise ValueError("no JSON array in the output")
    data = json.loads(stdout[start:])
    for entry in data if isinstance(data, list) else []:
        if isinstance(entry, dict) and str(entry.get("name") or "") == PLUGIN:
            return entry
    return None


class HermesHostHooks:
    """The real host: asks the Hermes CLI, then runs one real child whose hooks must answer."""

    def __init__(self, cwd: str | Path, *, env_factory: Callable[..., dict[str, str]] = hermes_child_env,
                 argv_factory: Callable[[], list[str]] = hermes_argv, runner: Callable[..., Any] = run,
                 list_timeout: float = 180.0, probe_timeout: float = 300.0):
        self.cwd = Path(cwd)
        self.env_factory = env_factory
        self.argv_factory = argv_factory
        self.runner = runner
        self.list_timeout = list_timeout
        self.probe_timeout = probe_timeout
        self.home = str(self.env_factory().get("HERMES_HOME") or "")

    def plugin_status(self) -> str | None:
        argv = self.argv_factory() + ["plugins", "list", "--json"]
        res = self.runner(argv, self.cwd, env=self.env_factory(), timeout=self.list_timeout)
        if not res.ok:
            raise _refusal(self.home, "could not be asked which plugins it enables",
                           f"`hermes plugins list --json` exited {res.rc}: {short(res.err or res.out, 200)}")
        try:
            entry = _plugin_entry(res.out)
        except ValueError as exc:
            raise _refusal(self.home, "did not answer with a readable plugin list", short(exc, 200)) from exc
        return None if entry is None else str(entry.get("status") or "")

    def probe(self) -> dict[str, Any]:
        """Dispatch a native command in the same home: no model, file tools or build are invoked."""
        tmp = Path(tempfile.mkdtemp(prefix="bp-hooks-probe-"))
        wt = tmp / "wt"
        handshake = tmp / "hook-handshake.json"
        nonce = secrets.token_hex(16)
        try:
            try:
                with worktree_lock(self.cwd):
                    git(["worktree", "add", "--detach", str(wt), "HEAD"], self.cwd, timeout=120)
            except BranchpointError as exc:
                raise _refusal(self.home, "could not be probed (the probe worktree could not be created)",
                               short(exc, 200)) from exc
            write_marker(wt, {
                "route": PROBE_ROUTE, "run_id": PROBE_ROUTE,
                "strategy": {"axis": "hooks-proof", "thesis": "prove this plugin's hooks are loaded", "key_decisions": []},
                "confine_root": str(wt), "worktree": str(wt),
                "events_path": str(tmp / "tool_events.jsonl"), "checks_log": str(tmp / "checks_log.jsonl"),
                "check_every": 1000, "max_tool_calls": 1, "token_cap": None,
                "checks": [], "base_failures": [],
                "handshake": {"path": str(handshake), "nonce": nonce},
                "probe_escape": str(tmp / "outside-the-worktree.txt"),
            })
            prompt_file = tmp / "probe-prompt.txt"
            prompt_file.write_text(PROBE_PROMPT + " " + shlex.quote(str(wt)), encoding="utf-8")
            # Same shape as a real build child (same env, same file toolset, same quiet query mode). No
            # ``--accept-hooks``: that flag approves config-declared shell hooks, not plugin hooks.
            argv = self.argv_factory() + [
                "chat", "-Q", "--query-file", str(prompt_file), "--in", str(wt), "-t", "file",
                "--max-turns", "1", "--run-budget", "120", "--ignore-rules", "--source", "tool",
            ]
            res = self.runner(argv, wt, env=self.env_factory(wt), timeout=self.probe_timeout)
            try:
                return verify_handshake(handshake, nonce=nonce, route=PROBE_ROUTE)
            except HooksNotConfirmed as exc:
                raise _refusal(self.home, "has `branchpoint` enabled, but the build child never reported its hooks",
                               f"{exc}; the probe child exited {res.rc}: {short(res.err or res.out, 300)}") from exc
        finally:
            with worktree_lock(self.cwd):
                git(["worktree", "remove", "--force", str(wt)], self.cwd, check=False)
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- the guards

class HookGuard:
    """One proof per run, taken before the first real build in it (routes build in parallel)."""

    def __init__(self, host_factory: Callable[[Path], HostHooks] | None = None):
        self._host_factory = host_factory or (lambda cwd: HermesHostHooks(cwd))
        self._proof: HookProof | None = None
        self._lock = threading.Lock()

    def confirm(self, repo: str | Path) -> HookProof:
        """Prove the build child loads the hooks, or raise ``HooksNotConfirmed``."""
        with self._lock:
            if self._proof is not None:
                return self._proof
            host = self._host_factory(Path(repo))
            status = host.plugin_status()
            if status is None:
                raise _refusal(host.home, "has no `branchpoint` plugin installed")
            if status != "enabled":
                raise _refusal(host.home, f"has `branchpoint` installed but not enabled (status: {status!r})")
            data = host.probe()
            self._proof = HookProof(
                home=host.home, status=status, route=str(data.get("route") or ""),
                confinement=str(data.get("confinement") or ""), pid=int(data.get("pid") or 0),
            )
            return self._proof


class InProcessGuard:
    """For runners that do not start a host child: a build in this process is written by the caller itself.

    There is no second process that could silently miss the hooks, so there is nothing to prove. Real builds
    always go through ``HermesAgentRunner``, which uses ``HookGuard``.
    """

    def confirm(self, repo: str | Path) -> HookProof:
        return HookProof(home="", status="in-process", pid=os.getpid(), source="in-process")
