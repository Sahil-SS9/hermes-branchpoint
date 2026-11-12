"""Tests / lint / type checks, run by Branchpoint (never by a shell the build agent controls).

The build agent has file tools only, so these are the only commands a route executes — but they execute
code *from inside the route worktree*, and the agent wrote that code. A ``conftest.py``, a ``package.json``
test script, a ``pytest.ini`` plugin line or a ``sitecustomize.py`` is enough to run arbitrary commands at
the next check. So every check runs inside the OS sandbox from ``sandbox.py``: worktree writable,
everything else read-only, the home directory masked, no network, and an allowlisted environment. When no
sandbox is available, ``run_check`` refuses rather than running the code unsandboxed.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import protocol, sandbox
from .util import allow_env, clean_env, run, short, tail


@dataclass
class CheckSpec:
    kind: str                # test | lint | type
    argv: list[str]
    timeout: float = 240
    label: str = ""

    @property
    def cmd(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


@dataclass
class CheckResult:
    kind: str
    cmd: str
    rc: int
    seconds: float
    passed: bool
    summary: str
    failures: list[str] = field(default_factory=list)
    output: str = ""
    timed_out: bool = False
    skipped: bool = False          # the check could not run (missing tool/interpreter): no evidence either way

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cmd": self.cmd, "rc": self.rc, "seconds": round(self.seconds, 2),
                "passed": self.passed, "skipped": self.skipped, "summary": self.summary, "failures": self.failures[:20]}


_PROBES: dict[tuple[str, str], bool] = {}


def python_has(py: str, module: str) -> bool:
    key = (py, module)
    if key not in _PROBES:
        _PROBES[key] = run([py, "-c", f"import {module}"], Path(py).parent if Path(py).is_absolute() else ".", env=clean_env(), timeout=30).ok
    return _PROBES[key]


def find_python(root: Path, module: str | None = None) -> str:
    """The project's interpreter: its own venv first, then PATH, then the host's own.

    A candidate inside ``root`` is trusted only when it *resolves* (symlink and all) outside ``root``: a route
    worktree holds files the build agent wrote, so ``<wt>/.venv/bin/python`` could be its own file rather than
    the user's linked venv. With ``module``, the first interpreter that can import it wins (the host's own
    interpreter usually has no pytest)."""
    cands: list[str] = []
    # A *linked worktree* holds files the build agent wrote, so an interpreter that is a real file inside it is
    # the agent's, not the user's linked venv. The user's own repo is different: a `.venv/bin/python` there is
    # exactly what should run its checks.
    linked_worktree = protocol.gitdir_of(root) is not None
    for cand in (root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python"):
        if cand.exists():
            if linked_worktree:
                try:
                    inside = cand.resolve().is_relative_to(Path(root).resolve())
                except (OSError, RuntimeError):
                    inside = True
                if inside and not cand.is_symlink():
                    continue
            cands.append(str(cand))
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            cands.append(found)
    cands.append(sys.executable)
    cands = list(dict.fromkeys(cands))
    if module:
        for c in cands:
            if python_has(c, module):
                return c
    return cands[0]


def _pyproject(root: Path) -> dict[str, Any]:
    try:
        return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _split(cmd: str, python: str) -> list[str]:
    return [python if a == "{python}" else a for a in shlex.split(cmd)]


def detect_checks(root: str | Path, settings: dict[str, Any] | None = None) -> list[CheckSpec]:
    """Configured commands win; otherwise detect conservatively from the project files."""
    root = Path(root)
    settings = settings or {}
    python = find_python(root)
    pytest_python = find_python(root, "pytest")
    specs: list[CheckSpec] = []
    have: set[str] = set()   # kinds that are configured, or explicitly switched off
    for kind, key in (("test", "test_command"), ("lint", "lint_command"), ("type", "type_command")):
        cmd = str(settings.get(key) or "").strip()
        if cmd.lower() in ("none", "off", "-"):
            have.add(kind)
        elif cmd:
            have.add(kind)
            specs.append(CheckSpec(kind, _split(cmd, python), label=key))
    pp = _pyproject(root)
    tool = pp.get("tool") or {}
    if "test" not in have:
        if (root / "pytest.ini").exists() or "pytest" in tool or (root / "tests").is_dir() or (root / "conftest.py").exists():
            specs.append(CheckSpec("test", [pytest_python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "--maxfail=25"], label="auto:pytest"))
        elif (root / "package.json").exists():
            try:
                scripts = json.loads((root / "package.json").read_text(encoding="utf-8")).get("scripts") or {}
            except (OSError, ValueError):
                scripts = {}
            if scripts.get("test") and "no test specified" not in scripts["test"]:
                specs.append(CheckSpec("test", ["npm", "test", "--silent"], label="auto:npm test"))
    if "lint" not in have:
        if (root / "ruff.toml").exists() or (root / ".ruff.toml").exists() or "ruff" in tool:
            ruff = shutil.which("ruff")
            if ruff:
                specs.append(CheckSpec("lint", [ruff, "check", "--output-format", "concise", "."], label="auto:ruff"))
        elif (root / "package.json").exists():
            try:
                scripts = json.loads((root / "package.json").read_text(encoding="utf-8")).get("scripts") or {}
            except (OSError, ValueError):
                scripts = {}
            if scripts.get("lint"):
                specs.append(CheckSpec("lint", ["npm", "run", "lint", "--silent"], label="auto:npm lint"))
    if "type" not in have:
        if (root / "mypy.ini").exists() or "mypy" in tool:
            # No `python -m mypy --version` probe here: with the worktree as cwd, `-m` would import an
            # agent-written `mypy.py` on the host with the full environment (0.2.2 review). The interpreter is
            # chosen by importing mypy from outside the tree; the sandboxed run reports it missing if no
            # interpreter has it.
            specs.append(CheckSpec("type", [find_python(root, "mypy"), "-m", "mypy", "."], label="auto:mypy"))
        elif (root / "tsconfig.json").exists() and (root / "node_modules" / ".bin" / "tsc").exists():
            specs.append(CheckSpec("type", [str(root / "node_modules" / ".bin" / "tsc"), "--noEmit"], label="auto:tsc"))
    return specs


_PYTEST_FAIL_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$", re.M)
_PYTEST_SUM_RE = re.compile(r"(\d+)\s+(failed|passed|error|errors|skipped|xfailed|xpassed|deselected)", re.I)
_RUFF_RE = re.compile(r"^(\S+?):(\d+):(\d+):\s+([A-Z]+\d+\S*)\s+(.*)$", re.M)
_MYPY_RE = re.compile(r"^(\S+?):(\d+):(?:\d+:)?\s+error:\s+(.*)$", re.M)
_TSC_RE = re.compile(r"^(\S+?)\((\d+),\d+\):\s+error\s+(TS\d+):\s+(.*)$", re.M)


def parse_output(spec: CheckSpec, rc: int, out: str, err: str) -> tuple[str, list[str]]:
    text = out + ("\n" + err if err else "")
    failures: list[str] = []
    if spec.kind == "test":
        for m in _PYTEST_FAIL_RE.finditer(text):
            failures.append(f"{m.group(2)}" + (f" - {short(m.group(3), 140)}" if m.group(3) else ""))
        counts = {}
        for n, word in _PYTEST_SUM_RE.findall(text.splitlines()[-1] if text.strip() else ""):
            counts[word.lower()] = int(n)
        if counts:
            summary = ", ".join(f"{v} {k}" for k, v in counts.items())
        else:
            summary = "passed" if rc == 0 else f"exit {rc}"
        if not failures and rc != 0:
            failures.append(short(tail(text, 400).strip().splitlines()[-1] if text.strip() else f"exit {rc}", 200))
        return summary, failures
    if spec.kind == "lint":
        for m in _RUFF_RE.finditer(text):
            failures.append(f"{m.group(1)}:{m.group(2)} {m.group(4)} {short(m.group(5), 120)}")
        return (f"{len(failures)} finding(s)" if failures else ("clean" if rc == 0 else f"exit {rc}")), failures or (
            [short(tail(text, 300), 200)] if rc != 0 else [])
    for m in _MYPY_RE.finditer(text):
        failures.append(f"{m.group(1)}:{m.group(2)} {short(m.group(3), 140)}")
    for m in _TSC_RE.finditer(text):
        failures.append(f"{m.group(1)}:{m.group(2)} {m.group(3)} {short(m.group(4), 120)}")
    return (f"{len(failures)} error(s)" if failures else ("clean" if rc == 0 else f"exit {rc}")), failures or (
        [short(tail(text, 300), 200)] if rc != 0 else [])


UNSAFE_ENV = "BRANCHPOINT_UNSAFE_CHECKS"


def unsafe_allowed(flag: bool = False) -> bool:
    """Whether unsandboxed checks were accepted explicitly, by flag or by the same-named env var.

    Both spellings exist so a test harness or a scripted demo can accept the risk the same way a user does with
    ``--unsafe-checks``; there is no other way past a missing sandbox.
    """
    if flag:
        return True
    return str(os.environ.get(UNSAFE_ENV) or "").strip().lower() in ("1", "true", "yes")


def run_check(root: str | Path, spec: CheckSpec, extra_env: dict[str, str] | None = None, *,
              unsafe: bool = False, network: bool = False) -> CheckResult:
    """Run one check inside the OS sandbox, or refuse when this host has none.

    ``unsafe=True`` runs it unsandboxed with an allowlisted environment, and only exists because the user
    asked for that by name (``--unsafe-checks``); the Decision Record records it.
    """
    prepared = sandbox.check_argv(spec.argv, root, network=network, extra_env=extra_env)
    home: str | None = None
    if prepared is None:
        if not unsafe_allowed(unsafe):
            raise sandbox.refusal(root)
        argv, env = list(spec.argv), allow_env(extra_env)
    else:
        argv, env, home = prepared
    try:
        res = run(argv, root, env=env, timeout=spec.timeout)
    finally:
        if home:
            shutil.rmtree(home, ignore_errors=True)
    summary, failures = parse_output(spec, res.rc, res.out, res.err)
    passed = res.ok and not (spec.kind != "test" and failures)
    if res.timed_out:
        summary, passed = f"timed out after {spec.timeout:.0f}s", False
        failures = failures or [summary]
    skipped = False
    blob = res.out + res.err
    # bwrap reports a missing command itself ("bwrap: execvp <cmd>: No such file or directory", exit 1).
    if res.rc == 127 or (prepared is not None and re.search(r"^bwrap: execvp .*: No such file or directory$", blob, re.M)):
        summary, passed, skipped = f"command not found: {spec.argv[0]}", False, True
        failures = failures or [summary]
    elif res.rc != 0 and re.search(r"No module named ['\"]?(pytest|ruff|mypy)\b", blob):
        summary, passed, skipped = "could not run: the interpreter has no " + re.search(r"No module named ['\"]?(\w+)", blob).group(1), False, True
        failures = []
    return CheckResult(spec.kind, spec.cmd, res.rc, res.seconds, passed, summary, failures,
                       output=tail((res.out + ("\n" + res.err if res.err else "")), 5000), timed_out=res.timed_out, skipped=skipped)


def run_checks(root: str | Path, specs: list[CheckSpec], kinds: tuple[str, ...] | None = None, *,
               unsafe: bool = False) -> list[CheckResult]:
    return [run_check(root, s, unsafe=unsafe) for s in specs if kinds is None or s.kind in kinds]
