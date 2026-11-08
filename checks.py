"""Tests / lint / type checks, run by Branchpoint (never by the build agent) inside a route worktree.

The build agent only has file tools, so these are the only commands a route executes. They are the
project's own checks, started by the plugin the same way the user would start them.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import clean_env, run, short, tail


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

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cmd": self.cmd, "rc": self.rc, "seconds": round(self.seconds, 2),
                "passed": self.passed, "summary": self.summary, "failures": self.failures[:20]}


def find_python(root: Path) -> str:
    for cand in (root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return shutil.which("python3") or sys.executable


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
            specs.append(CheckSpec("test", [python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "--maxfail=25"], label="auto:pytest"))
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
            res = run([python, "-m", "mypy", "--version"], root, timeout=30)
            if res.ok:
                specs.append(CheckSpec("type", [python, "-m", "mypy", "."], label="auto:mypy"))
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


def run_check(root: str | Path, spec: CheckSpec, extra_env: dict[str, str] | None = None) -> CheckResult:
    res = run(spec.argv, root, env=clean_env(extra_env), timeout=spec.timeout)
    summary, failures = parse_output(spec, res.rc, res.out, res.err)
    passed = res.ok and not (spec.kind != "test" and failures)
    if res.timed_out:
        summary, passed = f"timed out after {spec.timeout:.0f}s", False
        failures = failures or [summary]
    if res.rc == 127:
        summary, passed = f"command not found: {spec.argv[0]}", False
        failures = failures or [summary]
    return CheckResult(spec.kind, spec.cmd, res.rc, res.seconds, passed, summary, failures,
                       output=tail((res.out + ("\n" + res.err if res.err else "")), 5000), timed_out=res.timed_out)


def run_checks(root: str | Path, specs: list[CheckSpec], kinds: tuple[str, ...] | None = None) -> list[CheckResult]:
    return [run_check(root, s) for s in specs if kinds is None or s.kind in kinds]
