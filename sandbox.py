"""An OS sandbox for every command that runs a route's own code.

A route's project checks (pytest, ``npm test``, ruff, mypy) execute code that lives inside the route
worktree — and the build agent, which has file tools, wrote that code. So the checks are the first
place where "file tools only" stops bounding what a route can do: a ``conftest.py``, a ``package.json``
test script, a ``pytest.ini`` plugin line or a ``sitecustomize.py`` is enough to run arbitrary code with
the caller's privileges, network and environment.

Branchpoint's confinement therefore cannot be a property of the harness's hooks. This module owns it
instead: checks run inside an OS sandbox with the worktree writable, everything else read-only, the
home directory masked and no network. One implementation, whichever harness launched the build.

Two rules make this honest:

* **The sandbox is probed, not assumed.** ``detect()`` runs a positive control and three negative ones
  in the exact profile a check will get. A host where ``bwrap`` exists but cannot build a namespace
  (no user namespaces, a restricted container) fails the positive control and is treated as absent.
* **No sandbox means no build.** ``require()`` raises; ``commands`` turns that into a refusal unless the
  user passed ``--unsafe-checks``, which is recorded in the Decision Record.

This is defence in depth, not proof of immunity: it bounds a route's code to the worktree and cuts its
network, and it does not claim to stop kernel bugs.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .util import BranchpointError, run

# Environment a check is allowed to see. An allowlist, never a denylist: provider keys, cloud
# credentials, SSH agents and every other inherited variable stay outside the sandbox.
ENV_ALLOWLIST = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR")

# Directories masked inside the sandbox even when the home directory is not: credential stores whose
# contents would otherwise be readable through the read-only root bind.
SECRET_DIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config/gh", ".config/gcloud",
    ".config/hermes", ".hermes", ".password-store", ".terraform.d", ".git-credentials",
)


class SandboxUnavailable(BranchpointError):
    """No OS sandbox is usable on this host, so a route's code cannot be run safely."""


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    backend: str
    detail: str = ""


def _sandbox_home(root: Path) -> Path:
    """A writable home inside the sandbox, outside the worktree so it never lands in the diff."""
    return Path(tempfile.mkdtemp(prefix="bp-sandbox-home-"))


class Sandbox:
    """One probed sandbox backend. ``wrap`` returns the argv to execute."""

    name = "none"

    def wrap(self, argv: Sequence[str], root: str | Path, *, network: bool, home: str | Path,
             extra_env: Mapping[str, str] | None = None) -> list[str]:
        raise NotImplementedError

    def env(self, root: str | Path, *, home: str | Path, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
        """The allowlisted environment for a sandboxed command (HOME points inside the sandbox)."""
        env = {k: os.environ[k] for k in ENV_ALLOWLIST if os.environ.get(k)}
        env["HOME"] = str(home)
        env["TMPDIR"] = str(home)
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CI"] = "1"
        env.update(extra_env or {})
        return env

    def probe(self, root: str | Path) -> ProbeResult:
        raise NotImplementedError


class NoSandbox(Sandbox):
    name = "none"

    def wrap(self, argv: Sequence[str], root, *, network, home, extra_env=None) -> list[str]:
        return list(argv)

    def probe(self, root) -> ProbeResult:
        return ProbeResult(False, self.name, "no OS sandbox backend is available")


class BwrapSandbox(Sandbox):
    """bubblewrap: unprivileged namespaces, no network, whole root read-only except the worktree."""

    name = "bwrap"

    def wrap(self, argv: Sequence[str], root: str | Path, *, network: bool, home: str | Path,
             extra_env: Mapping[str, str] | None = None) -> list[str]:
        root = Path(root).resolve()
        home = Path(home).resolve()
        cmd = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--ro-bind", "/", "/",
            # Mask the caller's home (and any credential dir outside it) before re-exposing the worktree,
            # so a route reads an empty directory instead of the user's keys. Later binds win.
            "--tmpfs", str(home),
            "--tmpfs", "/tmp",
        ]
        for rel in SECRET_DIRS:
            candidate = home / rel
            if candidate.exists():
                cmd += ["--tmpfs", str(candidate)]
        cmd += ["--bind", str(root), str(root), "--dev", "/dev", "--proc", "/proc"]
        if not network:
            cmd += ["--unshare-net"]
        cmd += ["--clearenv"]
        for key, value in (extra_env or {}).items():
            cmd += ["--setenv", str(key), str(value)]
        cmd += [
            "--setenv", "HOME", str(home),
            "--setenv", "TMPDIR", str(home),
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "CI", "1",
            "--chdir", str(root),
            "--",
            *argv,
        ]
        return cmd

    def probe(self, root: str | Path) -> ProbeResult:
        """Positive control + three denials, in exactly the profile a check gets."""
        if not shutil.which("bwrap"):
            return ProbeResult(False, self.name, "bwrap is not installed")
        root = Path(root).resolve()
        home = _sandbox_home(root)
        planted = home / "planted-secret.txt"
        marker = "BRANCHPOINT-PROBE-SECRET"
        try:
            home.mkdir(parents=True, exist_ok=True)
            planted.write_text(marker, encoding="utf-8")
            script = (
                # 0: the sandbox must actually run something (a broken backend passes denials for free)
                "echo CONTROL-OK; "
                # 1: no write outside the worktree
                f"touch {str(home / 'escape.txt')!s} 2>/dev/null && echo WRITE-LEAKED || echo write-blocked; "
                # 2: no read of a credential file that exists outside the worktree
                f"cat {str(planted)!s} 2>/dev/null | grep -q {marker} && echo READ-LEAKED || echo read-blocked; "
                # 3: no network egress
                "python3 -c \"import socket,sys; socket.setdefaulttimeout(2)\n"
                "try:\n socket.create_connection(('1.1.1.1',443)); print('NET-LEAKED')\n"
                "except Exception:\n print('net-blocked')\" 2>/dev/null || echo net-blocked"
            )
            argv = self.wrap(["/bin/sh", "-c", script], root, network=False, home=home)
            res = run(argv, root, env=self.env(root, home=home), timeout=90)
            out = res.out + res.err
            problems = []
            if "CONTROL-OK" not in out:
                problems.append(f"the sandbox could not run a command ({res.err.strip().splitlines()[-1][:160] if res.err.strip() else 'no output'})")
            if "WRITE-LEAKED" in out:
                problems.append("a write outside the worktree succeeded")
            if "READ-LEAKED" in out:
                problems.append("a credential file outside the worktree was readable")
            if "NET-LEAKED" in out:
                problems.append("network egress succeeded")
            if problems:
                return ProbeResult(False, self.name, "; ".join(problems))
            return ProbeResult(True, self.name, "write, read and network denied; command ran")
        except Exception as exc:                        # noqa: BLE001 - a probe never crashes a build
            return ProbeResult(False, self.name, f"probe failed: {exc}")
        finally:
            shutil.rmtree(home, ignore_errors=True)


class SeatbeltSandbox(Sandbox):
    """macOS ``sandbox-exec``: deny writes outside the worktree and deny network."""

    name = "sandbox-exec"

    _PROFILE = """(version 1)
(deny default)
(allow process*)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(allow file-write* (subpath "{root}") (subpath "{home}") (subpath "/private/var/folders"))
(deny network*)
"""

    def wrap(self, argv: Sequence[str], root, *, network, home, extra_env=None) -> list[str]:
        root = Path(root).resolve()
        home = Path(home).resolve()
        profile = self._PROFILE.format(root=root, home=home)
        if network:
            profile = profile.replace("(deny network*)\n", "")
        return ["sandbox-exec", "-p", profile, *argv]

    def probe(self, root) -> ProbeResult:
        if not shutil.which("sandbox-exec"):
            return ProbeResult(False, self.name, "sandbox-exec is not installed")
        root = Path(root).resolve()
        home = _sandbox_home(root)
        planted = home / "planted-secret.txt"
        try:
            planted.write_text("BRANCHPOINT-PROBE-SECRET", encoding="utf-8")
            script = (
                "echo CONTROL-OK; "
                f"touch {str(home / 'escape.txt')!s} 2>/dev/null && echo WRITE-LEAKED || echo write-blocked; "
                "python3 -c \"import socket; socket.setdefaulttimeout(2)\n"
                "try:\n socket.create_connection(('1.1.1.1',443)); print('NET-LEAKED')\n"
                "except Exception:\n print('net-blocked')\" 2>/dev/null || echo net-blocked"
            )
            res = run(self.wrap(["/bin/sh", "-c", script], root, network=False, home=home), root,
                      env=self.env(root, home=home), timeout=90)
            out = res.out + res.err
            problems = [label for token, label in (("WRITE-LEAKED", "a write outside the worktree succeeded"),
                                                   ("NET-LEAKED", "network egress succeeded")) if token in out]
            if "CONTROL-OK" not in out:
                problems.insert(0, "the sandbox could not run a command")
            return ProbeResult(not problems, self.name, "; ".join(problems) or "write and network denied")
        except Exception as exc:                        # noqa: BLE001
            return ProbeResult(False, self.name, f"probe failed: {exc}")
        finally:
            shutil.rmtree(home, ignore_errors=True)


_BACKENDS = (BwrapSandbox, SeatbeltSandbox)
_lock = threading.Lock()
_detected: ProbeResult | None = None


def detect(root: str | Path, *, refresh: bool = False) -> ProbeResult:
    """Probe each backend once per process and remember the verdict."""
    global _detected
    with _lock:
        if _detected is not None and not refresh:
            return _detected
        detail: list[str] = []
        for backend in _BACKENDS:
            result = backend().probe(root)
            if result.ok:
                _detected = result
                return result
            detail.append(f"{result.backend}: {result.detail}")
        _detected = ProbeResult(False, "none", "; ".join(detail) or "no backend available")
        return _detected


def get(root: str | Path) -> Sandbox | None:
    """The probed backend, or None when this host cannot sandbox a command."""
    result = detect(root)
    if not result.ok:
        return None
    for backend in _BACKENDS:
        if backend.name == result.backend:
            return backend()
    return None


def refusal(root: str | Path) -> SandboxUnavailable:
    """The error raised when a build needs a sandbox and this host has none."""
    result = detect(root)
    return SandboxUnavailable(
        "Branchpoint stopped: no OS sandbox is available on this host, so a route's project checks\n"
        "could not be run safely. Checks execute code that lives in the route worktree — code the build\n"
        "agent wrote — so without a sandbox a route can run arbitrary commands as you.\n"
        "\n"
        f"Probe: {result.detail}\n"
        "\n"
        "Fix it by installing bubblewrap (Debian/Ubuntu: `apt install bubblewrap`; Fedora: `dnf install\n"
        "bubblewrap`) or, on macOS, by using the built-in sandbox-exec. Then run again.\n"
        "\n"
        "To accept the risk for one run, pass `--unsafe-checks`: checks then run unsandboxed with an\n"
        "allowlisted environment, and the Decision Record records that they did.\n"
        "To compare plans with no checks at all, use `--execute none`."
    )


def check_argv(spec_argv: Sequence[str], root: str | Path, *, network: bool = False,
               extra_env: Mapping[str, str] | None = None) -> tuple[list[str], dict[str, str], str] | None:
    """Wrap one check command. Returns ``(argv, env, home)`` or None when no sandbox is available."""
    box = get(root)
    if box is None:
        return None
    home = _sandbox_home(root)
    return box.wrap(spec_argv, root, network=network, home=home, extra_env=extra_env), \
        box.env(root, home=home, extra_env=extra_env), str(home)
