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
import shlex
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import paths as _paths
from .util import BranchpointError, run

# Environment a check is allowed to see. An allowlist, never a denylist: provider keys, cloud
# credentials, SSH agents and every other inherited variable stay outside the sandbox.
ENV_ALLOWLIST = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "TMPDIR")

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


def masked_dirs() -> list[Path]:
    """The caller's real home, plus the Hermes home when it lives outside it: hidden from every check.

    Masking the sandbox's own fresh home would hide nothing (the 0.2.2 review): after ``--ro-bind / /``
    the real ``~/.hermes/.env``, ``~/.ssh`` and other profiles would still be readable, and a failing
    test's output goes back to the build model.
    """
    out: list[Path] = []
    try:
        out.append(Path.home().resolve())
    except (OSError, RuntimeError, KeyError):
        pass
    try:
        hh = _paths.hermes_home()
    except Exception:                                   # noqa: BLE001 - no Hermes means no Hermes home
        hh = None
    if hh is not None:
        hh = Path(hh).expanduser().resolve()
        if not any(hh == m or hh.is_relative_to(m) for m in out):
            out.append(hh)
    return [m for m in out if m != Path("/")]


def _venv_and_base(venv: Path) -> list[Path]:
    """A virtualenv and the interpreter install it was made from (``home =`` in ``pyvenv.cfg``)."""
    found = [venv]
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and value.strip():
                found += _install_prefix(Path(value.strip()) / "python3")
    except OSError:
        pass
    return found


def _install_prefix(exe: Path) -> list[Path]:
    """What running ``exe`` needs to see: its own install prefix, or just the file when it has none."""
    try:
        resolved = exe.resolve()
    except (OSError, RuntimeError):
        return []
    if (exe.parent.parent / "pyvenv.cfg").exists():
        return [*_venv_and_base(exe.parent.parent), resolved]
    prefix = resolved.parent.parent
    home = Path.home().resolve()
    # A real install prefix (uv/pyenv python, nvm node): bin/ next to lib/. Never the whole home or ~/.local,
    # whose share/ holds keyrings and other apps' data; for those, only the executable file itself.
    if resolved.parent.name == "bin" and (prefix / "lib").is_dir() and prefix not in (home, home / ".local"):
        return [prefix]
    return [exe, resolved]


_TMP_DIRS = (Path("/tmp"),)


def _symlink_chain(path: Path) -> list[tuple[Path, str]]:
    """Every symlink met while resolving ``path``, as ``(link, target text)``.

    Masking the home hides symlinks as well as files: a venv's ``bin/python`` points at a uv/pyenv install through
    a versionless alias (``cpython-3.12-…`` -> ``cpython-3.12.13-…``), and re-binding only the resolved directory
    leaves that alias dangling inside the sandbox.
    """
    out: list[tuple[Path, str]] = []
    todo = [Path(os.path.abspath(path))]
    seen: set[Path] = set()
    while todo and len(seen) < 64:
        p = todo.pop()
        cur = Path(p.anchor)
        for i, part in enumerate(p.parts[1:], 1):
            nxt = cur / part
            if nxt.is_symlink() and nxt not in seen:
                seen.add(nxt)
                target = os.readlink(nxt)
                out.append((nxt, target))
                follow = Path(os.path.normpath(nxt.parent / target))
                rest = p.parts[i + 1:]
                todo.append(follow.joinpath(*rest) if rest else follow)
                break
            cur = nxt
    return out


def _exposures(argv: Sequence[str], root: str | Path, masks: Sequence[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Paths under a mask that a check legitimately needs (re-bound read-only), and the symlinks leading to them.

    The route worktree links the user's ``.venv`` / ``venv`` / ``node_modules`` (``link_dirs``), and the check's
    interpreter usually lives under the home directory too. Those come back read-only; nothing else does.
    """
    root = Path(root).resolve()
    wanted: list[Path] = []
    starts: list[Path] = []
    for name in (".venv", "venv", "node_modules"):
        link = root / name
        if link.is_symlink():
            target = link.resolve()
            starts.append(link)
            wanted += _venv_and_base(target) if (target / "pyvenv.cfg").exists() else [target]
    exe = str(argv[0]) if argv else ""
    if exe and not os.path.isabs(exe):
        exe = shutil.which(exe) or ""
    if exe:
        starts.append(Path(exe))
        wanted += _install_prefix(Path(exe))
    for w in list(wanted):
        if (w / "pyvenv.cfg").exists():
            starts.append(w / "bin" / "python3")

    # bwrap also mounts an empty tmpfs over /tmp, so a project (or its linked venv) under /tmp is hidden too.
    hidden = [*masks, *(t for t in _TMP_DIRS if t not in masks)]

    def under_mask(p: Path) -> bool:
        return any(p.is_relative_to(m) for m in hidden) and p not in hidden

    dirs: list[Path] = []
    for p in wanted:
        p = Path(os.path.abspath(p))
        if p.exists() and under_mask(p) and p != root and not p.is_relative_to(root) and p not in dirs:
            dirs.append(p)
    links: list[tuple[Path, str]] = []
    for start in starts:
        for link, target in _symlink_chain(start):
            inside_bound = link.is_relative_to(root) or any(link.is_relative_to(d) for d in dirs)
            if under_mask(link) and not inside_bound and (link, target) not in links:
                links.append((link, target))
    return dirs, links


def reexposed_dirs(argv: Sequence[str], root: str | Path, masks: Sequence[Path]) -> list[Path]:
    return _exposures(argv, root, masks)[0]


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
        masks = masked_dirs()
        cmd = [
            "bwrap",
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--ro-bind", "/", "/",
        ]
        # Mask the caller's real home (and a Hermes home outside it) with empty tmpfs mounts, then bring back
        # only what a check needs. Later mounts win, so the order is: masks, read-only re-exposures (the
        # linked venv / node_modules and the interpreter's install), the sandbox's own home, the worktree.
        for mask in sorted(masks, key=lambda m: len(m.parts)):
            cmd += ["--tmpfs", str(mask)]
        cmd += ["--tmpfs", "/tmp"]
        keep_dirs, keep_links = _exposures(argv, root, masks)
        for link, target in keep_links:
            cmd += ["--symlink", target, str(link)]
        for keep in keep_dirs:
            cmd += ["--ro-bind", str(keep), str(keep)]
        cmd += ["--bind", str(home), str(home)]
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
        return _probe(self, root)


def _plant_canaries() -> tuple[list[Path], str]:
    """A secret file in a fresh folder under each masked directory: the real home, not a temp dir."""
    marker = "BRANCHPOINT-PROBE-SECRET"
    dirs: list[Path] = []
    for mask in masked_dirs():
        if not mask.is_dir():
            continue
        d = Path(tempfile.mkdtemp(dir=mask, prefix=".branchpoint-probe-"))
        (d / "planted-secret.txt").write_text(marker, encoding="utf-8")
        dirs.append(d)
    if not dirs:
        raise BranchpointError("no home directory to plant the probe's canary in")
    return dirs, marker


def _probe(box: "Sandbox", root: str | Path) -> ProbeResult:
    """Run the probe script in ``box`` and judge it. The canaries sit under the real home (the review of 0.2.2:
    a canary in the sandbox's own temp home proved nothing about ``~/.hermes/.env``)."""
    root = Path(root).resolve()
    home = _sandbox_home(root)
    canaries: list[Path] = []
    try:
        canaries, marker = _plant_canaries()
        reads = " ".join(
            f"cat {shlex.quote(str(d / 'planted-secret.txt'))} 2>/dev/null | grep -q {marker} && echo READ-LEAKED;"
            for d in canaries
        )
        writes = " ".join(f"touch {shlex.quote(str(d / 'escape.txt'))} 2>/dev/null;" for d in canaries)
        script = (
            # 0: the sandbox must actually run something (a broken backend passes denials for free)
            "echo CONTROL-OK; "
            # 1: no write outside the worktree reaches the host (checked on the host afterwards)
            f"{writes} "
            # 2: no read of a credential file under the real home
            f"{reads} echo reads-done; "
            # 3: no network egress
            "python3 -c \"import socket,sys; socket.setdefaulttimeout(2)\n"
            "try:\n socket.create_connection(('1.1.1.1',443)); print('NET-LEAKED')\n"
            "except Exception:\n print('net-blocked')\" 2>/dev/null || echo net-blocked"
        )
        argv = box.wrap(["/bin/sh", "-c", script], root, network=False, home=home)
        res = run(argv, root, env=box.env(root, home=home), timeout=90)
        out = res.out + res.err
        problems = []
        if "CONTROL-OK" not in out:
            last = res.err.strip().splitlines()[-1][:160] if res.err.strip() else "no output"
            problems.append(f"the sandbox could not run a command ({last})")
        if any((d / "escape.txt").exists() for d in canaries):
            problems.append("a write outside the worktree succeeded")
        if "READ-LEAKED" in out:
            problems.append("a credential file outside the worktree was readable")
        if "NET-LEAKED" in out:
            problems.append("network egress succeeded")
        if problems:
            return ProbeResult(False, box.name, "; ".join(problems))
        return ProbeResult(True, box.name, "write, home read and network denied; command ran")
    except Exception as exc:                        # noqa: BLE001 - a probe never crashes a build
        return ProbeResult(False, box.name, f"probe failed: {exc}")
    finally:
        shutil.rmtree(home, ignore_errors=True)
        for d in canaries:
            shutil.rmtree(d, ignore_errors=True)


class SeatbeltSandbox(Sandbox):
    """macOS ``sandbox-exec``: deny writes outside the worktree and deny network."""

    name = "sandbox-exec"

    @staticmethod
    def _q(path: Path) -> str:
        return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

    def profile(self, argv: Sequence[str], root: str | Path, *, network: bool, home: str | Path) -> str:
        root = Path(root).resolve()
        home = Path(home).resolve()
        masks = masked_dirs()
        keep = [root, home, *reexposed_dirs(argv, root, masks)]
        rules = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow file-read*)",
            # The last matching rule wins: hide the real home, then re-allow what a check needs.
            *(f"(deny file-read* (subpath {self._q(m)}))" for m in masks),
            "(allow file-read* " + " ".join(f"(subpath {self._q(k)})" for k in keep) + ")",
            # Only the worktree and the sandbox's own home: not all of /private/var/folders (every temp dir of
            # this user, other checks' homes included). The device files are what pytest, npm and shells
            # open for writing; without /dev/null pytest dies with an internal error.
            f"(allow file-write* (subpath {self._q(root)}) (subpath {self._q(home)}))",
            '(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/tty")'
            ' (regex #"^/dev/fd/") (regex #"^/dev/ttys[0-9]+$"))',
            "(allow file-ioctl (literal \"/dev/tty\") (regex #\"^/dev/ttys[0-9]+$\"))",
        ]
        if not network:
            rules.append("(deny network*)")
        return "\n".join(rules) + "\n"

    def wrap(self, argv: Sequence[str], root, *, network, home, extra_env=None) -> list[str]:
        return ["sandbox-exec", "-p", self.profile(argv, root, network=network, home=home), *argv]

    def probe(self, root) -> ProbeResult:
        if not shutil.which("sandbox-exec"):
            return ProbeResult(False, self.name, "sandbox-exec is not installed")
        return _probe(self, root)


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
