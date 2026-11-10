"""Isolated-host preparation for the real-Hermes e2e tests.

The expensive part of these tests is not the product path. A fresh ``HERMES_HOME`` makes the
host build a private dependency runtime for that data root. That provisioning has exceeded the
product command's own 420 s budget on the reference rig (478 s measured before the private uv
cache was warm; 99-328 s once it was), so it cannot live inside the product timeout. Preparation
and assertion are therefore separate concerns:

* the suite uses one prepared home. ``BRANCHPOINT_E2E_HOME`` opts into a home that is already
  bootstrapped; otherwise the suite provisions one, once per session, bounded by
  ``BOOTSTRAP_TIMEOUT``.
* a warm readiness probe runs in its own process, bounded by ``WARM_TIMEOUT``, and proves the home
  has no provisioning work left and emits no bootstrap diagnostics.
* each test resets only its own ephemeral state and holds an exclusive lock on the home, so
  concurrent tests serialise instead of sharing a half-written runtime. Nothing in the
  bootstrapped runtime (dependency installs, caches, tool store) is reset.

The supported provisioning entry point is the host's own launcher with an application command
(``hermes plugins list``), not ``hermes pm ...``: a ``pm`` invocation deliberately bypasses launch
preparation, while an application command under a borrowing data root provisions only that root's
dependency state (``hermes_cli/venv_sync._prepare_borrowed_launch``) and leaves the checkout, its
launchers and the owning data root untouched.

Preparation is a precondition, never an excuse: every failure raises ``HostPreparationError`` and
the fixture turns it into a test failure. It is never a skip.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

#: Opt into an already-bootstrapped data root (used for the real-host tests).
HOME_ENV = "BRANCHPOINT_E2E_HOME"
#: Opt into the cold-provisioning test (downloads a private runtime, so it is not a default).
ALLOW_COLD_ENV = "BRANCHPOINT_E2E_ALLOW_COLD"
#: Remove a data root the suite provisioned itself at session teardown.
CLEANUP_ENV = "BRANCHPOINT_E2E_CLEANUP"
#: Keep the data root a cold-provisioning run created (for reuse as ``HOME_ENV``).
KEEP_ENV = "BRANCHPOINT_E2E_KEEP"
#: Where preparation metrics are recorded.
METRICS_ENV = "BRANCHPOINT_E2E_METRICS"

PRODUCT_TIMEOUT = 420.0
BOOTSTRAP_TIMEOUT = 1200.0
WARM_TIMEOUT = 60.0
#: The host's cheapest application command: it loads the app environment, so it provisions a cold
#: home, but starts no model, gateway or session work.
READINESS_COMMAND = ("hermes", "plugins", "list")

#: Diagnostics the host prints while it provisions a data root. Their absence in the warm probe is
#: what proves the home was prepared outside the product command's budget.
BOOTSTRAP_MARKERS = (
    "preparing dependencies for this data root",
    "preparing dependencies for this update",
    "preparing the isolated hermes runtime",
    "completing source-update dependencies",
    "finishing an interrupted source update",
)

PLUGIN_COPY_IGNORE = (".git", ".venv", "__pycache__", "tests", "demo", "docs", ".ruff_cache", "*.pyc")


class HostPreparationError(RuntimeError):
    """A data root could not be made ready. Always surfaced as a test failure, never a skip."""


@dataclass(frozen=True)
class HostRun:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def diagnostics(self) -> str:
        state = "timed out" if self.timed_out else f"exit {self.returncode}"
        return (
            f"command: {' '.join(self.argv)}\n"
            f"state: {state} after {self.seconds:.1f}s\n"
            f"stderr:\n{self.stderr[-4000:]}\n"
            f"stdout:\n{self.stdout[-2000:]}"
        )


def data_root() -> Path:
    """A disposable directory outside every Hermes home.

    ``TMPDIR`` is not trusted: in an agent session it can point inside a Hermes profile's cache, and
    a data root inside a Hermes home resolves to that home's dependency state instead of
    provisioning its own (the whole point of the isolated host is defeated). Prefer the OS temp dir.
    """
    candidates: list[Path] = [Path("/tmp")] if os.name != "nt" else []
    candidates.append(Path(tempfile.gettempdir()))
    for candidate in candidates:
        if candidate.is_dir() and not inside_hermes_home(candidate):
            return candidate
    raise HostPreparationError("no disposable directory outside a Hermes home is available for e2e data roots")


def inside_hermes_home(path: Path) -> bool:
    """Whether *path* sits in a Hermes home, where dependency state is shared with the live runtime."""
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return True
    candidates = [Path.home() / ".hermes"]
    override = os.environ.get("HERMES_HOME")
    if override:
        candidates.append(Path(override).expanduser())
    for candidate in candidates:
        try:
            if resolved == candidate.resolve() or resolved.is_relative_to(candidate.resolve()):
                return True
        except OSError:
            continue
    return any(part == ".hermes" for part in resolved.parts)


def ensure_isolated(home: Path) -> Path:
    """Refuse a data root whose dependency state would be the live runtime's."""
    if inside_hermes_home(home):
        raise HostPreparationError(
            f"refusing data root {home}: it is inside a Hermes home, where dependency state is shared with "
            "the live runtime instead of provisioned privately; use a disposable root under the OS temp dir"
        )
    return home


def bootstrap_markers(stderr: str) -> list[str]:
    lowered = (stderr or "").lower()
    return [marker for marker in BOOTSTRAP_MARKERS if marker in lowered]


def host_env(home: Path, cwd: Path) -> dict[str, str]:
    """Environment for a host process: the plugin's scrubbed child env, pinned to *home*."""
    from hermes_branchpoint.util import hermes_child_env  # public plugin helper: scrubs session state

    env = hermes_child_env(cwd)
    # The plugin helper drops HERMES_*; this guard variable (leading underscore) marks "inside a
    # gateway" and would otherwise make a test host believe it is the live gateway's child.
    env.pop("_HERMES_GATEWAY", None)
    env["HERMES_HOME"] = str(home)
    return env


def run_host(home: Path, argv: tuple[str, ...], *, cwd: Path, timeout: float) -> HostRun:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(argv), cwd=str(cwd), env=host_env(home, cwd), capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return HostRun(tuple(argv), proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started)
    except subprocess.TimeoutExpired as exc:
        def text(value: object) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            return value if isinstance(value, str) else ""

        return HostRun(tuple(argv), 124, text(exc.stdout), text(exc.stderr), time.monotonic() - started, timed_out=True)


def provision(home: Path, *, cwd: Path) -> HostRun:
    """Bootstrap *home*'s private dependency runtime through the supported launcher path."""
    ensure_isolated(home)
    home.mkdir(parents=True, exist_ok=True)
    result = run_host(home, READINESS_COMMAND, cwd=cwd, timeout=BOOTSTRAP_TIMEOUT)
    if not result.ok:
        raise HostPreparationError(
            f"could not prepare the isolated Hermes data root at {home} within {BOOTSTRAP_TIMEOUT:.0f}s\n"
            + result.diagnostics()
        )
    return result


def probe_ready(home: Path, *, cwd: Path) -> HostRun:
    """Prove *home* is warm: a separate bounded process with no provisioning diagnostics."""
    ensure_isolated(home)
    if not home.is_dir():
        raise HostPreparationError(f"prepared data root {home} does not exist")
    result = run_host(home, READINESS_COMMAND, cwd=cwd, timeout=WARM_TIMEOUT)
    if result.timed_out:
        raise HostPreparationError(
            f"prepared data root {home} did not answer within {WARM_TIMEOUT:.0f}s; it is still provisioning\n"
            + result.diagnostics()
        )
    if result.returncode != 0:
        raise HostPreparationError(f"prepared data root {home} is unusable\n" + result.diagnostics())
    markers = bootstrap_markers(result.stderr)
    if markers:
        raise HostPreparationError(
            f"prepared data root {home} still provisions itself on use ({', '.join(markers)})\n"
            + result.diagnostics()
        )
    return result


@contextlib.contextmanager
def exclusive(home: Path):
    """Serialise tests that share one prepared home (a bootstrapped runtime is not concurrent-safe)."""
    with (home / ".branchpoint-e2e.lock").open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - pytest runs tests serially where fcntl is absent
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield home
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


#: Host state each test owns. The bootstrapped runtime (``installs``, ``cache``, ``tools``) and the
#: plugin install are never reset; everything here is recreated on demand.
EPHEMERAL_PATHS = (
    "plugin-data/branchpoint",
    "sessions",
    "runtime",
    "state",
    "pending_messages",
    "state.db",
    "state.db-shm",
    "state.db-wal",
    "state.db.fts_rebuild.lock",
    "state.db.quarantine.lock",
    "channel_directory.json",
    "projects.db",
    "spawn-ledger.json",
    "spawn-ledger.json.lock",
    "gateway.pid",
    "gateway.lock",
    "gateway.sock",
    "gateway_state.json",
)

#: Functional gateway markers a stopped gateway leaves behind.
GATEWAY_MARKERS = ("gateway.pid", "gateway.lock", "gateway.sock", "gateway_state.json")


def _gateway_record(home: Path) -> dict:
    try:
        data = json.loads((home / "gateway.pid").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _belongs_to_home(pid: int, home: Path) -> bool:
    """Whether *pid* is a gateway process launched for *home* (a stale pid must never hit another)."""
    proc = Path("/proc") / str(pid)
    try:
        cmdline = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
        environ = (proc / "environ").read_bytes().decode("utf-8", "replace").split("\0")
    except OSError:
        return False  # cannot verify on this platform: refuse to signal
    if "gateway" not in cmdline:
        return False
    try:
        expected = {f"HERMES_HOME={home}", f"HERMES_HOME={home.resolve()}"}
    except OSError:
        expected = {f"HERMES_HOME={home}"}
    return bool(expected & set(environ))


def stop_home_gateway(home: Path, *, wait_seconds: float = 30.0) -> None:
    """Stop the on-demand gateway a previous test auto-started in *home*, and clear its markers.

    ``hermes chat`` auto-starts a local gateway that outlives the client (it exits only after
    ``gateway.unmanaged_idle_exit_seconds``). Left running, the next test would share its model
    config and session state instead of starting its own. Only a process whose record names this
    home and whose own environment names it is signalled.
    """
    record = _gateway_record(home)
    pid = record.get("pid")
    recorded = record.get("hermes_home")
    same_home = True
    if isinstance(recorded, str) and recorded:
        try:
            same_home = Path(recorded).resolve() == home.resolve()
        except OSError:
            same_home = False
    if isinstance(pid, int) and same_home and _alive(pid) and _belongs_to_home(pid, home):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + wait_seconds
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5.0
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _alive(pid):
                raise HostPreparationError(f"gateway pid {pid} for data root {home} did not stop")
    for name in GATEWAY_MARKERS:
        (home / name).unlink(missing_ok=True)


def reset_ephemeral(home: Path) -> None:
    """Reset only this suite's own host state, never the bootstrapped runtime (installs/cache/tools)."""
    stop_home_gateway(home)
    for relative in EPHEMERAL_PATHS:
        path = home / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def install_plugin(home: Path, repo: Path) -> Path:
    """Install the checkout under test as the home's ``branchpoint`` plugin (replacing any prior copy)."""
    destination = home / "plugins" / "branchpoint"
    shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo, destination, ignore=shutil.ignore_patterns(*PLUGIN_COPY_IGNORE))
    return destination


def flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def metrics_path() -> Path:
    configured = os.environ.get(METRICS_ENV)
    if configured:
        return Path(configured).expanduser()
    try:
        return data_root() / "branchpoint-e2e-preparation.json"
    except HostPreparationError:  # no disposable root: metrics are diagnostic only
        return Path(tempfile.gettempdir()) / "branchpoint-e2e-preparation.json"


def record_metric(kind: str, values: dict) -> Path:
    """Merge one preparation measurement into the metrics file and return its path."""
    path = metrics_path()
    try:
        current = json.loads(path.read_text())
        if not isinstance(current, dict):
            current = {}
    except (OSError, ValueError):
        current = {}
    current[kind] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return path
