"""Regression tests for the catalog review of 7051eda (NousResearch/hermes-agent#152959).

1. The sandbox masks the caller's REAL home (and a Hermes home outside it), not a fresh temp dir; the
   probe plants its canary under the real home; a linked ``.venv`` is re-exposed read-only.
2. Detecting checks never runs ``python -m mypy`` with the route worktree as cwd (an agent-written
   ``mypy.py`` would run on the host with every key).
3. A run id must have the ``new_run_id`` shape before it becomes a path.
4. ``__version__`` matches ``plugin.yaml``; ``--allow-sensitive`` (parsed, never read) is gone.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PKG = "hermes_branchpoint"
sandbox = __import__(f"{PKG}.sandbox", fromlist=["sandbox"])
checks = __import__(f"{PKG}.checks", fromlist=["checks"])
paths = __import__(f"{PKG}.paths", fromlist=["paths"])
runmod = __import__(f"{PKG}.run", fromlist=["run"])
commands = __import__(f"{PKG}.commands", fromlist=["commands"])
util = __import__(f"{PKG}.util", fromlist=["util"])
plugin = __import__(PKG)

ROOT = Path(__file__).resolve().parents[1]
MARK = "BP-REVIEW-CANARY-7051eda"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture()
def worktree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "app.py").write_text("print('hi')\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "--detach", str(wt), "HEAD")
    return wt


@pytest.fixture()
def home_canary():
    """A secret file in the caller's real home directory, removed afterwards."""
    d = Path(tempfile.mkdtemp(dir=Path.home(), prefix=".bp-review-canary-"))
    secret = d / "provider.env"
    secret.write_text(f"KEY={MARK}\n", encoding="utf-8")
    try:
        yield secret
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _need_bwrap(root: Path) -> None:
    if sandbox.get(root) is None or sandbox.detect(root).backend != "bwrap":
        pytest.skip(f"bubblewrap sandbox not usable here ({sandbox.detect(root).detail})")


# --------------------------------------------------------------------------- 1. the real home is masked

def test_a_check_cannot_read_a_file_in_the_real_home(worktree: Path, home_canary: Path):
    _need_bwrap(worktree)
    code = (
        "import pathlib\n"
        "print('RAN-OK')\n"
        f"p = pathlib.Path({str(home_canary)!r})\n"
        "print('READ:' + (p.read_text() if p.exists() else 'absent'))\n"
    )
    result = checks.run_check(worktree, checks.CheckSpec("test", [sys.executable, "-c", code], timeout=120))
    assert "RAN-OK" in result.output, result.output       # positive control: the interpreter really ran
    assert MARK not in result.output, "a check read a file from the real home directory"


def test_the_bwrap_profile_masks_the_real_home_and_an_outside_hermes_home(tmp_path: Path, monkeypatch):
    outside = Path("/opt/bp-review-hermes-home")
    monkeypatch.setattr(paths, "hermes_home", lambda: outside)
    root = tmp_path / "wt"
    root.mkdir()
    argv = sandbox.BwrapSandbox().wrap(["true"], root, network=False, home=tmp_path / "h")
    tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
    assert str(Path.home().resolve()) in tmpfs
    assert str(outside) in tmpfs
    # Masks come before the worktree is re-bound (later mounts win in bwrap).
    root_bind = max(i for i, a in enumerate(argv) if a == "--bind" and argv[i + 1] == str(root.resolve()))
    home_mask = argv.index(str(Path.home().resolve()))
    assert home_mask < root_bind


def test_the_probe_plants_its_canary_under_the_real_home(tmp_path: Path):
    """A backend that forgets to mask the real home must fail the probe."""
    if not shutil.which("bwrap"):
        pytest.skip("bwrap is not installed")
    real_home = str(Path.home().resolve())

    class LeakyHome(sandbox.BwrapSandbox):
        def wrap(self, argv, root, *, network, home, extra_env=None):
            cmd = super().wrap(argv, root, network=network, home=home, extra_env=extra_env)
            out, skip = [], False
            for i, a in enumerate(cmd):
                if skip:
                    skip = False
                    continue
                if a == "--tmpfs" and cmd[i + 1] == real_home:
                    skip = True
                    continue
                out.append(a)
            return out

    if not sandbox.BwrapSandbox().probe(tmp_path).ok:
        pytest.skip("bwrap cannot build a namespace here")
    result = LeakyHome().probe(tmp_path)
    assert not result.ok
    assert "credential file" in result.detail


def test_a_linked_venv_under_the_home_is_reexposed_read_only(tmp_path: Path):
    venv = Path(tempfile.mkdtemp(dir=Path.home(), prefix=".bp-review-venv-"))
    try:
        (venv / "bin").mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        root = tmp_path / "wt"
        root.mkdir()
        (root / ".venv").symlink_to(venv)
        argv = sandbox.BwrapSandbox().wrap([str(root / ".venv" / "bin" / "python")], root, network=False,
                                           home=tmp_path / "h")
        ro = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--ro-bind"]
        assert (str(venv.resolve()), str(venv.resolve())) in ro
        assert ("/", "/") in ro
        # never re-exposed writable
        rw = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
        assert str(venv.resolve()) not in rw
    finally:
        shutil.rmtree(venv, ignore_errors=True)


def test_the_macos_profile_hides_the_real_home_and_reallows_the_worktree(tmp_path: Path):
    root = tmp_path / "wt"
    root.mkdir()
    profile = sandbox.SeatbeltSandbox().profile(["true"], root, network=False, home=tmp_path / "h")
    lines = profile.splitlines()
    deny = f'(deny file-read* (subpath "{Path.home().resolve()}"))'
    assert deny in lines
    allow = next(i for i, line in enumerate(lines) if line.startswith("(allow file-read* (subpath"))
    assert lines.index(deny) < allow                    # last matching rule wins in SBPL
    assert f'(subpath "{root.resolve()}")' in lines[allow]
    assert "(deny network*)" in lines


def test_a_linked_venv_under_tmp_is_reexposed_too(tmp_path: Path):
    """bwrap mounts an empty tmpfs over /tmp as well, so a project under /tmp (CI runners) needs the same."""
    venv = Path(tempfile.mkdtemp(dir="/tmp", prefix="bp-review-venv-"))
    try:
        (venv / "bin").mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        root = tmp_path / "wt"
        root.mkdir()
        (root / ".venv").symlink_to(venv)
        argv = sandbox.BwrapSandbox().wrap([str(root / ".venv" / "bin" / "python")], root, network=False,
                                           home=tmp_path / "h")
        ro = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--ro-bind"]
        assert (str(venv.resolve()), str(venv.resolve())) in ro
        assert argv.index("/tmp") < argv.index(str(venv.resolve()))      # bound after the /tmp mask
    finally:
        shutil.rmtree(venv, ignore_errors=True)


def test_the_macos_profile_writes_only_the_worktree_home_and_devices(tmp_path: Path):
    root = tmp_path / "wt"
    root.mkdir()
    profile = sandbox.SeatbeltSandbox().profile(["true"], root, network=False, home=tmp_path / "h")
    writes = [line for line in profile.splitlines() if line.startswith("(allow file-write*")]
    assert '(subpath "/private/var/folders")' not in profile             # not every temp dir of the user
    assert any(f'(subpath "{root.resolve()}")' in line for line in writes)
    assert any('(literal "/dev/null")' in line for line in writes)      # pytest needs it


def test_a_command_the_sandbox_cannot_find_is_reported_as_not_found(tmp_path: Path, monkeypatch):
    for message, code in (("sandbox-exec: execvp() of 'nope' failed: No such file or directory", 71),
                          ("bwrap: execvp nope: No such file or directory", 1)):
        argv = ["/bin/sh", "-c", f"echo {message!r} >&2; exit {code}"]
        monkeypatch.setattr(sandbox, "check_argv", lambda *a, _argv=argv, **k: (_argv, {"PATH": "/usr/bin:/bin"}, ""))
        res = checks.run_check(tmp_path, checks.CheckSpec("test", ["nope"], timeout=30))
        assert res.skipped and "command not found" in res.summary, (message, res.summary)


# --------------------------------------------------------------------------- 2. no unsandboxed mypy probe

def test_detecting_checks_never_imports_a_mypy_from_the_worktree(worktree: Path, tmp_path: Path):
    escaped = tmp_path / "mypy-ran-on-the-host.txt"
    (worktree / "mypy.ini").write_text("[mypy]\n", encoding="utf-8")
    (worktree / "mypy.py").write_text(
        f"import pathlib\npathlib.Path({str(escaped)!r}).write_text('escaped')\n", encoding="utf-8"
    )
    checks.detect_checks(worktree)
    assert not escaped.exists(), "detect_checks ran the worktree's mypy.py on the host"


# --------------------------------------------------------------------------- 3. run ids

@pytest.mark.parametrize("bad", ["../../etc", "..", "0512-201500-ab12/../../x", "/abs", "0512-201500-AB12",
                                 "0512-201500-ab1", "run-1", ""])
def test_a_malformed_run_id_never_becomes_a_path(tmp_path: Path, bad: str):
    with pytest.raises(util.BranchpointError):
        runmod.RunPaths(tmp_path).run_dir(bad)
    with pytest.raises(util.BranchpointError):
        runmod.RunPaths(tmp_path).wt_root(bad)
    with pytest.raises(util.BranchpointError):
        runmod.Run.load(tmp_path, bad)


def test_new_run_ids_have_the_validated_shape():
    for _ in range(20):
        rid = runmod.new_run_id()
        assert re.fullmatch(r"\d{4}-\d{6}-[0-9a-f]{4}", rid)
        runmod.RunPaths(Path("/nonexistent")).run_dir(rid)


def test_the_command_surface_refuses_a_path_shaped_run_id(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(commands, "data_dir", lambda: tmp_path)
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "run.json").write_text('{"run_id": "x"}', encoding="utf-8")
    out = commands.BranchpointService().handle("status --run ../secret")
    assert "run id" in out.lower()


# --------------------------------------------------------------------------- 4. version and flags

def test_version_matches_the_manifest():
    manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    version = re.search(r"^version:\s*(\S+)", manifest, re.M).group(1)
    assert plugin.__version__ == version


def test_allow_sensitive_is_not_advertised_or_parsed():
    assert "--allow-sensitive" not in commands.USAGE
    assert "--allow-sensitive" not in commands._BOOLS
    assert "allow_sensitive" not in commands.BranchpointService().settings({})
