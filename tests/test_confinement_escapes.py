"""Regression tests for the routes a build agent can use to escape its confinement.

Each test reproduces one finding from the 0.2.2 hardening review against the real code path, so a
regression shows up here rather than in a route:

1. checks execute agent-written code (``tests/conftest.py``) — must run inside the OS sandbox, and must
   not see a planted provider key;
2. a rewritten ``<wt>/.git`` hijacks a parent git call — must be refused;
3. the tool guard fails closed on unknown keys, unified diffs and protected paths;
4. the fingerprint covers every module, not three;
5. dotfile plans keep their leading dot;
6. ``python -m pytest`` without ``hermes`` skips e2e instead of erroring;
7. the anchor log stays bounded.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PKG = "hermes_branchpoint"          # the repo root *is* the plugin package (see tests/conftest.py)
protocol = __import__(f"{PKG}.protocol", fromlist=["protocol"])
sandbox = __import__(f"{PKG}.sandbox", fromlist=["sandbox"])
util = __import__(f"{PKG}.util", fromlist=["util"])
guard = __import__(f"{PKG}.guard", fromlist=["guard"])
anchor = __import__(f"{PKG}.anchor", fromlist=["anchor"])
checks = __import__(f"{PKG}.checks", fromlist=["checks"])
route = __import__(f"{PKG}.route", fromlist=["route"])
server = __import__(f"{PKG}.server", fromlist=["server"])
hooks = __import__(f"{PKG}.hooks", fromlist=["hooks"])
hook_fingerprint = guard.hook_fingerprint
Hooks = hooks.Hooks


# --------------------------------------------------------------------------- fixtures

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "app.py").write_text("print('hi')\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture()
def worktree(repo: Path, tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "--detach", str(wt), "HEAD")
    return wt


@pytest.fixture()
def marker(worktree: Path, tmp_path: Path) -> dict:
    return {
        "route": "B", "run_id": "run-1",
        "strategy": {"axis": "x", "thesis": "t", "key_decisions": []},
        "confine_root": str(worktree), "worktree": str(worktree),
        "events_path": str(tmp_path / "events.jsonl"),
        "checks_log": str(tmp_path / "checks.jsonl"),
        "check_every": 1000, "max_tool_calls": 60, "token_cap": None, "checks": [], "base_failures": [],
    }


def _guard(marker: dict, tool: str, args: dict) -> dict | None:
    return Hooks()._guard_tool(marker, tool, args)


# --------------------------------------------------------------------------- 1. checks are sandboxed

def test_check_cannot_escape_the_worktree_or_read_credentials(worktree: Path, marker: dict, tmp_path: Path,
                                                              monkeypatch):
    """A route writes ``tests/conftest.py``; the next check must not run it outside the sandbox."""
    (worktree / "tests").mkdir()
    outside = tmp_path / "escaped.txt"
    (worktree / "tests" / "conftest.py").write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(outside)!r}).write_text('escaped')\n"
        "pathlib.Path(os.environ.get('HERMES_HOME', '/tmp') + '/leak.txt').write_text('x')\n"
        "print('KEY=' + os.environ.get('FAKE_PROVIDER_API_KEY', 'none'))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-do-not-leak")
    monkeypatch.delenv(checks.UNSAFE_ENV, raising=False)   # assert the refusal, not the suite-wide acceptance

    CheckSpec, run_check = checks.CheckSpec, checks.run_check

    if sandbox.get(worktree) is None:
        # No usable sandbox on this host: the only correct behaviour is to refuse, never to run it.
        with pytest.raises(sandbox.SandboxUnavailable):
            run_check(worktree, CheckSpec("test", [sys.executable, "-c", "import tests.conftest"], timeout=60))
        pytest.skip(f"no OS sandbox on this host ({sandbox.detect(worktree).detail}); refusal path verified")

    result = run_check(worktree, CheckSpec("test", [sys.executable, "-c", "import tests.conftest"], timeout=120))
    assert not outside.exists(), "a check wrote outside the worktree"
    assert "sk-do-not-leak" not in result.output, "a check saw a planted provider key"


def test_sandbox_probe_requires_a_positive_control(tmp_path: Path):
    """A backend that runs nothing must not pass the probe just because every write was denied."""
    class DeadSandbox(sandbox.BwrapSandbox):
        def wrap(self, argv, root, *, network, home, extra_env=None):
            return ["/bin/false"]          # runs nothing, denies everything by never running

    result = DeadSandbox().probe(tmp_path)
    assert not result.ok
    assert "could not run a command" in result.detail


def test_unsafe_checks_is_the_only_way_past_a_missing_sandbox(worktree: Path, monkeypatch):
    CheckSpec, run_check = checks.CheckSpec, checks.run_check

    monkeypatch.delenv(checks.UNSAFE_ENV, raising=False)
    monkeypatch.setattr(sandbox, "detect", lambda *a, **k: sandbox.ProbeResult(False, "none", "forced"))
    monkeypatch.setattr(sandbox, "_detected", sandbox.ProbeResult(False, "none", "forced"), raising=False)
    with pytest.raises(sandbox.SandboxUnavailable):
        run_check(worktree, CheckSpec("test", [sys.executable, "-c", "print(1)"]))
    result = run_check(worktree, CheckSpec("test", [sys.executable, "-c", "print('ran')"]), unsafe=True)
    assert result.passed and "ran" in result.output


def test_unsafe_checks_env_spelling_is_honoured(worktree: Path, monkeypatch):
    """The same acceptance as ``--unsafe-checks``, spelled as an env var, for a test harness or a scripted demo."""
    monkeypatch.setenv(checks.UNSAFE_ENV, "1")
    monkeypatch.setattr(sandbox, "_detected", sandbox.ProbeResult(False, "none", "forced"), raising=False)
    monkeypatch.setattr(sandbox, "detect", lambda *a, **k: sandbox.ProbeResult(False, "none", "forced"))
    result = checks.run_check(worktree, checks.CheckSpec("test", [sys.executable, "-c", "print('ran')"]))
    assert result.passed


def test_unsafe_checks_environment_is_allowlisted(worktree: Path, monkeypatch):
    """``--unsafe-checks`` still must not hand a check the caller's provider keys."""
    CheckSpec, run_check = checks.CheckSpec, checks.run_check

    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-do-not-leak")
    monkeypatch.setattr(sandbox, "_detected", sandbox.ProbeResult(False, "none", "forced"), raising=False)
    monkeypatch.setattr(sandbox, "detect", lambda *a, **k: sandbox.ProbeResult(False, "none", "forced"))
    result = run_check(worktree, CheckSpec("test", [sys.executable, "-c",
                                                    "import os; print(os.environ.get('FAKE_PROVIDER_API_KEY'))"]),
                       unsafe=True)
    assert "sk-do-not-leak" not in result.output


# --------------------------------------------------------------------------- 2. .git redirect is refused

def test_rewritten_dotgit_cannot_hijack_a_parent_git_call(repo: Path, worktree: Path, tmp_path: Path):
    """A route points ``<wt>/.git`` at its own gitdir with ``core.fsmonitor``; the commit must refuse."""
    attacker = tmp_path / "attacker-gitdir"
    attacker.mkdir()
    _git(attacker, "init", "-q")
    evidence = tmp_path / "fsmonitor-fired.txt"
    _git(attacker, "config", "core.fsmonitor", f"touch {evidence}")
    (worktree / ".git").write_text(f"gitdir: {attacker}\n", encoding="utf-8")

    commit_worktree = route.commit_worktree

    with pytest.raises(util.BranchpointError):
        commit_worktree(repo, worktree, "route build", [])
    assert not evidence.exists(), "a redirected git dir ran a command in the parent"


def test_commit_worktree_pins_the_gitdir_and_blocks_fsmonitor(repo: Path, worktree: Path, tmp_path: Path):
    """With a healthy worktree, the parent commit still refuses to run repository-configured commands."""
    evidence = tmp_path / "fsmonitor-fired.txt"
    _git(repo, "config", "core.fsmonitor", f"touch {evidence}")
    (worktree / "app.py").write_text("print('changed')\n", encoding="utf-8")

    commit_worktree = route.commit_worktree

    sha = commit_worktree(repo, worktree, "route build", [])
    assert len(sha) == 40
    assert not evidence.exists(), "core.fsmonitor was allowed to run during git add"


def test_worktree_gitdir_ignores_a_forged_pointer(repo: Path, worktree: Path, tmp_path: Path):
    """The git dir is resolved from the repo side, so a forged ``<wt>/.git`` cannot choose it."""
    real = util.worktree_gitdir(repo, worktree)
    (worktree / ".git").write_text(f"gitdir: {tmp_path / 'elsewhere'}\n", encoding="utf-8")
    assert util.worktree_gitdir(repo, worktree) == real
    assert not util.dotgit_intact(worktree, real)


# --------------------------------------------------------------------------- 3. guard fails closed

@pytest.mark.parametrize("tool,args", [
    ("write_file", {"file": "/etc/passwd", "content": "x"}),          # unparsed path key
    ("write_file", {"filePath": "/etc/passwd", "content": "x"}),      # OpenCode's spelling
    ("search_files", {"paths": ["/etc/passwd"], "pattern": "root"}),  # list of paths
    ("read_file", {"path": "ok.txt", "unexpected": 1}),              # unknown argument
])
def test_guard_refuses_arguments_it_cannot_parse(marker: dict, worktree: Path, tool: str, args: dict):
    verdict = _guard(marker, tool, args)
    assert verdict is not None and verdict["action"] == "block", f"{tool} {args} was allowed"


def test_guard_refuses_a_unified_diff_that_escapes(marker: dict, worktree: Path):
    patch = ("--- a/app.py\n+++ b/../../outside.txt\n@@ -1 +1 @@\n-x\n+y\n")
    verdict = _guard(marker, "patch", {"path": "app.py", "patch": patch})
    assert verdict is not None and verdict["action"] == "block"


def test_guard_refuses_a_patch_with_no_readable_header(marker: dict, worktree: Path):
    verdict = _guard(marker, "patch", {"path": "app.py", "patch": "just some text\n"})
    assert verdict is not None and verdict["action"] == "block"


def test_guard_still_allows_an_ordinary_edit(marker: dict, worktree: Path):
    assert _guard(marker, "write_file", {"path": "app.py", "content": "print(2)\n"}) is None
    assert _guard(marker, "read_file", {"path": "app.py", "offset": 1, "limit": 20}) is None
    assert _guard(marker, "search_files", {"pattern": "hi", "path": ".", "target": "content",
                                           "file_glob": "*.py", "limit": 5, "offset": 0, "order": "discovery",
                                           "output_mode": "content", "context": 2}) is None


@pytest.mark.parametrize("path", [
    ".git", ".git/config", ".gitattributes", ".gitmodules",
    ".claude/settings.json", ".codex/hooks.json", ".opencode/plugins/x.ts",
    ".pi/extensions/x.ts", ".hermes/plugins/x/__init__.py", ".github/workflows/ci.yml", ".envrc",
])
def test_guard_refuses_protected_paths(marker: dict, worktree: Path, path: str):
    verdict = _guard(marker, "write_file", {"path": path, "content": "x"})
    assert verdict is not None and verdict["action"] == "block", f"{path} was writable"


def test_guard_allows_a_file_merely_named_like_a_protected_dir(marker: dict, worktree: Path):
    assert _guard(marker, "write_file", {"path": "docs/.githubnotes.md", "content": "x"}) is None


# --------------------------------------------------------------------------- 4. secret list, 5. fingerprint

@pytest.mark.parametrize("rel", [
    ".envrc", ".git-credentials", "hosts.yml", "prod.tfvars", "terraform.tfstate",
    "kubeconfig", ".docker/config.json", "service-account-prod.json",
])
def test_secret_paths_are_recognised(rel: str):
    assert util.is_secret_path(rel), f"{rel} was not treated as a secret"


def test_fingerprint_changes_with_any_module(tmp_path: Path, monkeypatch):
    """A stale ``util.py`` used to pass the proof: the fingerprint must cover it."""
    root = Path(guard.__file__).parent
    before = hook_fingerprint()
    victim = root / "util.py"
    original = victim.read_bytes()
    try:
        victim.write_bytes(original + b"\n# fingerprint probe\n")
        assert hook_fingerprint() != before, "a changed util.py was not detected"
    finally:
        victim.write_bytes(original)
    assert hook_fingerprint() == before


# --------------------------------------------------------------------------- 6. dotfiles keep their dot

def test_normalise_rel_keeps_leading_dots():
    assert protocol.normalise_rel(".github/ci.yml") == ".github/ci.yml"
    assert protocol.normalise_rel("./.github/ci.yml") == ".github/ci.yml"
    assert protocol.normalise_rel("./src/app.py") == "src/app.py"
    assert protocol.normalise_rel("/abs/x.py") == "abs/x.py"


def test_no_character_stripping_remains():
    """``lstrip("./")`` strips characters: it turned ``.github/ci.yml`` into ``github/ci.yml``.

    Checked on the parsed AST, so the docstring in ``protocol.py`` that quotes the old idiom does not
    count as an offender.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "lstrip" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "." in node.args[0].value
                    and set(node.args[0].value) <= {".", "/"}):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"character-stripping path normalisation remains at {offenders}"


# --------------------------------------------------------------------------- 7. anchor log stays bounded

def test_turn_log_is_pruned_not_just_its_refs(tmp_path: Path, repo: Path, monkeypatch):
    monkeypatch.setattr(anchor, "MAX_KEPT_ANCHORS", 5)
    calls = {"n": 0}

    def fake_snapshot(*_a, **_k):
        calls["n"] += 1
        return f"{calls['n']:040x}"

    monkeypatch.setattr(anchor, "snapshot_tree", fake_snapshot)
    monkeypatch.setattr(anchor, "git", lambda *a, **k: "")
    monkeypatch.setattr(anchor, "head_sha", lambda *a, **k: None)

    for i in range(1, 13):
        anchor.record_turn(tmp_path, repo, "sess", i, f"prompt {i}")

    turns = anchor.load_turns(tmp_path, repo, "sess")
    assert len(turns) == 5, f"turn log grew to {len(turns)} entries with a cap of 5"


def test_timestamp_parsing_honours_the_offset():
    assert anchor._row_ts({"ts": "2026-01-01T00:00:00Z"}) == 1767225600.0
    assert anchor._row_ts({"ts": "2026-01-01T00:00:00+00:00"}) == 1767225600.0
    # the same instant written with an offset must parse to the same epoch, not to local time
    assert anchor._row_ts({"ts": "2026-01-01T02:00:00+02:00"}) == 1767225600.0


# --------------------------------------------------------------------------- 8. server run id

def test_view_server_rejects_a_path_shaped_run_id():
    assert server._RUN_ID_RE.match("20261111-201500-ab12")
    assert not server._RUN_ID_RE.match("../../etc/passwd")
    assert not server._RUN_ID_RE.match("<script>alert(1)</script>")
