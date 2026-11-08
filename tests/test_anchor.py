"""Anchor: prompt N -> code state (turn snapshot, Hermes checkpoint, git history, override)."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time

import pytest
from bp_helpers import git, mod, write


def msgs(*pairs):
    """[(prompt, ts)] -> transcript rows"""
    rows = []
    for text, ts in pairs:
        rows.append({"role": "user", "content": text, "timestamp": ts})
        rows.append({"role": "assistant", "content": "ok", "timestamp": (ts + 1) if ts else None})
    return rows


def test_turn_snapshot_gives_exact_state_at_prompt(repo, data):
    A, U = mod("anchor"), mod("util")
    write(repo, "service.py", "# state at prompt 1\n")
    assert A.record_turn(data, repo, "s1", 1, "add caching")
    state1 = U.snapshot_tree(repo)
    write(repo, "service.py", "# state after the agent built things\n")
    write(repo, "cache.py", "x = 1\n")
    A.record_turn(data, repo, "s1", 2, "now add logging")
    a = A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="s1", n=1, messages=msgs(("add caching", 100), ("now add logging", 200)))
    assert a.source == "turn-snapshot" and a.has_baseline
    assert U.tree_of(repo, a.base_sha) == U.tree_of(repo, state1)
    assert a.goal == "add caching" and a.followups == ["now add logging"]
    assert a.head_sha != a.base_sha
    # the user's tree is untouched
    assert (repo / "service.py").read_text().startswith("# state after")


def test_record_turn_is_idempotent_and_never_raises(repo, data, tmp_path):
    A = mod("anchor")
    assert A.record_turn(data, repo, "s", 1, "p") is not None
    assert A.record_turn(data, repo, "s", 1, "p") is None
    assert A.record_turn(data, tmp_path / "not-a-repo", "s", 1, "p") is None


def test_base_override_and_no_baseline(repo, data):
    A = mod("anchor")
    a = A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="", n=1, base_override="HEAD", goal_override="do the thing", messages=[])
    assert a.source == "override" and a.has_baseline is False and any("no baseline" in n for n in a.notes)
    with pytest.raises(mod("util").BranchpointError, match="not a commit"):
        A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="", n=1, base_override="nope", goal_override="g", messages=[])


def test_git_history_fallback_uses_prompt_timestamp(repo, data):
    A = mod("anchor")
    first = git(repo, "rev-parse", "HEAD")
    time.sleep(1.1)
    cut = time.time()
    time.sleep(1.1)
    write(repo, "later.py", "x=1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    a = A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="s9", n=1, messages=msgs(("add a thing", cut)))
    assert a.source == "git-history" and a.base_sha == first


def test_missing_prompt_and_no_state_is_a_clear_error(repo, data):
    A = mod("anchor")
    with pytest.raises(mod("util").BranchpointError, match="Prompt 5 not found"):
        A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="s", n=5, messages=msgs(("a", 1)))
    with pytest.raises(mod("util").BranchpointError, match="Could not find a code state"):
        A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=None, session_id="s", n=1, messages=msgs(("a", None)))


def make_checkpoint_store(home, repo, taken_at, files):
    """Synthesise a Hermes shadow store (layout verified against tools/checkpoint_manager.py v2)."""
    store = home / "checkpoints" / "store"
    store.mkdir(parents=True)
    subprocess.run(["git", "init", "--bare", "-q", str(store)], check=True)
    work = home / "ckpt-work"
    work.mkdir()
    for rel, text in files.items():
        write(work, rel, text)
    env = {"GIT_DIR": str(store), "GIT_WORK_TREE": str(work), "GIT_AUTHOR_NAME": "h", "GIT_AUTHOR_EMAIL": "h@h", "GIT_COMMITTER_NAME": "h",
           "GIT_COMMITTER_EMAIL": "h@h", "HOME": str(home), "PATH": os.environ["PATH"]}
    subprocess.run(["git", "add", "-A"], env=env, cwd=work, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "pre-edit", "-m", f"Hermes-Taken-At: {taken_at}"], env=env, cwd=work, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], env=env, cwd=work, capture_output=True, text=True).stdout.strip()
    dirhash = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:16]
    subprocess.run(["git", "update-ref", f"refs/hermes/{dirhash}", sha], env=env, cwd=work, check=True)
    return sha


def test_hermes_checkpoint_overlay(repo, data, tmp_path):
    A, U = mod("anchor"), mod("util")
    home = tmp_path / "hh"
    home.mkdir()
    t0 = time.time()
    make_checkpoint_store(home, repo, t0 + 5, {"service.py": "# checkpointed state\n", "extra.py": "y=2\n"})
    write(repo, "service.py", "# what the agent finished with\n")
    a = A.resolve_rewind(cwd=repo, data_dir=data, hermes_home=home, session_id="s2", n=1, messages=msgs(("add caching", t0)))
    assert a.source == "hermes-checkpoint"
    assert git(repo, "show", f"{a.base_sha}:service.py").strip() == "# checkpointed state"
    assert git(repo, "show", f"{a.base_sha}:extra.py").strip() == "y=2"
    assert "fastapi_stub.py" in git(repo, "ls-tree", "-r", "--name-only", a.base_sha)       # HEAD files not in the checkpoint survive
    assert a.has_baseline
    del U


def test_checkpoint_before_prompt_is_ignored(repo, data, tmp_path):
    A = mod("anchor")
    home = tmp_path / "hh2"
    home.mkdir()
    t0 = time.time()
    make_checkpoint_store(home, repo, t0 - 500, {"service.py": "# stale\n"})
    assert A.find_hermes_checkpoint(home, repo, t0) is None


def test_baseline_stats_from_transcript():
    A = mod("anchor")
    rows = msgs(("first", 10), ("second", 100)) + [{"role": "tool", "content": "x" * 400, "timestamp": 160}, {"role": "tool", "content": "y", "timestamp": 170}]
    st = A.baseline_stats(rows, 2)
    assert st["tool_calls"] == 2 and st["seconds"] > 60 and st["tokens"] > 0
