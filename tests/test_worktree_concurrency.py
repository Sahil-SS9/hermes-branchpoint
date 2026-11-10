"""Git publishes worktree metadata in steps: readers must not see another in-progress add."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from bp_helpers import git, mod


def test_concurrent_add_cannot_observe_partial_commondir(repo, tmp_path, monkeypatch):
    R = mod("route")
    real_git = R.git
    first_inside = threading.Event()
    second_started = threading.Event()
    second_inside = threading.Event()
    release = threading.Event()
    common = repo / ".git"
    partial = common / "worktrees" / "not-yet-published"
    a, b = tmp_path / "wt-a", tmp_path / "wt-b"
    head = git(repo, "rev-parse", "HEAD")

    def paused_git(args, cwd, **kwargs):
        if args[:2] == ["worktree", "add"]:
            if str(a) in args:
                partial.mkdir(parents=True)
                first_inside.set()
                try:
                    assert release.wait(10), "test did not release first add"
                finally:
                    partial.rmdir()
            elif str(b) in args:
                second_inside.set()
        return real_git(args, cwd, **kwargs)

    monkeypatch.setattr(R, "git", paused_git)

    def second_add():
        second_started.set()
        R.worktree_add(repo, b, head, branch="branchpoint/concurrent/B")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(R.worktree_add, repo, a, head, "branchpoint/concurrent/A")
        try:
            assert first_inside.wait(5)
            second = pool.submit(second_add)
            assert second_started.wait(5)
            # Without a repository-scoped lock, B enters git while A's commondir is unpublished.
            overlapped = second_inside.wait(0.3)
        finally:
            release.set()
        first.result(timeout=20)
        second.result(timeout=20)
    assert not overlapped, "another worktree mutation entered git while commondir metadata was partial"
    assert git(a, "rev-parse", "HEAD") == head
    assert git(b, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""


def test_lock_is_visible_to_other_processes_and_released_on_error(repo):
    import subprocess
    import sys
    import pytest
    U = mod("util")
    lock = repo / ".git" / "branchpoint-worktrees.lock"
    code = """import fcntl,sys
with open(sys.argv[1], 'a') as h:
    try:
        fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(23)
"""
    with pytest.raises(ValueError, match="release proof"):
        with U.worktree_lock(repo):
            assert subprocess.run([sys.executable, "-c", code, str(lock)]).returncode == 23
            raise ValueError("release proof")
    assert subprocess.run([sys.executable, "-c", code, str(lock)]).returncode == 0
    # Never delete the lock file: a waiter could still have the inode open.
    assert lock.exists()


def test_lock_wait_has_a_bounded_timeout(repo):
    import pytest
    U = mod("util")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with U.worktree_lock(repo):
            def waiter():
                with U.worktree_lock(repo, timeout=0.05):
                    raise AssertionError("entered a held lock")
            future = pool.submit(waiter)
            with pytest.raises(U.BranchpointError, match="Timed out waiting"):
                future.result(timeout=2)
