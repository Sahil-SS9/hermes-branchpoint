#!/usr/bin/env python3
"""Create a throw-away git repo of the demo service for Branchpoint to explore.

    python demo/prepare_demo.py /tmp/reports-demo            # prospective: "before we build caching"
    python demo/prepare_demo.py /tmp/reports-demo --built    # retrospective: you already built a TTL cache (route A)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "demo", "GIT_AUTHOR_EMAIL": "demo@example.com",
                        "GIT_COMMITTER_NAME": "demo", "GIT_COMMITTER_EMAIL": "demo@example.com", "PATH": "/usr/bin:/bin:/usr/local/bin"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--built", action="store_true", help="also apply the hand-rolled TTL cache as uncommitted work (route A)")
    ap.add_argument("--python", help="interpreter that has fastapi, httpx and pytest: written as .venv/bin/python so Branchpoint's "
                                     "checks (and every worktree, via a symlink) use it. Without it, checks use `python3` from PATH.")
    args = ap.parse_args()
    dest = Path(args.dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} is not empty", file=sys.stderr)
        return 2
    shutil.copytree(HERE / "service", dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"), dirs_exist_ok=True)
    (dest / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.venv/\n")
    git(dest, "init", "-q", "-b", "main")
    git(dest, "add", "-A")
    git(dest, "commit", "-q", "-m", "reports service: slow /report endpoint")
    if args.built:
        shutil.copy(HERE / "baseline_ttl" / "reports.py", dest / "app" / "reports.py")
        shutil.copy(HERE / "baseline_ttl" / "tests" / "test_cache.py", dest / "tests" / "test_cache.py")
    if args.python:
        shim = dest / ".venv" / "bin" / "python"
        shim.parent.mkdir(parents=True)
        shim.write_text(f'#!/bin/sh\nexec "{Path(args.python).resolve()}" "$@"\n')
        shim.chmod(0o755)
    print(f"demo repo ready at {dest}")
    print("  cd", dest)
    if args.built:
        print('  /branchpoint rewind 1 --base HEAD --goal "add caching to the slow /report endpoint"')
    else:
        print('  /branchpoint fork "add caching to the slow GET /report/{customer_id} endpoint" --n 4')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
