"""Shared fixtures. The repo root is itself the plugin package, so register it as `hermes_branchpoint`."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The offline suite asserts what the engine records about checks, which needs checks to run. On a host with no
# usable OS sandbox that means accepting unsandboxed checks exactly as `--unsafe-checks` does. This is also what
# makes the suite meaningful on such a host; the refusal itself is asserted directly in
# tests/test_confinement_escapes.py.
os.environ.setdefault("BRANCHPOINT_UNSAFE_CHECKS", "1")
PKG = "hermes_branchpoint"


def _load_pkg():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod


_load_pkg()
sys.path.insert(0, str(Path(__file__).parent))
from bp_helpers import SERVICE, STUB, TEST_SERVICE, git, write  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    """A tiny git repo: service.py + fastapi_stub.py + tests, one commit."""
    r = tmp_path / "proj"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    write(r, "service.py", SERVICE)
    write(r, "fastapi_stub.py", STUB)
    write(r, "tests/test_service.py", TEST_SERVICE)
    write(r, "pytest.ini", "[pytest]\npythonpath = .\n")
    write(r, ".gitignore", ".venv/\n__pycache__/\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    # the project's interpreter (untracked + gitignored, like a real .venv): the host python that runs the tests
    (r / ".venv" / "bin").mkdir(parents=True)
    shim = r / ".venv" / "bin" / "python"
    shim.write_text(f'#!/bin/sh\nexec "{Path(sys.prefix) / "bin" / "python"}" "$@"\n')
    shim.chmod(0o755)
    return r


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def pkg():
    return sys.modules[PKG]


@pytest.fixture
def py():
    return sys.executable
