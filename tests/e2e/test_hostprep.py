"""Unit guards for the preparation helpers. No host, no provisioning.

These cover the invariants that keep preparation honest: a data root is never inside a live Hermes
home (where dependency state would be shared instead of provisioned privately), bootstrap output is
recognised, and resetting per-test state never touches the bootstrapped runtime.
"""
from __future__ import annotations

from pathlib import Path

import hostprep


def test_data_root_is_outside_a_hermes_home():
    root = hostprep.data_root()
    assert root.is_absolute()
    assert not hostprep.inside_hermes_home(root)


def test_inside_hermes_home_rejects_a_live_profile_root():
    assert hostprep.inside_hermes_home(Path.home() / ".hermes" / "profiles" / "octacon")
    assert not hostprep.inside_hermes_home(hostprep.data_root())


def test_bootstrap_markers_are_matched_case_insensitively():
    assert hostprep.bootstrap_markers("Preparing the isolated Hermes runtime…")
    assert hostprep.bootstrap_markers("hermes: preparing dependencies for this data root (the checkout's ...)")
    assert hostprep.bootstrap_markers("all good") == []


def test_reset_ephemeral_keeps_the_bootstrapped_runtime(tmp_path):
    home = tmp_path / "home"
    for relative in ("installs/abc/environments", "cache/uv", "tools/python", "plugins/branchpoint",
                     "plugin-data/branchpoint/runs", "sessions"):
        (home / relative).mkdir(parents=True)
    (home / "state.db").write_text("session state")
    hostprep.reset_ephemeral(home)
    for kept in ("installs/abc/environments", "cache/uv", "tools/python", "plugins/branchpoint"):
        assert (home / kept).is_dir(), kept
    for reset in ("plugin-data/branchpoint", "sessions", "state.db"):
        assert not (home / reset).exists(), reset
