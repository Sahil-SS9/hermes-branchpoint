"""Opt-in measurement of cold data-root provisioning, kept separate from the product assertions.

Off by default: it downloads the private dependency runtime for a fresh data root, which is the
expensive work the rest of the suite deliberately avoids. Enable it with
``BRANCHPOINT_E2E_ALLOW_COLD=1``. The measured seconds are recorded as JSON so a regression in
provisioning cost is visible rather than anecdotal.

It asserts only the provisioning contract: bounded time, a separate warm probe with no bootstrap
diagnostics, and a private dependency state under the new root. The product path is covered by the
other two e2e tests.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import hostprep
import pytest
from hostprep import HostPreparationError

pytestmark = pytest.mark.e2e_cold


def test_cold_data_root_provisions_within_the_bound_and_is_then_warm():
    if not hostprep.flag(hostprep.ALLOW_COLD_ENV):
        pytest.skip(f"opt-in: set {hostprep.ALLOW_COLD_ENV}=1 to measure cold provisioning")
    home = Path(tempfile.mkdtemp(prefix="branchpoint-e2e-cold-", dir=hostprep.data_root()))
    try:
        try:
            provisioned = hostprep.provision(home, cwd=home)
            probe = hostprep.probe_ready(home, cwd=home)
        except HostPreparationError as exc:
            pytest.fail(str(exc), pytrace=False)
        metrics = hostprep.record_metric("cold-home", {
            "home": str(home),
            "source": "cold",
            "provision_seconds": round(provisioned.seconds, 1),
            "bootstrap_timeout_seconds": hostprep.BOOTSTRAP_TIMEOUT,
            "warm_seconds": round(probe.seconds, 1),
            "warm_timeout_seconds": hostprep.WARM_TIMEOUT,
            "install_key_dirs": sorted(p.name for p in (home / "installs").glob("*") if p.is_dir()),
        })
        print(
            f"\n[e2e] cold provisioning: {provisioned.seconds:.1f}s (bound {hostprep.BOOTSTRAP_TIMEOUT:.0f}s),"
            f" warm probe {probe.seconds:.1f}s; home {home}; metrics {metrics}",
            flush=True,
        )
        assert provisioned.seconds <= hostprep.BOOTSTRAP_TIMEOUT
        assert probe.seconds <= hostprep.WARM_TIMEOUT
        assert (home / "installs").is_dir(), "cold provisioning must create the root's own dependency state"
        assert not hostprep.bootstrap_markers(probe.stderr), "a warm root must not provision on use"
    finally:
        if not hostprep.flag(hostprep.KEEP_ENV):
            shutil.rmtree(home, ignore_errors=True)
