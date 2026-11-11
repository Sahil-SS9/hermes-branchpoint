"""Fixtures for the real-host e2e tests: one prepared data root per session, isolated per test.

See ``hostprep`` for why preparation is separate from assertion. The short version: provisioning a
fresh ``HERMES_HOME`` costs minutes, the product command gets 420 s, and a prepared home is reused
only after a bounded warm probe proves it needs no provisioning work.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Import mode is importlib (pyproject), so a conftest's own directory is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hostprep  # noqa: E402
from hostprep import HostPreparationError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# The e2e suite drives a real ``hermes`` binary. Without one, ``python -m pytest`` (the command the README
# gives for the offline suite) must skip these tests instead of erroring in collection.
if shutil.which("hermes") is None:
    collect_ignore_glob = ["test_*.py"]



@pytest.fixture(scope="session")
def prepared_home():
    """A bootstrapped, warm data root, reused by every test in the session.

    ``BRANCHPOINT_E2E_HOME`` names an existing prepared home (a previous bootstrap, for example).
    Without it the suite provisions one here, once, bounded by ``hostprep.BOOTSTRAP_TIMEOUT``.
    Either way the home is probed in a separate bounded process and a failure fails the run.
    """
    configured = os.environ.get(hostprep.HOME_ENV)
    if configured:
        home = Path(configured).expanduser().resolve()
        try:
            probe = hostprep.probe_ready(home, cwd=home)
        except HostPreparationError as exc:
            pytest.fail(f"{hostprep.HOME_ENV}={configured} is not a prepared Hermes data root: {exc}", pytrace=False)
        metrics = hostprep.record_metric("prepared-home", {
            "home": str(home), "source": "opt-in", "warm_seconds": round(probe.seconds, 1),
            "warm_timeout_seconds": hostprep.WARM_TIMEOUT,
        })
        print(f"\n[e2e] reusing prepared home {home} (warm in {probe.seconds:.1f}s); metrics: {metrics}", flush=True)
        yield home
        return

    home = Path(tempfile.mkdtemp(prefix="branchpoint-e2e-home-", dir=hostprep.data_root()))
    try:
        provisioned = hostprep.provision(home, cwd=home)
        probe = hostprep.probe_ready(home, cwd=home)
    except HostPreparationError:
        shutil.rmtree(home, ignore_errors=True)
        raise
    metrics = hostprep.record_metric("suite-home", {
        "home": str(home), "source": "bootstrap", "provision_seconds": round(provisioned.seconds, 1),
        "bootstrap_timeout_seconds": hostprep.BOOTSTRAP_TIMEOUT, "warm_seconds": round(probe.seconds, 1),
    })
    print(
        f"\n[e2e] provisioned isolated home {home} in {provisioned.seconds:.1f}s; warm probe {probe.seconds:.1f}s."
        f"\n[e2e] reuse it with {hostprep.HOME_ENV}={home}; metrics: {metrics}",
        flush=True,
    )
    yield home
    if hostprep.flag(hostprep.CLEANUP_ENV):
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def host(prepared_home):
    """The prepared home, with only this test's ephemeral state reset and the lock held.

    The bootstrapped runtime (dependency installs, tool store, caches) is never reset. The home's
    ``config.yaml`` is restored afterwards so one test's fake-model config cannot leak into the
    next, and ``plugins/branchpoint`` always holds this checkout.
    """
    home = prepared_home
    with hostprep.exclusive(home):
        config = home / "config.yaml"
        original = config.read_bytes() if config.is_file() else None
        hostprep.reset_ephemeral(home)
        hostprep.install_plugin(home, REPO_ROOT)
        try:
            yield home
        finally:
            hostprep.stop_home_gateway(home)
            if original is None:
                config.unlink(missing_ok=True)
            else:
                config.write_bytes(original)
