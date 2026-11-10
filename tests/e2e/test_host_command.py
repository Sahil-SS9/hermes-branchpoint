"""The product path inside a real Hermes: `/branchpoint fork ...` typed as a slash command.

Real host: plugin install, slash-command dispatch, ctx.llm, plugin data dir, the headless build agent
(file tools only, hooks active). Scripted model, so no credentials and no spend.

The data root is prepared once per session by the e2e fixtures, outside this test's product budget
(see ``hostprep``). This test only exercises the product path.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import hostprep
import pytest
from bp_helpers import IN_MEMORY, chat_router, git
from fake_model import FakeModel

pytestmark = pytest.mark.e2e
if not shutil.which("hermes"):
    pytest.skip("hermes binary not on PATH", allow_module_level=True)


def test_fork_command_end_to_end_in_a_real_host(repo, host):
    home = host
    cache_test = "from service import get_report\n\n\ndef test_cached():\n    assert get_report(2) is get_report(2)\n"
    steps = [("read_file", {"path": "service.py"}), ("write_file", {"path": "service.py", "content": IN_MEMORY}),
             ("write_file", {"path": "tests/test_cache.py", "content": cache_test})]
    with FakeModel(steps, chat=chat_router()) as fm:
        (home / "config.yaml").write_text(f"model:\n  provider: custom\n  default: fake-model\n  base_url: {fm.url}\n  api_key: fake\n"
                                          "plugins:\n  enabled:\n    - branchpoint\n"
                                          f"terminal:\n  cwd: {repo}\n")
        env = hostprep.host_env(home, repo)
        cmd = '/branchpoint fork "add caching to the slow /report endpoint" --n 3 --execute top1 --wait --no-view --budget 5m'
        res = subprocess.run(["hermes", "chat", "-Q", "-q", cmd], env=env, capture_output=True, text=True,
                             timeout=hostprep.PRODUCT_TIMEOUT)
    out = res.stdout
    assert "The same task goes to 3 agents" in out, (res.stdout[-1500:], res.stderr[-800:])
    assert "FAKE-MODEL-ERROR" not in out
    assert "[decided]" in out and "★ winner" in out
    assert "Your working tree is not touched" in out
    # state landed in the profile's plugin-data dir, never in the install dir or the repo
    runs = list((home / "plugin-data" / "branchpoint" / "runs").glob("*/run.json"))
    assert len(runs) == 1
    meta = json.loads(runs[0].read_text())
    assert meta["status"] == "decided" and len(meta["executed"]) == 1
    assert not (home / "plugins" / "branchpoint" / "runs").exists()
    ledgers = [json.loads(p.read_text()) for p in runs[0].parent.glob("routes/*/ledger.json")]
    assert len(ledgers) == 3 and all(l["status"] == "done" for l in ledgers)
    built = next(l for l in ledgers if l["fidelity"] == "execute")
    assert built["diff_stats"]["files"] == 2 and built["cost"]["estimated"] is False
    assert (runs[0].parent / "decision.md").read_text().startswith("# Decision Record")
    # the user's tree is untouched
    assert git(repo, "status", "--porcelain") == ""
    assert os.path.isdir(repo)
