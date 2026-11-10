"""Native guard preflight: no model request and no reliance on slash-command cwd."""
from __future__ import annotations

import pytest
from bp_helpers import git, mod
from fake_model import FakeModel

pytestmark = pytest.mark.e2e


def test_native_preflight_uses_explicit_worktree_without_model(repo, host, monkeypatch):
    G = mod("guard")
    with FakeModel([]) as fm:
        (host / "config.yaml").write_text(
            f"model:\n  provider: custom\n  default: fake-model\n  base_url: {fm.url}\n  api_key: fake\n"
            "plugins:\n  enabled: [branchpoint]\n"
            f"terminal:\n  cwd: {repo}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(host))
        proof = G.HookGuard().confirm(repo)
        assert proof.status == "enabled" and proof.confinement == "blocked"
        assert proof.pid > 0
        assert not any("messages" in r for r in fm.requests), "preflight must never request a model completion"
    assert git(repo, "status", "--porcelain") == ""


def test_standalone_planning_has_no_file_shell_or_network_tools(repo, host, monkeypatch):
    L = mod("llm")
    escape = repo / "must-not-exist.txt"
    # Attempt a forbidden file write through the host's lazy tool bridge, not just a schema check.
    with FakeModel([]) as fm:
        (host / "config.yaml").write_text(
            f"model:\n  provider: custom\n  default: fake-model\n  base_url: {fm.url}\n  api_key: fake\n"
            "plugins:\n  enabled: []\n"
            f"terminal:\n  cwd: {repo}\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(host))
        monkeypatch.chdir(repo)
        reply = L.HermesCliLLM(timeout=90).complete("Reply in JSON.", 'api_key="EXAMPLE_ONLY"')
        completions = [r for r in fm.requests if "messages" in r]
        assert completions
        names = {t.get("function", {}).get("name") for r in completions for t in r.get("tools", [])}
        assert names <= {"todo_list", "tool_search", "tool_describe", "tool_call"}, names
        fm.steps = [("tool_call", {"calls": [{"name": "write_file", "arguments": {"path": str(escape), "content": "blocked"}}]})]
        with pytest.raises(mod("util").BranchpointError):
            L.HermesCliLLM(timeout=90).complete("Reply in JSON.", "Do not call any tools.")
        assert not escape.exists(), "the forbidden tool must never write a file, even through the lazy bridge"
        assert all("EXAMPLE_ONLY" not in str(r["messages"]) for r in completions)
        assert reply.text
    assert git(repo, "status", "--porcelain") == ""


def test_disabled_plugin_refuses_before_any_model_or_build(repo, host, monkeypatch):
    G = mod("guard")
    with FakeModel([]) as fm:
        (host / "config.yaml").write_text(
            f"model:\n  provider: custom\n  default: fake-model\n  base_url: {fm.url}\n  api_key: fake\n"
            "plugins:\n  enabled: []\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(host))
        with pytest.raises(G.HooksNotConfirmed, match="not enabled"):
            G.HookGuard().confirm(repo)
        assert fm.requests == []
    assert git(repo, "status", "--porcelain") == ""
