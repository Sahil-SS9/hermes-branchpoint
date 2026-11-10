"""Release blocker: a route is never built by a child that does not run this plugin's hooks.

The build agent is a separate ``hermes chat`` process, and the hooks loaded *there* are the only thing that
confines its writes to the route's worktree, caps it, records ``tool_events.jsonl``/``checks_log.jsonl`` and
feeds check results back. An installed-but-not-enabled plugin (or an absent one) leaves a plain Hermes
session with file tools: the build runs unconfined, the evidence files stay 0 bytes, and the run still
reports a green route - which is what the independent trial recorded (routes A and Z: ``status: done``,
``tool_calls: 0``, empty ``tool_events.jsonl``).

Every test here is written against observed behaviour, not intent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from bp_helpers import fake_builder, git, make_llm, mod

G = mod("guard")


class Host:
    """A stand-in host around one Hermes home: the plugin's reported status and a child's answer."""

    home = "/home/test/.hermes"

    def __init__(self, status: str | None = "enabled", answer=None, error=None):
        self.status = status
        self.answer = answer if answer is not None else {
            "nonce": "n0nce", "route": G.PROBE_ROUTE, "confinement": "blocked", "fingerprint": G.hook_fingerprint(), "pid": 4242,
        }
        self.error = error

    def plugin_status(self):
        return self.status

    def probe(self):
        if self.error is not None:
            raise self.error
        return dict(self.answer)


def build_run(repo, data, *, agent, execute="all"):
    A, R = mod("anchor"), mod("run")
    anchor = A.resolve_fork(cwd=repo, goal="add caching to the slow /report endpoint")
    settings = {"n": 3, "execute": execute, "budget_tokens": 5_000_000, "max_parallel": 2, "approach": "assigned"}
    return R.Run.create(data, anchor=anchor, mode="fork", goal=anchor.goal, settings=settings, llm=make_llm(), agent=agent)


def ledgers(run) -> dict:
    return {l.route: l for l in run.ledgers()}


# --------------------------------------------------------------------------- the refusal

def test_a_run_refuses_to_build_when_the_child_would_have_no_hooks(repo, data, monkeypatch):
    """The trial's shape: the plugin is installed but not enabled, so no child may be started at all."""
    AG = mod("agent")
    runner = AG.HermesAgentRunner(guard=G.HookGuard(host_factory=lambda cwd: Host(status="not enabled")))
    started: list[str] = []
    monkeypatch.setattr(AG.HermesAgentRunner, "run", lambda self, job, abort: started.append(job.route_id))

    run = build_run(repo, data, agent=runner)
    run.prepare()
    run.go(background=False)

    # the run stops, and says exactly what to run - in the home that would have built it
    assert run.meta["status"] == "failed", run.meta.get("warnings")
    error = run.meta.get("error") or ""
    assert "hermes plugins enable branchpoint" in error, error
    assert Host.home in error, error
    assert "plugins.enabled: [branchpoint]" in error, error
    assert "--execute none" in error, error

    # no child build was started, and no route worktree was created: the refusal is before any build write
    assert started == [], started
    assert not run.meta.get("executed"), run.meta.get("executed")
    for rid in run.meta.get("routes") or []:
        assert not (run.wt_root / rid).exists(), rid
    assert not list((run.dir / "routes").glob("*/agent.log")), "a build child was started anyway"
    assert not [l for l in ledgers(run).values() if l.fidelity == "execute"], "a route was built anyway"

    # the user's tree and the base worktree are untouched
    assert git(repo, "status", "--porcelain") == ""
    assert "_CACHE" not in (run.wt_root / "base" / "service.py").read_text()


def test_the_refusal_names_an_absent_plugin_and_never_installs_it(repo, data):
    """A home without the plugin is told to install it; Branchpoint changes no config to get past the gate."""
    AG = mod("agent")
    runner = AG.HermesAgentRunner(guard=G.HookGuard(host_factory=lambda cwd: Host(status=None)))
    run = build_run(repo, data, agent=runner)
    run.prepare()
    run.go(background=False)
    error = run.meta.get("error") or ""
    assert "has no `branchpoint` plugin installed" in error, error
    assert "hermes plugins install" in error, error
    assert not [l for l in ledgers(run).values() if l.fidelity == "execute"], ledgers(run)


# --------------------------------------------------------------------------- the proven path

def test_a_route_build_starts_once_the_child_proves_its_hooks(repo, data, monkeypatch):
    """The installed path still runs: with the plugin enabled and a child that reports, the build proceeds."""
    AG, P = mod("agent"), mod("protocol")
    runner = AG.HermesAgentRunner(guard=G.HookGuard(host_factory=lambda cwd: Host(status="enabled")))
    built: list[str] = []

    def hooked_child(self, job, abort):
        built.append(job.route_id)
        marker = P.read_marker(job.worktree)                  # what a real hooked child sees and reports
        Path(marker["handshake"]["path"]).write_text(json.dumps({
            "nonce": marker["handshake"]["nonce"], "route": job.route_id, "confinement": "blocked", "fingerprint": G.hook_fingerprint(), "pid": 4242,
        }))
        return fake_builder(job)

    monkeypatch.setattr(AG.HermesAgentRunner, "run", hooked_child)
    run = build_run(repo, data, agent=runner)
    run.prepare()
    run.go(background=False)

    assert built, "the proven build was never started"
    led = ledgers(run)[built[0]]
    assert led.status == "done", (led.status, led.notes, [i.detail for i in led.issues])
    assert not [i for i in led.issues if i.kind == "runtime"], [i.detail for i in led.issues]
    assert led.diff_stats["files"] >= 1


def test_a_build_child_that_never_reports_is_not_a_finished_route(repo, data, monkeypatch):
    """Defence in depth: if the proof were ever to slip, a child that wrote files still cannot be a route."""
    AG = mod("agent")
    runner = AG.HermesAgentRunner(guard=G.HookGuard(host_factory=lambda cwd: Host(status="enabled")))

    def silent_child(self, job, abort):
        return fake_builder(job)                               # writes the route's files, reports nothing

    monkeypatch.setattr(AG.HermesAgentRunner, "run", silent_child)
    run = build_run(repo, data, agent=runner)
    run.prepare()
    run.go(background=False)

    for rid in run.meta["executed"]:
        led = ledgers(run)[rid]
        assert led.status == "failed", (rid, led.status)
        assert not led.is_complete, led.completeness()
        stop = [i for i in led.issues if i.kind == "runtime" and "hooks" in i.detail]
        assert stop and stop[0].severity == "high", [(i.kind, i.detail) for i in led.issues]
    assert run.meta["status"] == "failed"


# --------------------------------------------------------------------------- the host side of the proof

def plugins_json(entries: list[dict]) -> str:
    return json.dumps(entries)


class Cmd:
    def __init__(self, rc=0, out="", err=""):
        self.rc, self.out, self.err, self.timed_out = rc, out, err, False

    @property
    def ok(self):
        return self.rc == 0 and not self.timed_out


def host_hooks(repo, tmp_path, *, list_out, chat):
    """A real ``HermesHostHooks`` with the two host processes scripted (no hermes binary needed)."""
    calls: list[list[str]] = []

    def runner(argv, cwd, *, env=None, timeout=0, **_kw):
        calls.append(list(argv))
        return chat(argv, Path(cwd)) if "chat" in argv else Cmd(0, list_out)

    hooks = G.HermesHostHooks(repo, env_factory=lambda *a: {"HERMES_HOME": str(tmp_path / "home")},
                              argv_factory=lambda: ["hermes"], runner=runner)
    return hooks, calls


def test_the_guard_refuses_an_installed_but_disabled_plugin(repo, tmp_path):
    hooks, calls = host_hooks(repo, tmp_path, list_out=plugins_json([{"name": "branchpoint", "status": "not enabled"}]),
                              chat=lambda argv, cwd: Cmd(0, "READY"))
    guard = G.HookGuard(host_factory=lambda cwd: hooks)
    with pytest.raises(G.HooksNotConfirmed) as exc:
        guard.confirm(repo)
    message = str(exc.value)
    assert "not enabled" in message and "hermes plugins enable branchpoint" in message
    assert len(calls) == 1, calls            # it stopped at the status check: no probe child, no spend


def test_the_guard_refuses_when_the_enabled_plugin_never_answers(repo, tmp_path):
    """Enabled is not proof: the child itself has to report, and a silent child is refused with its output."""
    hooks, calls = host_hooks(repo, tmp_path, list_out=plugins_json([{"name": "branchpoint", "status": "enabled"}]),
                              chat=lambda argv, cwd: Cmd(4, "", "hermes: could not load plugins"))
    with pytest.raises(G.HooksNotConfirmed) as exc:
        G.HookGuard(host_factory=lambda cwd: hooks).confirm(repo)
    message = str(exc.value)
    assert "never reported its hooks" in message and "could not load plugins" in message, message
    assert any("chat" in c for c in calls), calls


def test_the_guard_proves_confinement_not_just_hook_presence(repo, tmp_path):
    """A child that answers but says the escape path was allowed is refused: the decision path was not live."""
    def answering_child(argv, cwd):
        marker = mod("protocol").read_marker(cwd)
        Path(marker["handshake"]["path"]).write_text(json.dumps({
            "nonce": marker["handshake"]["nonce"], "route": marker["route"], "confinement": "unconfined", "pid": 7,
        }))
        return Cmd(0, "READY")

    hooks, _ = host_hooks(repo, tmp_path, list_out=plugins_json([{"name": "branchpoint", "status": "enabled"}]),
                          chat=answering_child)
    with pytest.raises(G.HooksNotConfirmed) as exc:
        G.HookGuard(host_factory=lambda cwd: hooks).confirm(repo)
    assert "confinement" in str(exc.value), str(exc.value)


def test_the_guard_accepts_a_child_that_reports_and_confines(repo, tmp_path):
    seen: dict = {}

    def answering_child(argv, cwd):
        marker = mod("protocol").read_marker(cwd)
        seen.update({"route": marker["route"], "nonce": marker["handshake"]["nonce"], "cwd": str(cwd)})
        Path(marker["handshake"]["path"]).write_text(json.dumps({
            "nonce": marker["handshake"]["nonce"], "route": marker["route"], "confinement": "blocked", "fingerprint": G.hook_fingerprint(), "pid": 99,
        }))
        return Cmd(0, "READY")

    hooks, calls = host_hooks(repo, tmp_path, list_out=plugins_json([{"name": "branchpoint", "status": "enabled"}]),
                              chat=answering_child)
    proof = G.HookGuard(host_factory=lambda cwd: hooks).confirm(repo)
    assert proof.status == "enabled" and proof.confinement == "blocked" and proof.pid == 99
    assert proof.is_child_process
    probe_argv = next(c for c in calls if "chat" in c)
    # the probe is the same shape as a build child, and never asks for hook approval it does not need
    assert probe_argv[:4] == ["hermes", "chat", "-Q", "--query-file"] and "-t" in probe_argv
    assert "--accept-hooks" not in probe_argv
    # the probe worktree is gone again
    assert not seen["cwd"] or not Path(seen["cwd"]).exists()


def test_one_proof_per_run(repo, tmp_path):
    """Routes build in parallel; the proof is taken once, so a run pays for one probe, not one per route."""
    calls: list[str] = []

    def answering_child(argv, cwd):
        calls.append(str(cwd))
        marker = mod("protocol").read_marker(cwd)
        Path(marker["handshake"]["path"]).write_text(json.dumps({
            "nonce": marker["handshake"]["nonce"], "route": marker["route"], "confinement": "blocked", "fingerprint": G.hook_fingerprint(), "pid": 1,
        }))
        return Cmd(0, "READY")

    hooks, _ = host_hooks(repo, tmp_path, list_out=plugins_json([{"name": "branchpoint", "status": "enabled"}]),
                          chat=answering_child)
    guard = G.HookGuard(host_factory=lambda cwd: hooks)
    first, second = guard.confirm(repo), guard.confirm(repo)
    assert first == second and len(calls) == 1, calls


# --------------------------------------------------------------------------- handshake checking

@pytest.mark.parametrize("payload,expected", [
    ({"nonce": "other", "route": "A", "confinement": "blocked"}, "nonce mismatch"),
    ({"nonce": "n", "route": "B", "confinement": "blocked"}, "was for route"),
    ({"nonce": "n", "route": "A"}, "confinement"),
    ({"nonce": "n", "route": "A", "confinement": "blocked", "fingerprint": "stale"}, "installed hook code differs"),
])
def test_verify_handshake_rejects_a_report_that_does_not_belong(tmp_path, payload, expected):
    path = tmp_path / "hs.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(G.HooksNotConfirmed) as exc:
        G.verify_handshake(path, nonce="n", route="A")
    assert expected in str(exc.value), str(exc.value)


def test_verify_handshake_rejects_a_missing_report(tmp_path):
    with pytest.raises(G.HooksNotConfirmed) as exc:
        G.verify_handshake(tmp_path / "absent.json", nonce="n", route="A")
    assert "never reported" in str(exc.value)


# --------------------------------------------------------------------------- the plugin side

def test_pre_llm_call_reports_the_hooks_before_any_tool_call(repo, tmp_path, monkeypatch):
    """The report comes from pre_llm_call: earlier than any tool call, so before a build can write."""
    R, P, H = mod("route"), mod("protocol"), mod("hooks")
    wt = tmp_path / "wt" / "B"
    R.worktree_add(repo, wt, git(repo, "rev-parse", "HEAD"), branch="branchpoint/t/handshake")
    handshake = tmp_path / "hs.json"
    outside = tmp_path / "outside-the-worktree.txt"
    P.write_marker(wt, {"route": "B", "run_id": "r1", "strategy": {"axis": "x", "thesis": "y", "key_decisions": []},
                        "confine_root": str(wt), "worktree": str(wt),
                        "events_path": str(tmp_path / "ev.jsonl"), "checks_log": str(tmp_path / "cl.jsonl"),
                        "check_every": 5, "max_tool_calls": 50, "token_cap": None, "checks": [], "base_failures": [],
                        "handshake": {"path": str(handshake), "nonce": "abc"}, "probe_escape": str(outside)})
    monkeypatch.chdir(wt)

    h = H.Hooks()
    assert not handshake.exists()
    h.pre_llm_call(session_id="s", user_message="go")
    report = json.loads(handshake.read_text())
    assert report["nonce"] == "abc" and report["route"] == "B" and report["reported_by"] == "pre_llm_call"
    assert report["confinement"] == "blocked", report          # the real escape decision, not a claim
    assert "pre_tool_call" in report["hooks"] and str(wt) in report["confine_root"]
    written = handshake.stat().st_mtime_ns
    h.pre_llm_call(session_id="s", user_message="again")
    assert handshake.stat().st_mtime_ns == written, "the report was rewritten on every call"


def test_no_report_when_the_marker_does_not_ask_for_one(repo, tmp_path, monkeypatch):
    """An ordinary route in a user's session is unchanged: no handshake in the marker, no report written."""
    R, P, H = mod("route"), mod("protocol"), mod("hooks")
    wt = tmp_path / "wt" / "C"
    R.worktree_add(repo, wt, git(repo, "rev-parse", "HEAD"), branch="branchpoint/t/nohandshake")
    P.write_marker(wt, {"route": "C", "confine_root": str(wt), "events_path": str(tmp_path / "ev.jsonl"),
                        "checks_log": str(tmp_path / "cl.jsonl"), "checks": [], "base_failures": [],
                        "strategy": {"axis": "x", "thesis": "y", "key_decisions": []}})
    monkeypatch.chdir(wt)
    H.Hooks().pre_llm_call(session_id="s", user_message="go")
    assert not list(tmp_path.glob("*.json")) or not (tmp_path / "hs.json").exists()
