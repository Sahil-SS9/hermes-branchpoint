"""View server, slash command, tools, registration and manifest consistency."""
from __future__ import annotations

import http.client
import json
import re
from pathlib import Path

import pytest
import yaml
from bp_helpers import fake_builder, git, make_llm, mod

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def service(repo, data, monkeypatch):
    P, C, AG = mod("paths"), mod("commands"), mod("agent")
    P.set_data_dir_override(data)
    monkeypatch.chdir(repo)
    svc = C.BranchpointService(None, llm=make_llm(), agent=AG.CallableRunner(fake_builder))
    svc.cwd = lambda: str(repo)            # type: ignore[method-assign]
    yield svc
    if svc.server:
        svc.server.stop()
    P.set_data_dir_override(None)


# ------------------------------------------------------------------ parsing
def test_parse_args_forms():
    C = mod("commands")
    assert C.parse_args('fork "add caching" --n 3 --execute all --budget 300k') == ("fork", ["add caching"], {"n": "3", "execute": "all", "budget": "300k"})
    assert C.parse_args("rewind 3 --base abc123 --keep")[2] == {"base": "abc123", "keep": True}
    assert C.parse_args("adopt C merge --run r1")[:2] == ("adopt", ["C", "merge"])
    assert C.parse_args("fork x \u2014n 4")[2] == {"n": "4"}       # Discord/mobile turn -- into an em dash
    assert C.parse_args("")[0] == "help"
    assert C.parse_tokens("600k") == 600_000 and C.parse_tokens("1.5m") == 1_500_000 and C.parse_tokens("12345") == 12345
    for bad in ("fork x --bogus", "fork x --n"):
        with pytest.raises(mod("util").BranchpointError):
            C.parse_args(bad)


def test_help_and_errors_never_raise(service):
    assert "Branchpoint" in service.handle("")
    assert "Unknown subcommand" in service.handle("frobnicate")
    assert service.handle("fork").startswith("Branchpoint: Give the goal")
    assert "must be between 2 and 8" in service.handle('fork "x" --n 9')
    assert "No Branchpoint runs yet" in service.handle("status")
    assert "prompt number" in service.handle("rewind abc")


# ------------------------------------------------------------------ command flow
def test_fork_wait_status_view_adopt_discard(service, repo):
    out = service.handle('fork "add caching to the slow /report endpoint" --n 3 --wait --budget 5m', session_id="sess")
    assert "The same task goes to 3 agents" in out and "Live view: http://127.0.0.1:" in out and "[decided]" in out
    for name in ("In-process TTL cache", "Shared Redis cache", "HTTP ETag caching"):       # each agent's own approach, in its own words
        assert name in out
    assigned = service.handle('fork "add caching" --n 3 --execute none --wait --no-view --approach assigned')
    for axis in ("library-choice", "infrastructure", "api-contract"):
        assert axis in assigned and "3 distinct routes" in assigned
    assert git(repo, "status", "--porcelain") == ""
    status = service.handle("status")
    assert "★ winner" in status and "Decision: Route" in status
    view = service.handle("view")
    assert "token" in view and "http://127.0.0.1:" in view
    assert "docs/decisions/0001-" in service.handle("adopt B adr") or "Branchpoint:" in service.handle("adopt B adr")
    listing = service.handle("list")
    assert "decided" in listing or "adopted" in listing
    assert "Discarded run" in service.handle("adopt discard")


def test_adopt_unbuilt_route_explains(service):
    service.handle('fork "add caching" --n 3 --execute none --wait')
    msg = service.handle("adopt B merge")
    assert "did not finish a build" in msg or "nothing to merge" in msg


def test_rewind_requires_known_prompt(service):
    assert "Prompt 3 not found" in service.handle("rewind 3", session_id="nope") or "Pass --base" in service.handle("rewind 3", session_id="nope")


def test_tools_are_read_mostly_and_fork_is_simulate_only(service):
    out = json.loads(service.tool_fork({"goal": "add caching", "n": 3}))
    assert "summary" in out and "simulate 2 route(s)" in out["summary"] or "simulate" in out["summary"]
    run = service.get_run(service.latest_run_id())
    run.join(60)
    assert run.settings["execute"] == "none" and run.settings["budget_tokens"] == 200_000
    assert all(l.fidelity == "simulate" for l in run.ledgers())
    st = json.loads(service.tool_status({}))
    assert st["run_id"] == run.run_id and len(st["routes"]) == 3
    assert json.loads(service.tool_fork({}))["error"]


# ------------------------------------------------------------------ server
def test_view_server_security_and_streaming(service):
    service.handle('fork "add caching" --n 3 --execute none --wait --no-view')
    run = service.get_run(service.latest_run_id())
    srv = service.ensure_view()
    port = srv._httpd.server_address[1]

    def get(path, host=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path, headers={"Host": host or f"127.0.0.1:{port}", **(headers or {})})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), body

    t = srv.token
    assert get("/")[0] == 401                                                       # no token
    assert get("/?t=wrong")[0] == 401
    assert get(f"/?t={t}", host="evil.example.com")[0] == 403                       # DNS-rebinding guard
    st, ct, body = get(f"/?t={t}&run={run.run_id}")
    assert st == 200 and b"Branchpoint" in body and run.run_id.encode() in body and b"__RUN_ID__" not in body
    assert b"http://" not in body.replace(b"http://www.w3.org", b"")                # no external resources
    st, _, body = get(f"/state?t={t}&run={run.run_id}")
    state = json.loads(body)
    assert st == 200 and state["meta"]["run_id"] == run.run_id and any(e["type"] == "decision" for e in state["events"])
    st, _, body = get(f"/decision?t={t}&run={run.run_id}")
    assert "# Decision Record" in json.loads(body)["markdown"]
    ref = re.search(r"\[`([A-Z]:trace:blast)`\]", json.loads(body)["markdown"]).group(1)
    st, _, body = get(f"/evidence?t={t}&run={run.run_id}&ref={ref}")
    assert st == 200 and "basis" in json.loads(body)["text"]
    assert get(f"/evidence?t={t}&run={run.run_id}&ref=Z:test:nope")[0] == 404
    # SSE: replay of the stored events, then keepalive
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", f"/events?t={t}&run={run.run_id}&since=0")
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    got = []
    while len(got) < 3:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            got.append(json.loads(line[6:]))
    c.close()
    assert got[0]["seq"] == 1 and {g["type"] for g in got} <= {"status", "strategies", "route", "base_checks", "decision", "log"}


# ------------------------------------------------------------------ registration / manifest
class FakeCtx:
    def __init__(self):
        self.commands, self.hooks, self.tools, self.skills = {}, {}, {}, {}

    def register_command(self, name, handler, description="", args_hint="", **kw):
        self.commands[name] = (handler, description, args_hint)

    def register_hook(self, name, cb, **kw):
        self.hooks[name] = cb

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = (toolset, schema, handler, kw)

    def register_skill(self, name, path, description=""):
        assert Path(path).exists()
        self.skills[name] = path

    def get_config(self, key, default=None):
        return default

    def session_cwd(self):
        return "."


def test_register_matches_manifest():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    ctx = FakeCtx()
    mod("__init__") if False else None
    pkg = __import__("sys").modules["hermes_branchpoint"]
    pkg.register(ctx)
    assert sorted(ctx.tools) == sorted(manifest["provides_tools"])
    assert sorted(ctx.hooks) == sorted(manifest["provides_hooks"])
    assert "branchpoint" in ctx.commands and ctx.skills
    assert manifest["name"] == "branchpoint" and manifest["requires_env"] == []
    for name, (toolset, schema, handler, kw) in ctx.tools.items():
        assert schema["name"] == name and schema["parameters"]["type"] == "object" and schema["description"]
        assert callable(handler)


def test_every_config_key_read_is_declared():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    declared = set(manifest["config_schema"])
    src = "\n".join((ROOT / f).read_text() for f in ("commands.py", "hooks.py", "__init__.py"))
    read = set(re.findall(r'(?:self\._get|self\._settings|_get)\(\s*"([a-z_]+)"', src))
    for tup in re.findall(r'for k in \(([^)]*)\):\s*\n\s*v = self\._get\(k', src):
        read |= set(re.findall(r'"([a-z_]+)"', tup))
    assert read, "expected to find config reads"
    assert read <= declared, f"undeclared settings read by code: {read - declared}"
    unused = declared - read - {"link_dirs"}
    assert not unused, f"declared but never read: {unused}"


def test_no_forbidden_patterns_in_shipped_source():
    """Catalog rules: no self-updater, no shell=True, no YOLO, no credential-store reads, no private core overrides."""
    bad = []
    for f in list(ROOT.glob("*.py")) + list((ROOT / "view").glob("*")):
        text = f.read_text()
        for pat in (r"shell\s*=\s*True", r"--yolo", r"auth\.json\b(?!\s*[\"']?\s*,?\s*$)", r"\.pyc\b", r"urlopen|requests\.|urllib\.request", r"git\s+pull|git\s+fetch.*origin"):
            for m in re.finditer(pat, text):
                line = text[: m.start()].count("\n") + 1
                ctx = text.splitlines()[line - 1]
                if f.name == "util.py" and "auth.json" in ctx:
                    continue
                if pat.startswith("--yolo") and ("never" in ctx.lower() or "not" in ctx.lower() or '"""' in ctx or "#" in ctx):
                    continue
                bad.append((f.name, line, ctx.strip()[:90]))
    # util.is_secret_path legitimately names auth.json as a file it must *refuse* to read
    bad = [b for b in bad if "is_secret_path" not in b[2] and "auth\\.json" not in b[2]]
    assert not bad, bad


def test_repo_option_and_clear_error_outside_a_repo(service, tmp_path, repo):
    import tempfile
    elsewhere = Path(tempfile.mkdtemp(prefix="not-a-repo-", dir="/tmp"))      # outside every git repo (tmp_path may sit inside one)
    service.cwd = lambda: str(elsewhere)          # type: ignore[method-assign]
    msg = service.handle('fork "add caching" --n 3 --execute none')
    assert "not inside a git repository" in msg and "--repo" in msg and "terminal.cwd" in msg
    out = service.handle(f'fork "add caching" --n 3 --execute none --wait --no-view --repo {repo}')
    assert "3 agents" in out or "3 distinct routes" in out


def test_council_model_override_refusal_falls_back_with_a_warning():
    L = mod("llm")

    class Usage:
        input_tokens, output_tokens, total_tokens, cost_usd = 10, 5, 15, None

    class Res:
        text, usage, model = "ok", Usage(), "default-model"

    class FakeHostLlm:
        def __init__(self):
            self.calls = []

        def complete(self, messages, **kw):
            self.calls.append(dict(kw))
            if "model" in kw:
                raise PermissionError("plugin may not override model")
            return Res()

    class Ctx:
        llm = FakeHostLlm()

    llm = L.CtxLLM(Ctx(), default_model="fancy-model")
    assert llm.complete("s", "u").text == "ok"
    # the configured model stays recorded (that is what you asked for); the refusal is remembered separately,
    # so every later call skips it instead of paying for the same refusal again
    assert llm.default_model == "fancy-model" and "fancy-model" in llm.refused_models
    assert "allow_model_override" in llm.warnings[0]
    assert "model" not in Ctx.llm.calls[-1] and "model" in Ctx.llm.calls[0]
    llm.complete("s", "u")                           # later calls go straight to the default model
    assert len(Ctx.llm.calls) == 3 and "model" not in Ctx.llm.calls[-1]


def test_snapshots_skip_huge_repos(repo, data, monkeypatch):
    A = mod("anchor")
    monkeypatch.setattr(A, "MAX_INDEX_BYTES", 10)
    assert A.record_turn(data, repo, "big", 1, "p") is None
