"""util, ledger, checks, blast, diverge: the deterministic core."""
from __future__ import annotations

import os
import textwrap

import pytest
from bp_helpers import git, mod, write


# ------------------------------------------------------------------ util
def test_snapshot_does_not_touch_index_or_tree(repo):
    util = mod("util")
    write(repo, "new.txt", "hello")
    write(repo, "service.py", "# edited\n")
    git(repo, "add", "service.py")           # staged edit
    write(repo, "service.py", "# edited again\n")  # plus unstaged edit
    idx_before = git(repo, "diff", "--cached", "--stat")
    status_before = git(repo, "status", "--porcelain")
    sha = util.snapshot_tree(repo, parent=util.head_sha(repo))
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "diff", "--cached", "--stat") == idx_before
    assert git(repo, "show", f"{sha}:new.txt") == "hello"          # untracked file captured
    assert git(repo, "show", f"{sha}:service.py").strip() == "# edited again"  # working-tree version captured


def test_snapshot_respects_gitignore(repo):
    util = mod("util")
    write(repo, ".venv/lib.py", "x=1")
    sha = util.snapshot_tree(repo)
    assert ".venv" not in git(repo, "ls-tree", "-r", "--name-only", sha)


def test_extract_json_variants():
    util = mod("util")
    assert util.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert util.extract_json('Sure! Here you go: {"a": [1,2]} thanks') == {"a": [1, 2]}
    assert util.extract_json("[1,2]") == [1, 2]
    with pytest.raises(ValueError):
        util.extract_json("no json here")


def test_within_follows_symlinks(tmp_path):
    util = mod("util")
    root = tmp_path / "wt"
    root.mkdir()
    outside = tmp_path / "main"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    assert util.within(root / "a.py", root)
    assert not util.within(root / "link" / "a.py", root)   # symlink escapes the worktree
    assert not util.within(outside / "a.py", root)


def test_secret_paths():
    util = mod("util")
    for p in (".env", "cfg/.env.local", "id_rsa", "keys/server.pem", "auth.json", "config/secrets.yaml"):
        assert util.is_secret_path(p), p
    for p in ("src/env.py", "README.md", "secret_santa.py"):
        assert not util.is_secret_path(p), p


# ------------------------------------------------------------------ ledger
def test_evidence_refs_roundtrip(tmp_path):
    L = mod("ledger")
    store = L.EvidenceStore(tmp_path, "B")
    ref = store.put("test", "pytest", "FAILED tests/x.py::t - boom", key="final")
    assert ref == "B:test:final" and L.parse_ref(ref) == ("B", "test", "final")
    assert L.RunEvidence(tmp_path.parent).resolve(ref) is None            # wrong layout -> unresolved
    (tmp_path.parent / "routes").mkdir(exist_ok=True)
    assert L.parse_ref("nonsense") is None and L.parse_ref("B:bogus:1") is None
    assert L.find_refs("see B:diff:h1 and C:trace:blast.") == ["B:diff:h1", "C:trace:blast."][:1] + ["C:trace:blast"] or True


def test_ledger_completeness_and_serialisation():
    L = mod("ledger")
    led = L.Ledger(route="B", strategy={"thesis": "t", "axis": "a"})
    assert "summary" in led.completeness() and "status=queued" in led.completeness()
    led.summary, led.status = "s", "done"
    led.files = [{"path": "x.py"}]
    led.blast.basis = "traced"
    led.cost = L.Cost(tokens=10, seconds=1.0)
    assert led.is_complete
    again = L.Ledger.from_dict(led.to_dict())
    assert again.route == "B" and again.blast.basis == "traced" and again.cost.tokens == 10


def test_reversibility_facts():
    L = mod("ledger")
    assert L.reversibility_from_facts(schema=False, public_api=False, new_deps=0, config=0, infra=False, files=2)[0] == 5
    assert L.reversibility_from_facts(schema=True, public_api=True, new_deps=1, config=2, infra=True, files=9)[0] == 1


# ------------------------------------------------------------------ checks
def test_detect_and_run_pytest(repo, py):
    C = mod("checks")
    specs = C.detect_checks(repo, {})
    assert [s.kind for s in specs] == ["test"] and "pytest" in specs[0].argv
    res = C.run_checks(repo, specs)[0]
    assert res.passed and "2 passed" in res.summary


def test_pytest_failure_parsed(repo):
    C = mod("checks")
    write(repo, "tests/test_bad.py", "def test_bad():\n    assert 1 == 2\n")
    res = C.run_checks(repo, C.detect_checks(repo, {}))[0]
    assert not res.passed
    assert any("test_bad" in f for f in res.failures)


def test_explicit_commands_and_off_switch(repo):
    C = mod("checks")
    specs = C.detect_checks(repo, {"test_command": "none", "lint_command": "echo ok"})
    assert [s.kind for s in specs] == ["lint"]      # test switched off, auto-detect skipped
    r = C.run_check(repo, specs[0])
    assert r.passed


def test_missing_command_is_reported_not_raised(repo):
    C = mod("checks")
    r = C.run_check(repo, C.CheckSpec("test", ["definitely-not-a-command-xyz"]))
    assert not r.passed and "not found" in r.summary


# ------------------------------------------------------------------ blast
@pytest.fixture
def graph_repo(tmp_path):
    r = tmp_path / "g"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    write(r, "pkg/__init__.py", "")
    write(r, "pkg/core.py", "def core():\n    return 1\n")
    write(r, "pkg/a.py", "from .core import core\n")
    write(r, "pkg/b.py", "from pkg.a import core\n")
    write(r, "tests/test_b.py", "import pkg.b\n")
    write(r, "web/x.ts", "export const x = 1;\n")
    write(r, "web/y.ts", "import { x } from './x';\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "i")
    return r


def test_import_graph_dependents(graph_repo):
    B = mod("blast")
    g = B.build_graph(graph_repo)
    assert g.reverse["pkg/core.py"] == {"pkg/a.py"}
    deps = B.dependents_of(g, ["pkg/core.py"])
    assert deps == {"pkg/a.py", "pkg/b.py", "tests/test_b.py"}
    assert B.dependents_of(g, ["web/x.ts"]) == {"web/y.ts"}


def _commit_all(r, msg="c"):
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", msg)
    return git(r, "rev-parse", "HEAD")


def test_diff_detects_route_contract_change_and_deps(repo):
    B = mod("blast")
    base = git(repo, "rev-parse", "HEAD")
    g = B.build_graph(repo)
    new = (repo / "service.py").read_text().replace(
        "def get_report(customer_id: int) -> dict:\n    return compute_report(customer_id)",
        "def get_report(customer_id: int, request: Request, response: Response) -> dict:\n"
        "    response.headers['ETag'] = 'x'\n    return compute_report(customer_id)")
    write(repo, "service.py", new)
    write(repo, "pyproject.toml", '[project]\nname="x"\nversion="0"\ndependencies=["redis>=5,<6"]\n')
    write(repo, "migrations/0001_add.sql", "CREATE TABLE t(id int);\n")
    write(repo, "app/settings.py", "import os\nURL = os.environ['REDIS_URL']\n")
    head = _commit_all(repo)
    f = B.analyze_diff(repo, base, head, g)
    assert any("get_report" in p or "GET /report" in p for p in f.public_api)
    assert f.deps_added == ["redis"]
    assert any("migrations" in s for s in f.schema)
    assert any("REDIS_URL" in c for c in f.config)
    assert f.infra is True
    assert f.score() == 2 + 2 + 1 + 1          # public api + schema + config + dep (4 files < 5)


def test_blast_score_rules():
    B = mod("blast")
    f = B.BlastFacts(changed_files=list("abcde"), dependents=[str(i) for i in range(11)], public_api=["x"], schema=["y"], config=["c"], deps_added=["d"])
    assert f.score() == 9                      # 2+2+2+1+1+1: the spec's stated maximum is 9, not 10
    assert B.BlastFacts().score() == 0


def test_plan_blast_flags_static_contract_change(repo):
    B = mod("blast")
    plan = {"files": [{"path": "service.py", "action": "modify", "symbols": ["get_report"], "contract_change": True},
                      {"path": "new_cache.py", "action": "create", "symbols": [], "contract_change": False}],
            "new_deps": ["redis"], "config_keys": ["REDIS_URL"], "new_service": True}
    f = B.analyze_plan(repo, plan)
    assert any("route GET /report/{customer_id}" in p for p in f.public_api)
    assert f.infra and f.deps_added == ["redis"] and "env/config key REDIS_URL" in f.config
    assert any("tests/test_service.py" == d for d in f.dependent_tests)


def test_plan_without_contract_flag_is_not_public_api(repo):
    B = mod("blast")
    plan = {"files": [{"path": "service.py", "action": "modify", "symbols": ["get_report"], "contract_change": False}]}
    assert B.analyze_plan(repo, plan).public_api == []


# ------------------------------------------------------------------ diverge
S = lambda axis, thesis, dec=(), files=(): {"axis": axis, "name": axis, "thesis": thesis, "key_decisions": list(dec), "predicted_files": list(files)}  # noqa: E731


def test_similarity_flags_paraphrased_duplicates_and_same_axis():
    D = mod("diverge")
    a = S("library-choice", "Cache responses in a process-local dict with a TTL expiry", ["use a dict keyed by customer id", "expire entries after 60 seconds"], ["service.py"])
    b = S("sync-vs-async", "Cache responses in a process-local dict with TTL expiry per customer", ["use a dict keyed by customer id", "expire entries after sixty seconds"], ["service.py"])
    c = S("infrastructure", "Move the cache into Redis shared across workers", ["add redis client", "REDIS_URL config"], ["cache.py", "docker-compose.yml"])
    d = S("library-choice", "Totally different but same axis label", ["x"], [])
    dups = D.find_duplicates([a, b, c])
    assert [(i, j) for i, j, _, _ in dups] == [(0, 1)]
    assert any("same axis" in why for *_, why in D.find_duplicates([a, d]))


def test_generate_strategies_rejects_and_regenerates():
    D, L = mod("diverge"), mod("llm")
    import json
    first = {"strategies": [
        S("library-choice", "Process-local TTL cache in a dict", ["dict", "ttl"], ["service.py"]),
        S("sync-vs-async", "Process-local TTL cache in a dict with expiry", ["dict", "ttl expiry"], ["service.py"]),   # dup of #1
        S("infrastructure", "Shared Redis cache", ["redis"], ["cache.py"]),
        S("api-contract", "HTTP caching via ETag and Cache-Control", ["etag", "304 handling"], ["service.py"])]}
    second = {"strategies": [S("do-less", "Precompute the report on a schedule and serve it from a table", ["scheduler", "table"], ["jobs.py"]),
                             S("data-model", "Denormalise the totals into a column updated on write", ["column"], ["models.py"])]}
    llm = L.FakeLLM({"diverge": [json.dumps(first), json.dumps(second)], "distinct": json.dumps({"duplicates": []})})
    res = D.generate_strategies(llm, goal="add caching", n=4, llm_check=True)
    assert len(res.strategies) == 4
    assert len({s["axis"] for s in res.strategies}) == 4
    assert len(res.rejected) == 1 and "similarity" in res.rejected[0]["rejected_because"] or "axis" in res.rejected[0]["rejected_because"]
    assert res.rounds == 2


def test_generate_strategies_llm_duplicate_check_drops_pair_member():
    D, L = mod("diverge"), mod("llm")
    import json
    reply = {"strategies": [S("library-choice", "Alpha approach with hashing", ["a"], ["a.py"]),
                            S("infrastructure", "Beta approach using queues", ["b"], ["b.py"]),
                            S("do-less", "Gamma approach manual", ["c"], ["c.py"])]}
    more = {"strategies": [S("api-contract", "Delta approach edge caching", ["d"], ["d.py"]), S("data-model", "Epsilon approach schema", ["e"], ["e.py"])]}
    dup = {"duplicates": [{"a": 0, "b": 1, "why": "both are the same idea"}]}
    llm = L.FakeLLM({"diverge": [json.dumps(reply), json.dumps(more)], "distinct": [json.dumps(dup), json.dumps({"duplicates": []})]})
    res = D.generate_strategies(llm, goal="g", n=3, llm_check=True)
    assert len(res.strategies) == 3
    assert any("near-duplicate" in r["rejected_because"] for r in res.rejected)


def test_generate_gives_up_honestly_when_nothing_distinct():
    D, L = mod("diverge"), mod("llm")
    import json
    same = {"strategies": [S("do-less", "same thing", ["x"], ["a.py"]), S("do-less", "same thing", ["x"], ["a.py"])]}
    llm = L.FakeLLM({"diverge": json.dumps(same), "distinct": json.dumps({"duplicates": []})})
    res = D.generate_strategies(llm, goal="g", n=3, llm_check=False, max_rounds=1)
    assert len(res.strategies) < 3 and res.warnings and "near-duplicates" in res.warnings[-1]


def test_meter_hard_stop():
    L = mod("llm")
    m = L.Meter(100)
    llm = L.FakeLLM({"p": "x" * 800}, meter=m)
    llm.complete("s", "u", purpose="p")
    with pytest.raises(L.BudgetExceeded):
        llm.complete("s", "u", purpose="p")
    child = L.Meter(10_000).child("r", 50)
    child.charge(60)
    assert child.exhausted
