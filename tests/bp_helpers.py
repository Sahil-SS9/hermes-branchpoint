"""Helpers shared by tests (importable as `bp_helpers`; conftest puts tests/ on sys.path)."""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
from pathlib import Path

PKG = "hermes_branchpoint"


def mod(name):
    return importlib.import_module(f"{PKG}.{name}")


def git(cwd, *args, check=True):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {args}: {r.stderr}")
    return r.stdout.strip()


SERVICE = '''\
"""Tiny service with a slow endpoint."""
import time

from fastapi_stub import App

app = App()


def compute_report(customer_id: int) -> dict:
    time.sleep(0)  # pretend this is expensive
    return {"customer": customer_id, "total": customer_id * 10}


@app.get("/report/{customer_id}")
def get_report(customer_id: int) -> dict:
    return compute_report(customer_id)
'''

STUB = '''\
class App:
    def get(self, path):
        def deco(fn):
            fn.route = path
            return fn
        return deco
'''

TEST_SERVICE = '''\
from service import compute_report, get_report


def test_report_value():
    assert get_report(3) == {"customer": 3, "total": 30}


def test_compute():
    assert compute_report(1)["total"] == 10
'''


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p




# ---------------------------------------------------------------------------------------------
# A scripted world: FakeLLM replies + a fake build agent, so the whole pipeline runs offline.
# ---------------------------------------------------------------------------------------------
IN_MEMORY = '''\
"""Tiny service with a slow endpoint."""
import time

from fastapi_stub import App

app = App()
_CACHE: dict = {}


def compute_report(customer_id: int) -> dict:
    time.sleep(0)  # pretend this is expensive
    return {"customer": customer_id, "total": customer_id * 10}


@app.get("/report/{customer_id}")
def get_report(customer_id: int) -> dict:
    if customer_id not in _CACHE:
        _CACHE[customer_id] = compute_report(customer_id)
    return _CACHE[customer_id]
'''

ETAG = '''\
"""Tiny service with a slow endpoint."""
import hashlib
import time

from fastapi_stub import App, Request, Response

app = App()


def compute_report(customer_id: int) -> dict:
    time.sleep(0)  # pretend this is expensive
    return {"customer": customer_id, "total": customer_id * 10}


@app.get("/report/{customer_id}")
def get_report(customer_id: int, request: Request = None, response: Response = None) -> dict:
    data = compute_report(customer_id)
    etag = hashlib.md5(str(data).encode()).hexdigest()
    if response is not None:
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "max-age=60"
    return data
'''

STUB2 = '''\
class Request:  # noqa
    headers: dict = {}


class Response:  # noqa
    def __init__(self):
        self.headers = {}


class App:
    def get(self, path):
        def deco(fn):
            fn.route = path
            return fn
        return deco
'''

REDIS_CACHE = '''\
import os

import redis  # new dependency

CLIENT = redis.Redis.from_url(os.environ["REDIS_URL"])
'''

STRATS = {
    "strategies": [
        {"axis": "library-choice", "name": "In-process TTL cache", "thesis": "Keep results in a process-local dict inside the service module",
         "key_decisions": ["module-level dict keyed by customer id", "no new dependency"], "predicted_files": ["service.py"], "trade_off": "not shared across workers"},
        {"axis": "infrastructure", "name": "Shared Redis cache", "thesis": "Store reports in Redis so every worker shares one cache",
         "key_decisions": ["redis client in cache.py", "REDIS_URL env var", "docker-compose redis service"], "predicted_files": ["cache.py", "service.py", "docker-compose.yml"], "trade_off": "new service to run"},
        {"axis": "api-contract", "name": "HTTP ETag caching", "thesis": "Push caching to clients and CDNs with ETag and Cache-Control headers",
         "key_decisions": ["compute an ETag per report", "return 304 on If-None-Match"], "predicted_files": ["service.py", "fastapi_stub.py"], "trade_off": "changes the API contract"},
    ]
}

PLANS = {
    "library-choice": {"summary": "Add a dict cache in service.py.", "plan_md": "1. add _CACHE\n2. read through it",
                       "files": [{"path": "service.py", "action": "modify", "symbols": ["get_report"], "est_lines": 6, "contract_change": False}],
                       "risks": [{"detail": "cache is per process, workers diverge", "severity": "medium"}], "new_deps": [], "new_service": False,
                       "config_keys": [], "schema_changes": [], "public_api_changes": [], "effort": {"tool_calls": 12, "minutes": 5, "tokens": 30000},
                       "reversibility": {"score": 5, "why": "pure code"}, "test_plan": "unit"},
    "infrastructure": {"summary": "Add redis cache module.", "plan_md": "1. cache.py\n2. wire into get_report",
                       "files": [{"path": "cache.py", "action": "create", "symbols": [], "est_lines": 25, "contract_change": False},
                                 {"path": "service.py", "action": "modify", "symbols": ["get_report"], "est_lines": 8, "contract_change": False},
                                 {"path": "docker-compose.yml", "action": "create", "symbols": [], "est_lines": 10, "contract_change": False}],
                       "risks": [{"detail": "redis outage takes the endpoint down", "severity": "high"}, {"detail": "stale data after writes", "severity": "medium"}],
                       "new_deps": ["redis"], "new_service": True, "config_keys": ["REDIS_URL"], "schema_changes": [], "public_api_changes": [],
                       "effort": {"tool_calls": 30, "minutes": 14, "tokens": 90000}, "reversibility": {"score": 2, "why": "needs infra teardown"}, "test_plan": "needs fakeredis"},
    "api-contract": {"summary": "Serve ETags.", "plan_md": "1. etag\n2. 304",
                     "files": [{"path": "service.py", "action": "modify", "symbols": ["get_report"], "est_lines": 14, "contract_change": True}],
                     "risks": [{"detail": "clients must send If-None-Match to benefit", "severity": "medium"}], "new_deps": [], "new_service": False,
                     "config_keys": [], "schema_changes": [], "public_api_changes": ["GET /report/{id} adds ETag + 304"],
                     "effort": {"tool_calls": 18, "minutes": 8, "tokens": 50000}, "reversibility": {"score": 3, "why": "clients may rely on headers"}, "test_plan": "headers"},
}


INDEPENDENT_AXIS = {"A": "library-choice", "B": "infrastructure", "C": "api-contract"}


def independent_plan(agent: str, axis: str | None = None) -> dict:
    """What agent ``agent`` would say when told only the task: its own approach plus a plan (reuses the canned plans)."""
    axis = axis or INDEPENDENT_AXIS.get(agent, "library-choice")
    strat = next(x for x in STRATS["strategies"] if x["axis"] == axis)
    return {"approach": {k: strat[k] for k in ("name", "axis", "thesis", "key_decisions", "trade_off")}, **PLANS[axis]}


def baseline_reply():
    return json.dumps({"strategies": [dict(STRATS["strategies"][0])]})


def _blocks(user: str):
    return re.split(r"(?m)^=== ", user)[1:]


def review_reply(system: str, user: str) -> str:
    out = {}
    for block in _blocks(user):
        label = block.split()[0]
        blast = int(re.search(r"Blast radius (\d+)/10", block).group(1))
        executed = "EXECUTED" in block.split("\n")[0]
        no_issues = "(none recorded)" in block
        ref = re.search(rf"{label}:trace:blast", block)
        out[label] = {"goal_fit": 7, "correctness": 8 if (executed and no_issues) else 5, "blast_radius": 10 - blast, "simplicity": 8 - min(blast, 5),
                      "reversibility": 8, "cost": 6, "rationale": f"blast {blast}/10 [{label}:trace:blast]",
                      "worst_flaw": {"text": f"blast radius is {blast}/10", "refs": [ref.group(0)] if ref else []}}
    return json.dumps({"scores": out})


def judge_reply_factory():
    state = {"n": 0}

    def judge(system: str, user: str) -> str:
        state["n"] += 1
        ranking = re.search(r"FINAL RANKING[^:]*: (.*)", user).group(1)
        order = [m.group(1) for m in re.finditer(r"#\d+ (\w+) score", ranking)]
        win, losers = order[0], order[1:]
        d = {"winner_why": [{"text": f"{win} has the smallest blast radius.", "refs": [f"{win}:trace:blast"]}],
             "losers": {l: [{"text": f"{l} touches more of the system.", "refs": [f"{l}:trace:blast"]}] for l in losers},
             "risks_carried": [{"text": "workers do not share the cache.", "refs": [f"{win}:trace:blast"]}],
             "borrow": [{"from": losers[0], "text": "its trace shows what to watch.", "refs": [f"{losers[0]}:trace:blast"]}] if losers else [],
             "confidence": "medium", "confidence_note": ""}
        if state["n"] == 1:   # first attempt contains a fabricated ref and an uncited sentence: both must be dropped
            d["winner_why"].append({"text": "It is faster by 40x.", "refs": ["Z:test:final"]})
            d["losers"][losers[0]].append({"text": "No evidence at all for this claim.", "refs": []}) if losers else None
        return json.dumps(d)
    return judge


def make_llm(strategies=None, plans=None):
    from hermes_branchpoint.llm import FakeLLM
    strategies = strategies or STRATS
    plans = plans or PLANS

    def simulate(system, user):
        if "YOU ARE AGENT" in user:
            return json.dumps(independent_plan(re.search(r"YOU ARE AGENT (\w+)", user).group(1)))
        axis = re.search(r"\(axis: ([\w-]+)\)", user).group(1)
        return json.dumps(plans[axis])

    script = {"baseline": baseline_reply(), "diverge": json.dumps(strategies), "distinct": json.dumps({"duplicates": []}),
              "review:Maintainer": review_reply, "review:Skeptic": review_reply, "review:User Advocate": review_reply,
              "judge": judge_reply_factory(), "judge-repair": judge_reply_factory()}
    llm = FakeLLM(script)
    # per-route simulate purposes
    for rid in "ABCDE":
        script[f"simulate:{rid}"] = simulate
    return llm


def fake_builder(job):
    """Pretend build agent: writes the route's files into the worktree based on the strategy axis."""
    from hermes_branchpoint.agent import AgentResult
    wt = Path(job.worktree)
    axis = re.search(r"Axis: ([\w-]+)", job.prompt).group(1)
    if axis == "library-choice":
        (wt / "service.py").write_text(IN_MEMORY)
        (wt / "tests" / "test_cache.py").write_text("from service import get_report\n\n\ndef test_cached():\n    assert get_report(2) is get_report(2)\n")
    elif axis == "infrastructure":
        (wt / "cache.py").write_text(REDIS_CACHE)
        (wt / "pyproject.toml").write_text('[project]\nname="x"\nversion="0"\ndependencies=["redis>=5,<6"]\n')
        (wt / "tests" / "test_cache.py").write_text("import cache\n\n\ndef test_needs_redis():\n    assert cache.CLIENT\n")   # fails: no redis / no env
    elif axis == "api-contract":
        (wt / "fastapi_stub.py").write_text(STUB2)
        (wt / "service.py").write_text(ETAG)
        (wt / "tests" / "test_etag.py").write_text("from service import get_report\n\n\ndef test_data():\n    assert get_report(1)['total'] == 10\n")
    return AgentResult(0, f"Built the {axis} route. Trade-off: see ledger. Nothing left undone.", "", 1.0)


def chat_router(strategies=None, plans=None):
    """Route a plain completion to the scripted reply for the Branchpoint step that asked for it (by prompt text)."""
    strategies = strategies or STRATS
    plans = plans or PLANS
    judge = judge_reply_factory()

    def route(system: str, user: str) -> str:
        if "Judge of a design council" in system:
            return judge(system, user)
        if "design council" in system or "Review blind" in system:
            return review_reply(system, user)
        if "strict reviewer of architecture proposals" in system:
            return json.dumps({"duplicates": []})
        if "implementation plan for ONE specific route" in system:
            axis = re.search(r"\(axis: ([\w-]+)\)", user).group(1)
            return json.dumps(plans[axis])
        if "each given the SAME task" in system:
            return json.dumps(independent_plan(re.search(r"YOU ARE AGENT (\w+)", user).group(1)))
        if "staff engineer exploring architecture options" in system:
            return baseline_reply() if "Describe this implementation" in user else json.dumps(strategies)
        raise ValueError("unrecognised prompt: " + system[:80])
    return route
