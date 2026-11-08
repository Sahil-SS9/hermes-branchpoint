#!/usr/bin/env python3
"""Offline replay of the headline demo: real git worktrees, real tests, real blast tracing, real council code.

Only the *model* is scripted (planning, build and council replies are canned), so this runs in about a minute
with no credentials, and the live view animates at a pace you can record. It is labelled as scripted everywhere.

    python scripts/offline_demo.py --python /path/to/python-with-fastapi-httpx-pytest [--delay 2] [--serve]

Then:  scripts/branchpoint --state-dir <state> adopt C merge    (run from the demo repo; tests go green)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_pkg():
    spec = importlib.util.spec_from_file_location("hermes_branchpoint", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_branchpoint"] = mod
    spec.loader.exec_module(mod)


# What the scripted "model" says. Every sentence is a fact about the demo routes, taken from their own strategies and ledgers.
FLAWS = {
    "failing": "its own tests fail because the redis package and a REDIS_URL are not set up",
    "library-choice": "the cache lives inside one process, so it is lost on every restart and is not shared between workers",
    "api-contract": "it changes the signature of GET /report, so any code calling it directly needs updating",
    "do-less": "it needs a background refresher and a table, and a report can be up to thirty seconds out of date",
    "infrastructure": "it adds a new service to run and monitor",
}
WIN_WHY = {
    "api-contract": "Route {win} fits the goal best: clients and CDNs do the caching, so the server skips the slow work whenever nothing has changed. It touches only {files} files and all its checks pass.",
}
RISK = {"api-contract": "The winner changes the signature of GET /report, so anyone calling it directly will need to update their code."}
LOSE_WHY = {
    "do-less": "A close second: its checks pass and nothing public changes, but it needs a background refresher, a table and {files} files, and reports can be up to thirty seconds old.",
    "library-choice": "It works and is the easiest to undo, but its cache lives inside one process, so it is lost on restart and not shared between workers.",
    "infrastructure": "It fails its own tests because the redis package and a REDIS_URL are not set up, and it adds a new dependency and a container to run.",
}
BORROW = {"do-less": "Refreshing reports ahead of time is worth keeping in mind for the slowest reports."}


STRATS = [
    {"axis": "infrastructure", "name": "Shared Redis cache", "thesis": "Store finished reports in Redis so every worker shares one cache and it survives restarts.",
     "key_decisions": ["new app/cache.py around redis-py", "REDIS_URL env var", "redis service in docker-compose"],
     "predicted_files": ["app/cache.py", "app/reports.py", "docker-compose.yml"], "trade_off": "a new service to run and monitor"},
    {"axis": "api-contract", "name": "HTTP ETag caching", "thesis": "Let clients and CDNs cache: compute an ETag per report and answer If-None-Match with 304.",
     "key_decisions": ["ETag from the report hash", "Cache-Control: private, max-age=60", "304 on conditional GET"],
     "predicted_files": ["app/main.py", "tests/test_etag.py"], "trade_off": "changes the API contract; only helps clients that revalidate"},
    {"axis": "do-less", "name": "Precompute on a schedule", "thesis": "Do the slow aggregation off the request path: refresh every report every 30 s and serve from a table.",
     "key_decisions": ["background refresher thread", "in-memory table keyed by customer", "request path only reads"],
     "predicted_files": ["app/precompute.py", "app/reports.py", "app/main.py"], "trade_off": "data can be up to 30 s stale; refresher must be supervised"},
]
BASELINE = {"axis": "library-choice", "name": "Hand-rolled in-memory TTL cache", "thesis": "Keep results in a process-local dict with a 60 s TTL inside app/reports.py.",
            "key_decisions": ["module-level dict keyed by customer id", "monotonic-clock TTL", "no new dependency"], "predicted_files": ["app/reports.py"],
            "trade_off": "per-process: workers do not share it and it is lost on restart"}

PLANS = {
    "infrastructure": {"summary": "Add a Redis-backed cache module and read through it in build_report.", "plan_md": "1. add app/cache.py (redis-py)\n2. read-through in build_report\n3. add redis to docker-compose and REDIS_URL to config\n4. tests need a Redis (or fakeredis)",
        "files": [{"path": "app/cache.py", "action": "create", "symbols": [], "est_lines": 14, "contract_change": False}, {"path": "app/reports.py", "action": "modify", "symbols": ["build_report"], "est_lines": 10, "contract_change": False},
                  {"path": "docker-compose.yml", "action": "create", "symbols": [], "est_lines": 10, "contract_change": False}],
        "risks": [{"detail": "Redis outage takes the endpoint down unless there is a fallback", "severity": "high"}, {"detail": "tests need a Redis server or a fake", "severity": "medium"}],
        "new_deps": ["redis"], "new_service": True, "config_keys": ["REDIS_URL"], "schema_changes": [], "public_api_changes": [],
        "effort": {"tool_calls": 28, "minutes": 14, "tokens": 90000}, "reversibility": {"score": 2, "why": "needs infrastructure teardown"}},
    "api-contract": {"summary": "Serve ETags and honour If-None-Match.", "plan_md": "1. hash each report into an ETag\n2. return 304 when If-None-Match matches\n3. set Cache-Control",
        "files": [{"path": "app/main.py", "action": "modify", "symbols": ["get_report"], "est_lines": 14, "contract_change": True}, {"path": "tests/test_etag.py", "action": "create", "symbols": [], "est_lines": 10, "contract_change": False}],
        "risks": [{"detail": "only clients that revalidate benefit; first request is still slow", "severity": "medium"}], "new_deps": [], "new_service": False,
        "config_keys": [], "schema_changes": [], "public_api_changes": ["GET /report/{customer_id} gains ETag/Cache-Control and may answer 304"],
        "effort": {"tool_calls": 14, "minutes": 7, "tokens": 45000}, "reversibility": {"score": 3, "why": "clients may start relying on the headers"}},
    "do-less": {"summary": "Refresh reports on a schedule and serve from a table.", "plan_md": "1. add app/precompute.py with a refresher thread\n2. build_report reads the table\n3. start the refresher at app startup",
        "files": [{"path": "app/precompute.py", "action": "create", "symbols": [], "est_lines": 25, "contract_change": False}, {"path": "app/reports.py", "action": "modify", "symbols": ["build_report"], "est_lines": 12, "contract_change": False},
                  {"path": "app/main.py", "action": "modify", "symbols": [], "est_lines": 6, "contract_change": False}],
        "risks": [{"detail": "reports can be up to 30 s stale", "severity": "medium"}, {"detail": "the refresher thread dying silently serves stale data forever", "severity": "medium"}],
        "new_deps": [], "new_service": False, "config_keys": [], "schema_changes": [], "public_api_changes": [],
        "effort": {"tool_calls": 18, "minutes": 9, "tokens": 55000}, "reversibility": {"score": 4, "why": "pure code, no data migration"}},
}
GOAL_FIT = {"library-choice": 6, "infrastructure": 7, "api-contract": 9, "do-less": 7}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="interpreter with fastapi, httpx and pytest (used for the project's checks)")
    ap.add_argument("--dest", help="demo repo directory (default: a fresh temp dir)")
    ap.add_argument("--state-dir", help="Branchpoint state dir (default: <dest>/../state)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds of pacing per scripted model step (try 2 when recording)")
    ap.add_argument("--serve", action="store_true", help="keep the live view up after the run")
    args = ap.parse_args()
    load_pkg()
    from hermes_branchpoint import paths
    from hermes_branchpoint.agent import AgentResult, CallableRunner
    from hermes_branchpoint.commands import BranchpointService
    from hermes_branchpoint.llm import FakeLLM

    base = Path(args.dest).resolve() if args.dest else Path(tempfile.mkdtemp(prefix="branchpoint-demo-")) / "reports-demo"
    state = Path(args.state_dir).resolve() if args.state_dir else base.parent / "state"
    if base.exists():
        shutil.rmtree(base)
    subprocess.run([sys.executable, str(ROOT / "demo" / "prepare_demo.py"), str(base), "--built", "--python", args.python], check=True, capture_output=True)
    paths.set_data_dir_override(state)
    pace = lambda mult=1.0: time.sleep(args.delay * mult) if args.delay else None  # noqa: E731

    def diverge(system, user):
        pace()
        return json.dumps({"strategies": STRATS})

    def baseline(system, user):
        pace()
        return json.dumps({"strategies": [BASELINE]})

    def simulate(system, user):
        pace(1.5)
        axis = re.search(r"\(axis: ([\w-]+)\)", user).group(1)
        return json.dumps(PLANS[axis])

    def review(system, user):
        pace(1.5)
        persona = "skeptic" if "Skeptic" in system else "maintainer" if "Maintainer" in system else "advocate"
        out = {}
        for block in re.split(r"(?m)^=== ", user)[1:]:
            label = block.split()[0]
            axis = re.search(r"\(axis: ([\w-]+)\)", block).group(1)
            blast = int(re.search(r"Blast radius (\d+)/10", block).group(1))
            files = int(re.search(r"Diff \((?:actual|predicted)\): (\d+) files", block).group(1))
            executed = "EXECUTED" in block.split("\n")[0]
            failing = "[test/high]" in block or "[lint/high]" in block
            correctness = (3 if failing else 8) if executed else 5
            rev = int(re.search(r"Reversibility (\d)/5", block).group(1))
            goal = GOAL_FIT[axis] + (1 if persona == "advocate" and axis == "api-contract" else 0)
            ref = (re.search(rf"{label}:test:final", block) if failing else None) or re.search(rf"{label}:diff:h1", block) or re.search(rf"{label}:trace:blast", block)
            if axis == "api-contract":
                ref = re.search(rf"{label}:trace:blast", block) or ref
            flaw = FLAWS["failing" if failing else axis]
            out[label] = {"goal_fit": goal, "correctness": correctness, "blast_radius": 10 - blast, "simplicity": max(2, 9 - files * (2 if persona == "maintainer" else 1)),
                          "reversibility": rev * 2, "cost": 6, "rationale": f"{flaw} [{ref.group(0)}]",
                          "worst_flaw": {"text": flaw, "refs": [ref.group(0)] if ref else []}}
        return json.dumps({"scores": out})

    def judge(system, user):
        pace(2)
        order = re.findall(r"#\d+ (\w+) score", re.search(r"FINAL RANKING[^:]*: (.*)", user).group(1))
        win, losers = order[0], order[1:]
        axis_of = {m.group(1): m.group(2) for m in re.finditer(r"(?m)^=== (\w+)[^\n]*\nStrategy \(axis: ([\w-]+)\)", user)}
        files_of = {m.group(1): m.group(2) for m in re.finditer(r"(?ms)^=== (\w+).*?Diff \((?:actual|predicted)\): (\d+) files", user)}
        has = lambda ref: bool(re.search(re.escape(ref), user))  # noqa: E731
        refs = lambda lab, *kinds: [f"{lab}:{k}" for k in kinds if has(f"{lab}:{k}")]  # noqa: E731
        wa = axis_of.get(win, "")
        d = {"winner_why": [{"text": WIN_WHY.get(wa, f"Route {win} scored highest overall.").format(win=win, files=files_of.get(win, "?")), "refs": refs(win, "trace:blast", "test:final")}],
             "losers": {}, "risks_carried": [], "borrow": [], "confidence": "medium"}
        if wa in RISK:
            d["risks_carried"] = [{"text": RISK[wa], "refs": refs(win, "trace:blast")}]
        for l in losers:
            la = axis_of.get(l, "")
            d["losers"][l] = [{"text": LOSE_WHY.get(la, f"Route {l} scored lower overall.").format(files=files_of.get(l, "?")),
                               "refs": refs(l, "test:final", "trace:blast") if la == "infrastructure" else refs(l, "diff:h1", "trace:blast")[:2]}]
        pick = next((l for l in losers if axis_of.get(l) in BORROW), None)
        if pick:
            d["borrow"] = [{"from": pick, "text": BORROW[axis_of[pick]], "refs": refs(pick, "trace:blast")}]
        return json.dumps(d)

    def builder(job):
        pace(3)
        wt = Path(job.worktree)
        axis = re.search(r"Axis: ([\w-]+)", job.prompt).group(1)
        R = ROOT / "demo" / "routes"
        put = lambda src, dst: (wt / dst).parent.mkdir(parents=True, exist_ok=True) or shutil.copy(R / src, wt / dst)  # noqa: E731
        if axis == "infrastructure":
            put("redis_cache.py", "app/cache.py"); put("redis_reports.py", "app/reports.py"); put("redis_test.py", "tests/test_cache.py")
            (wt / "docker-compose.yml").write_text("services:\n  redis:\n    image: redis:7\n    ports: ['6379:6379']\n")
            (wt / "requirements.txt").write_text("fastapi>=0.110,<1\nredis>=5,<6\n")
            summary = "Added a Redis cache module and a compose service. The tests need the redis package and a REDIS_URL."
        elif axis == "api-contract":
            put("etag_main.py", "app/main.py"); put("etag_test.py", "tests/test_etag.py")
            summary = "Reports now carry an ETag and Cache-Control; a matching If-None-Match gets a 304."
        else:
            put("precompute.py", "app/precompute.py"); put("precompute_reports.py", "app/reports.py"); put("precompute_main.py", "app/main.py"); put("precompute_test.py", "tests/test_precompute.py")
            summary = "A refresher thread rebuilds every report each 30 s; requests only read the table."
        return AgentResult(0, summary, "", 1.0)

    script = {"baseline": baseline, "diverge": diverge, "distinct": json.dumps({"duplicates": []}), "judge": judge, "judge-repair": judge,
              **{f"review:{p}": review for p in ("Maintainer", "Skeptic", "User Advocate")}, **{f"simulate:{r}": simulate for r in "BCDE"}}
    svc = BranchpointService(None, llm=FakeLLM(script), agent=CallableRunner(builder))
    svc.cwd = lambda: str(base)  # type: ignore[method-assign]
    svc.announce = lambda msg: print(msg, flush=True)
    import os
    os.chdir(base)
    print("SCRIPTED MODEL REPLIES: real git worktrees, real pytest runs, real blast tracing and council code.\n")
    text = svc.handle('rewind 1 --base HEAD --goal "add caching to the slow GET /report/{customer_id} endpoint" --n 4 --execute all --wait --budget 5m')
    print(text)
    print(f"\nDemo repo: {base}\nState:     {state}\nNext:      cd {base} && python {ROOT}/scripts/branchpoint --state-dir {state} adopt C merge")
    if args.serve and svc.server is not None:
        print("\nLive view is up; Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
