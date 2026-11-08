"""Council: three blind reviewers + a judge turn the route ledgers into a cited Decision Record.

* Reviewers never see route letters: each reviewer gets its own random R1..Rn order.
* Scores are aggregated in code with the fixed weights; the judge model only writes the narrative.
* Every sentence of the narrative must cite ledger evidence; refs are checked against the real evidence
  store and uncited or unresolvable sentences are dropped (never published).
"""
from __future__ import annotations

import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import REF_RE, Ledger, RunEvidence, find_refs, parse_ref
from .llm import LLM, BudgetExceeded, redact_for_model
from .util import extract_json, iso, now, short, slugify, write_json, atomic_write

WEIGHTS = {"goal_fit": 0.30, "correctness": 0.25, "blast_radius": 0.20, "simplicity": 0.10, "reversibility": 0.10, "cost": 0.05}
CRITERIA = tuple(WEIGHTS)
SIMULATED_CORRECTNESS_CAP = 6

PERSONAS = {
    "Maintainer": (
        "You are the Maintainer: the engineer who will own this code for the next two years. You care most about "
        "simplicity and blast radius: how much must be understood, operated and kept working, and what breaks "
        "elsewhere when this ships."
    ),
    "Skeptic": (
        "You are the Skeptic. You hunt for bugs, edge cases and hidden risk. You trust only evidence in the ledger "
        "(test output, diffs, traces). What the ledger does NOT show (untested paths, unverified claims, "
        "predicted-only risks) counts against a route."
    ),
    "User Advocate": (
        "You are the User Advocate. You judge fit to the stated goal and what users and API clients actually "
        "experience: behaviour, contract changes, latency or freshness trade-offs, and whether it does what was asked."
    ),
}

RUBRIC = """Score EVERY route 1-10 on each criterion (10 is best):
- goal_fit: does it deliver the stated goal, including later requests?
- correctness: how strong is the EVIDENCE that it works? Executed routes have test/lint/type output; routes marked
  "simulated" only have a plan and predicted risks, so cap their correctness at 6.
- blast_radius: 10 = tiny, contained change; 1 = touches public API, schema, many dependents, new infrastructure.
- simplicity: 10 = least new concepts, code and moving parts.
- reversibility: 10 = a plain revert undoes it; 1 = hard to undo later.
- cost: 10 = cheap to build and run.
For each route also name its single WORST FLAW and cite evidence for it. Cite evidence refs exactly as written
(for example R2:test:final). Reply with JSON only:
{"scores":{"R1":{"goal_fit":7,"correctness":6,"blast_radius":8,"simplicity":7,"reversibility":9,"cost":6,
"rationale":"<1-2 sentences with refs>","worst_flaw":{"text":"<one sentence>","refs":["R1:diff:h1"]}}}}"""


# --------------------------------------------------------------------------- rendering

def _fidelity_label(l: Ledger) -> str:
    return {"simulate": "SIMULATED (plan + static analysis; nothing was built)",
            "execute": "EXECUTED (built in an isolated worktree; checks run)",
            "baseline": "EXECUTED (already built earlier in the session; checks run on its snapshot)"}.get(l.fidelity, l.fidelity)


def _evidence_index(l: Ledger, rev: RunEvidence, limit: int = 30) -> list[tuple[str, str]]:
    items = json.loads((rev.run_dir / "routes" / l.route / "evidence.json").read_text(encoding="utf-8")) if (rev.run_dir / "routes" / l.route / "evidence.json").exists() else {}
    order = {"test": 0, "lint": 1, "type": 2, "trace": 3, "stat": 4, "diff": 5, "plan": 6, "issue": 7, "dep": 8, "event": 9, "runtime": 10}
    keys = sorted(items, key=lambda k: (order.get(items[k]["kind"], 99), k))
    return [(k, items[k]["title"]) for k in keys[:limit]]


def render_ledger(l: Ledger, label: str, rev: RunEvidence) -> str:
    def rw(text: str) -> str:
        return re.sub(rf"\b{re.escape(l.route)}:(?=(?:diff|test|lint|type|trace|plan|issue|event|stat|dep|runtime):)", f"{label}:", text)

    b = l.blast
    lines = [f"=== {label}  [{_fidelity_label(l)}] ===",
             f"Strategy (axis: {l.strategy.get('axis')}): {l.strategy.get('thesis')}",
             "Key decisions: " + "; ".join(l.strategy.get("key_decisions") or []),
             f"Summary: {l.summary}",
             f"Diff ({'predicted' if l.diff_stats.get('predicted') else 'actual'}): {l.diff_stats.get('files')} files, +{l.diff_stats.get('insertions')} -{l.diff_stats.get('deletions')}  "
             f"files: " + ", ".join(f"{f['path']}({f.get('action')})" for f in l.files[:12]),
             f"Blast radius {b.score}/10 [{label}:trace:blast]: dependents={b.dependents}; public_api={b.public_api or 'none'}; schema={b.schema or 'none'}; config={b.config or 'none'}; new deps={l.deps_added or 'none'}",
             f"Issues ({len(l.issues)}):"]
    for i in l.issues[:14]:
        lines.append(f"  - [{i.kind}/{i.severity}] {short(i.detail, 200)}  [{i.evidence_ref}]")
    if not l.issues:
        lines.append("  (none recorded)")
    c = l.cost
    lines.append(f"Cost ({'estimated' if c.estimated else 'measured'}): {c.tokens} tokens, {c.seconds:.0f}s, {c.tool_calls} tool calls")
    lines.append(f"Reversibility {l.reversibility}/5: {l.reversibility_notes}")
    if l.notes:
        lines.append("Notes: " + " | ".join(short(n, 160) for n in l.notes[:4]))
    lines.append("Evidence you may cite: " + "; ".join(f"{k} ({short(t, 70)})" for k, t in _evidence_index(l, rev)))
    return rw("\n".join(lines))


def _unmap(text: str, back: dict[str, str]) -> str:
    return re.sub(r"\bR\d+:(?=(?:diff|test|lint|type|trace|plan|issue|event|stat|dep|runtime):)", lambda m: back.get(m.group(0)[:-1], m.group(0)[:-1]) + ":", text)


# --------------------------------------------------------------------------- reviewers

def _clamp(v: Any) -> int | None:
    try:
        return max(1, min(10, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def _fallback_scores(l: Ledger) -> dict[str, int]:
    """Deterministic scores from ledger facts, used only when a reviewer reply is unusable."""
    high = len([i for i in l.issues if i.severity == "high" and i.kind != "predicted"])
    pred_high = len([i for i in l.issues if i.severity == "high" and i.kind == "predicted"])
    corr = max(1, 8 - 2 * high) if l.fidelity != "simulate" else max(2, 5 - pred_high)
    return {"goal_fit": 5, "correctness": corr, "blast_radius": max(1, 10 - l.blast.score), "simplicity": max(1, 9 - len(l.files)),
            "reversibility": max(1, min(10, l.reversibility * 2)), "cost": 5}


@dataclass
class ReviewerResult:
    persona: str
    scores: dict[str, dict[str, int]]          # real route id -> criterion -> score
    flaws: dict[str, dict[str, Any]]           # real route id -> {text, refs}
    rationale: dict[str, str]
    fallback_routes: list[str] = field(default_factory=list)
    raw: str = ""


def run_reviewer(llm: LLM, persona: str, ledgers: list[Ledger], rev: RunEvidence, goal: str, rng: random.Random) -> ReviewerResult:
    order = ledgers[:]
    rng.shuffle(order)
    labels = {l.route: f"R{i + 1}" for i, l in enumerate(order)}
    back = {v: k for k, v in labels.items()}
    body = "\n\n".join(render_ledger(l, labels[l.route], rev) for l in order)
    user = f"GOAL:\n{redact_for_model(goal)}\n\nLEDGERS (route names are anonymised):\n\n{body}\n\n{RUBRIC}"
    raw = ""
    data: dict[str, Any] = {}
    for attempt in range(2):
        reply = llm.complete(PERSONAS[persona] + " Review blind: you do not know which route is whose.", user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON for this schema. Reply with the JSON object only.",
                             max_tokens=3500, purpose=f"review:{persona}")
        raw = reply.text
        try:
            data = extract_json(raw)
            if isinstance(data, dict) and isinstance(data.get("scores"), dict):
                break
        except ValueError:
            data = {}
    scores: dict[str, dict[str, int]] = {}
    flaws: dict[str, dict[str, Any]] = {}
    rationale: dict[str, str] = {}
    fb: list[str] = []
    raw_scores = (data.get("scores") if isinstance(data, dict) else None) or {}
    by_id = {l.route: l for l in ledgers}
    for label, entry in raw_scores.items():
        rid = back.get(str(label).strip())
        if rid is None or not isinstance(entry, dict):
            continue
        row = {c: _clamp(entry.get(c)) for c in CRITERIA}
        if any(v is None for v in row.values()):
            fb_row = _fallback_scores(by_id[rid])
            row = {c: (row[c] if row[c] is not None else fb_row[c]) for c in CRITERIA}
            fb.append(rid)
        if by_id[rid].fidelity == "simulate":
            row["correctness"] = min(row["correctness"], SIMULATED_CORRECTNESS_CAP)  # type: ignore[type-var]
        scores[rid] = {c: int(v) for c, v in row.items()}  # type: ignore[arg-type]
        wf = entry.get("worst_flaw") if isinstance(entry.get("worst_flaw"), dict) else {"text": str(entry.get("worst_flaw") or ""), "refs": []}
        refs = [_unmap(r, back) for r in (wf.get("refs") or []) if isinstance(r, str)]
        refs += find_refs(_unmap(str(wf.get("text") or ""), back))
        flaws[rid] = {"text": _unmap(short(wf.get("text") or "", 300), back), "refs": list(dict.fromkeys(refs))}
        rationale[rid] = _unmap(short(entry.get("rationale") or "", 400), back)
    for l in ledgers:
        if l.route not in scores:
            scores[l.route] = _fallback_scores(l)
            fb.append(l.route)
            flaws.setdefault(l.route, {"text": "", "refs": []})
    return ReviewerResult(persona, scores, flaws, rationale, fb, raw)


# --------------------------------------------------------------------------- aggregation

def aggregate(ledgers: list[Ledger], reviews: list[ReviewerResult], weights: dict[str, float] = WEIGHTS) -> list[dict[str, Any]]:
    rows = []
    for l in ledgers:
        per = {c: round(sum(r.scores[l.route][c] for r in reviews) / len(reviews), 2) for c in CRITERIA}
        total = round(sum(per[c] * weights[c] for c in CRITERIA), 2)
        rows.append({"route": l.route, "score": total, "criteria": per, "blast": l.blast.score, "files": l.diff_stats.get("files", 0),
                     "fidelity": l.fidelity, "name": l.strategy.get("name", ""),
                     "reviewer_totals": {r.persona: round(sum(r.scores[l.route][c] * weights[c] for c in CRITERIA), 2) for r in reviews}})
    # ties (equal to 2 dp) go to the smaller blast radius, then fewer files
    rows.sort(key=lambda r: (-r["score"], r["blast"], r["files"], r["route"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# --------------------------------------------------------------------------- judge

JUDGE_SYSTEM = (
    "You are the Judge of a design council. The ranking is already computed from the reviewers' scores and is final; "
    "you explain it. Every single claim must cite evidence refs exactly as they appear (e.g. C:test:final). "
    "Never state a fact you cannot cite. Be specific and brief. Reply with JSON only."
)

JUDGE_SCHEMA = (
    '{"winner_why":[{"text":"<sentence>","refs":["C:test:final"]}],'
    '"losers":{"<route id>":[{"text":"<why it lost>","refs":["B:trace:blast"]}]},'
    '"risks_carried":[{"text":"<risk the winner still carries>","refs":["C:issue:p1"]}],'
    '"borrow":[{"from":"<route id>","text":"<idea worth taking from a losing route>","refs":["B:diff:h2"]}],'
    '"confidence":"high|medium|low","confidence_note":"<one sentence, cite refs>"}'
)


def _valid_sentence(item: Any, rev: RunEvidence) -> tuple[bool, dict[str, Any]]:
    if not isinstance(item, dict):
        return False, {}
    text = str(item.get("text") or "").strip()
    refs = [r.strip() for r in (item.get("refs") or []) if isinstance(r, str)]
    refs += [r for r in find_refs(text) if r not in refs]
    good = [r for r in refs if parse_ref(r) and rev.exists(r)]
    bad = [r for r in refs if r not in good]
    return bool(text and good and not bad), {"text": text, "refs": good, "bad_refs": bad}


def _validate_narrative(data: dict[str, Any], rev: RunEvidence) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dropped: list[dict[str, Any]] = []
    out: dict[str, Any] = {"winner_why": [], "losers": {}, "risks_carried": [], "borrow": []}

    def keep(items: Any, extra_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        kept = []
        for it in items or []:
            ok, clean = _valid_sentence(it, rev)
            if ok:
                for k in extra_keys:
                    clean[k] = it.get(k)
                kept.append(clean)
            else:
                dropped.append({"text": (it.get("text") if isinstance(it, dict) else str(it)), "reason": "missing or unresolvable evidence ref",
                                "bad_refs": clean.get("bad_refs", []) if isinstance(clean, dict) else []})
        return kept

    out["winner_why"] = keep(data.get("winner_why"))
    for rid, items in (data.get("losers") or {}).items():
        out["losers"][str(rid)] = keep(items)
    out["risks_carried"] = keep(data.get("risks_carried"))
    out["borrow"] = keep(data.get("borrow"), ("from",))
    return out, dropped


def _deterministic_facts(l: Ledger) -> list[dict[str, Any]]:
    """Always-true, always-cited sentences built straight from the ledger."""
    facts: list[dict[str, Any]] = []
    b = l.blast
    facts.append({"text": f"Route {l.route} has a blast radius of {b.score}/10 ({b.dependents} dependent file(s), "
                          f"{len(b.public_api)} public-API change(s), {len(l.deps_added)} new dependenc{'y' if len(l.deps_added) == 1 else 'ies'}).",
                  "refs": [b.evidence_ref] if b.evidence_ref else []})
    chk = [i for i in l.issues if i.kind in ("test", "lint", "type") and i.severity == "high"]
    if l.fidelity != "simulate":
        ref = next((r for r in (f"{l.route}:test:final", f"{l.route}:lint:final", f"{l.route}:type:final") if r), "")
        if chk:
            facts.append({"text": f"Route {l.route} finished with {len(chk)} failing check(s), first: {short(chk[0].detail, 120)}.", "refs": [chk[0].evidence_ref]})
        else:
            facts.append({"text": f"Route {l.route} finished with no failing checks.", "refs": [ref]})
    else:
        pr = [i for i in l.issues if i.kind == "predicted"]
        if pr:
            facts.append({"text": f"Route {l.route} was only simulated; its top predicted risk is: {short(pr[0].detail, 140)}.", "refs": [pr[0].evidence_ref]})
    return facts


def run_judge(llm: LLM, ledgers: list[Ledger], ranking: list[dict[str, Any]], reviews: list[ReviewerResult],
              rev: RunEvidence, goal: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    by_id = {l.route: l for l in ledgers}
    winner, runner = ranking[0]["route"], (ranking[1]["route"] if len(ranking) > 1 else None)
    flaws = {rid: [r.flaws[rid] for r in reviews if r.flaws.get(rid, {}).get("text")] for rid in by_id}
    prompt = "\n\n".join([
        f"GOAL:\n{redact_for_model(goal)}",
        "FINAL RANKING (computed, do not change): " + "; ".join(f"#{r['rank']} {r['route']} score {r['score']}" for r in ranking),
        "REVIEWER WORST-FLAW NOTES (route ids are real here):\n" + "\n".join(f"{rid}: " + " | ".join(f['text'] for f in fl) for rid, fl in flaws.items()),
        "LEDGERS:\n\n" + "\n\n".join(render_ledger(l, l.route, rev) for l in ledgers),
        f"Write the decision narrative: why {winner} won" + (f" (runner-up {runner})" if runner else "") + ", why each other route lost, "
        "risks the winner still carries, and ideas worth borrowing from losing routes. "
        f"Cite only refs listed under 'Evidence you may cite'. Reply with JSON only: {JUDGE_SCHEMA}",
    ])
    reply = llm.complete(JUDGE_SYSTEM, prompt, max_tokens=3500, purpose="judge")
    raw = reply.text
    try:
        data = extract_json(raw)
    except ValueError:
        data = {}
    narrative, dropped = _validate_narrative(data if isinstance(data, dict) else {}, rev)
    if dropped or not narrative["winner_why"]:
        repair = (prompt + "\n\nThese sentences were REJECTED because their evidence refs were missing or did not exist:\n"
                  + "\n".join(f"- {d['text']} (bad refs: {d['bad_refs']})" for d in dropped[:12])
                  + "\nRe-issue the full JSON. Every sentence needs at least one ref from the allowed lists.")
        try:
            reply2 = llm.complete(JUDGE_SYSTEM, repair, max_tokens=3500, purpose="judge-repair")
            data2 = extract_json(reply2.text)
            narrative2, dropped2 = _validate_narrative(data2 if isinstance(data2, dict) else {}, rev)
            if len(narrative2["winner_why"]) >= len(narrative["winner_why"]):
                narrative, dropped = narrative2, dropped2
        except (ValueError, BudgetExceeded):
            pass
    conf = str((data or {}).get("confidence") or "medium").lower() if isinstance(data, dict) else "medium"
    narrative["confidence"] = conf if conf in ("high", "medium", "low") else "medium"
    ok, note = _valid_sentence({"text": (data or {}).get("confidence_note", "") if isinstance(data, dict) else "", "refs": []}, rev)
    narrative["confidence_note"] = note["text"] if ok else ""
    return narrative, dropped, raw


# --------------------------------------------------------------------------- decision record

def _snippet(rev: RunEvidence, ref: str, lines: int = 14) -> str:
    item = rev.resolve(ref)
    if not item:
        return ""
    text = item.get("text", "")
    parts = text.splitlines()
    body = "\n".join(parts[:lines]) + (f"\n… ({len(parts) - lines} more lines)" if len(parts) > lines else "")
    return f"**`{ref}`** — {item.get('title')}\n```\n{body}\n```"


def _cite(refs: list[str]) -> str:
    return " " + " ".join(f"[`{r}`]" for r in refs) if refs else ""


def render_decision(*, goal: str, run_id: str, mode: str, ledgers: list[Ledger], ranking: list[dict[str, Any]],
                    reviews: list[ReviewerResult], narrative: dict[str, Any], dropped: list[dict[str, Any]],
                    rev: RunEvidence, failed: list[Ledger], warnings: list[str]) -> str:
    by_id = {l.route: l for l in ledgers}
    win = ranking[0]
    wl = by_id[win["route"]]
    runner = ranking[1] if len(ranking) > 1 else None
    cited: list[str] = []

    def add(refs: list[str]) -> None:
        for r in refs:
            if r not in cited:
                cited.append(r)

    out = [f"# Decision Record — {short(goal, 90)}", "",
           f"*Branchpoint run `{run_id}` · {iso()} · {mode}*", "",
           "## Decision", "",
           f"**Adopt Route {win['route']} — {wl.strategy.get('name')}** (weighted score {win['score']}/10, confidence: {narrative.get('confidence', 'medium')})."]
    if narrative.get("confidence_note"):
        out.append(narrative["confidence_note"])
    if runner:
        out.append(f"Runner-up: Route {runner['route']} — {runner['name']} ({runner['score']}/10).")
    if any(l.fidelity == "simulate" for l in ledgers):
        sim = ", ".join(l.route for l in ledgers if l.fidelity == "simulate")
        out.append(f"> Routes {sim} were simulated, not built: their correctness is capped at {SIMULATED_CORRECTNESS_CAP}/10 and rests on a plan plus static analysis.")
    out += ["", "## Ranking", "", "| # | Route | Approach | Score | Blast | Issues | Cost | Fidelity |", "|---|---|---|---|---|---|---|---|"]
    for r in ranking:
        l = by_id[r["route"]]
        c = l.cost
        out.append(f"| {r['rank']} | {r['route']} | {short(r['name'], 40)} | **{r['score']}** | {l.blast.score}/10 [`{l.blast.evidence_ref}`] | {len(l.issues)} | "
                   f"{c.tokens // 1000}k tok / {c.seconds:.0f}s{' (est.)' if c.estimated else ''} | {l.fidelity} |")
        add([l.blast.evidence_ref] if l.blast.evidence_ref else [])
    out += ["", "Weights: goal fit 30%, correctness 25%, blast radius 20%, simplicity 10%, reversibility 10%, cost 5%. "
            "Reviewers (Maintainer, Skeptic, User Advocate) scored blind; ties go to the smaller blast radius.", "",
            "| Route | " + " | ".join(c.replace("_", " ") for c in CRITERIA) + " |", "|---|" + "---|" * len(CRITERIA)]
    for r in ranking:
        out.append(f"| {r['route']} | " + " | ".join(str(r["criteria"][c]) for c in CRITERIA) + " |")

    out += ["", f"## Why Route {win['route']} won", ""]
    for s in narrative["winner_why"] or _deterministic_facts(wl):
        out.append(f"- {s['text']}{_cite(s['refs'])}")
        add(s["refs"])

    out += ["", "## Why the others lost", ""]
    for r in ranking[1:]:
        rid = r["route"]
        l = by_id[rid]
        flaw = next((rv.flaws[rid] for rv in reviews if rv.flaws.get(rid, {}).get("text") and any(parse_ref(x) and rev.exists(x) for x in rv.flaws[rid]["refs"])), None)
        out.append(f"### Route {rid} — {l.strategy.get('name')} (#{r['rank']}, {r['score']}/10)")
        if flaw:
            refs = [x for x in flaw["refs"] if parse_ref(x) and rev.exists(x)]
            out.append(f"- **Worst flaw:** {flaw['text']}{_cite(refs)}")
            add(refs)
        for s in narrative["losers"].get(rid) or _deterministic_facts(l):
            out.append(f"- {s['text']}{_cite(s['refs'])}")
            add(s["refs"])
        out.append("")
    wflaw = next((rv.flaws[win['route']] for rv in reviews if rv.flaws.get(win['route'], {}).get("text")), None)
    out += ["## Risks carried forward", ""]
    risks = narrative["risks_carried"]
    if wflaw and wflaw["text"]:
        refs = [x for x in wflaw["refs"] if parse_ref(x) and rev.exists(x)]
        if refs:
            out.append(f"- **Worst flaw of the winner (reviewers):** {wflaw['text']}{_cite(refs)}")
            add(refs)
    for s in risks:
        out.append(f"- {s['text']}{_cite(s['refs'])}")
        add(s["refs"])
    if not risks and not (wflaw and wflaw["text"]):
        for i in [i for i in wl.issues if i.severity != "low"][:4]:
            out.append(f"- {short(i.detail, 200)} [`{i.evidence_ref}`]")
            add([i.evidence_ref])
    out += ["", "## Ideas worth borrowing from the other routes", ""]
    if narrative["borrow"]:
        for s in narrative["borrow"]:
            out.append(f"- From Route {s.get('from')}: {s['text']}{_cite(s['refs'])}")
            add(s["refs"])
    else:
        out.append("- None identified with cited evidence.")
    if failed:
        out += ["", "## Routes that did not finish", ""]
        out += [f"- Route {l.route} — {l.strategy.get('name')}: {short(l.summary, 160)}" for l in failed]
    if dropped or warnings:
        out += ["", "## Integrity notes", ""]
        out += [f"- {w}" for w in warnings]
        if dropped:
            out.append(f"- {len(dropped)} narrative sentence(s) were withheld because they had no resolvable evidence ref.")
    out += ["", "## Evidence", "", "Every `[ref]` above resolves to a stored snippet (route ledgers under the run directory).", ""]
    for r in cited:
        sn = _snippet(rev, r)
        if sn:
            out += [sn, ""]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- entry point

@dataclass
class CouncilResult:
    winner: str
    runner_up: str | None
    ranking: list[dict[str, Any]]
    decision_md: str
    narrative: dict[str, Any]
    reviews: list[ReviewerResult]
    dropped: list[dict[str, Any]]
    warnings: list[str]


def run_council(llm: LLM, ledgers: list[Ledger], rev: RunEvidence, *, goal: str, run_id: str, mode: str,
                run_dir: Path, seed: int | None = None) -> CouncilResult:
    done = [l for l in ledgers if l.status == "done"]
    failed = [l for l in ledgers if l.status != "done"]
    if not done:
        raise RuntimeError("no route finished, so there is nothing for the council to judge")
    warnings: list[str] = []
    if len(done) == 1:
        warnings.append("only one route finished; the 'decision' is the only candidate, not a comparison")
    rng_seed = seed if seed is not None else int.from_bytes(run_id.encode()[-4:].ljust(4, b"0"), "big")
    results: list[ReviewerResult | None] = [None] * len(PERSONAS)
    errors: list[str] = []

    def work(i: int, persona: str) -> None:
        try:
            results[i] = run_reviewer(llm, persona, done, rev, goal, random.Random(rng_seed + i * 7919))
        except BudgetExceeded as exc:
            errors.append(f"{persona}: {exc}")
        except Exception as exc:
            errors.append(f"{persona}: {short(exc, 160)}")

    threads = [threading.Thread(target=work, args=(i, p), daemon=True) for i, p in enumerate(PERSONAS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    reviews = [r for r in results if r is not None]
    if not reviews:
        raise RuntimeError("no reviewer produced scores: " + "; ".join(errors))
    if errors:
        warnings.append("reviewer(s) unavailable: " + "; ".join(errors))
    for r in reviews:
        if r.fallback_routes:
            warnings.append(f"{r.persona}: unusable scores for route(s) {', '.join(r.fallback_routes)}; ledger-derived scores used")
    ranking = aggregate(done, reviews)
    try:
        narrative, dropped, raw = run_judge(llm, done, ranking, reviews, rev, goal)
    except BudgetExceeded as exc:
        narrative, dropped, raw = {"winner_why": [], "losers": {}, "risks_carried": [], "borrow": [], "confidence": "low", "confidence_note": ""}, [], ""
        warnings.append(f"judge unavailable ({short(exc, 100)}); narrative generated from ledger facts only")
    md = render_decision(goal=goal, run_id=run_id, mode=mode, ledgers=done, ranking=ranking, reviews=reviews, narrative=narrative,
                         dropped=dropped, rev=rev, failed=failed, warnings=warnings)
    cdir = run_dir / "council"
    for r in reviews:
        write_json(cdir / f"review-{slugify(r.persona)}.json", {"persona": r.persona, "scores": r.scores, "flaws": r.flaws, "rationale": r.rationale, "fallback": r.fallback_routes})
    write_json(cdir / "aggregate.json", ranking)
    write_json(cdir / "judge.json", {"narrative": narrative, "dropped": dropped})
    atomic_write(run_dir / "decision.md", md)
    write_json(run_dir / "decision.json", {"run_id": run_id, "winner": ranking[0]["route"], "runner_up": ranking[1]["route"] if len(ranking) > 1 else None,
                                           "ranking": ranking, "narrative": narrative, "warnings": warnings, "created": now()})
    return CouncilResult(ranking[0]["route"], ranking[1]["route"] if len(ranking) > 1 else None, ranking, md, narrative, reviews, dropped, warnings)
