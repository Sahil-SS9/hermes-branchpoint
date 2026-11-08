"""Diverge: turn a goal into N *substantively different* build directions.

Diversity is enforced, not hoped for: every strategy sits on its own axis, near-duplicates are rejected
by a lexical similarity check (TF-IDF cosine over thesis + decisions, blended with predicted-file
overlap) plus an optional LLM "same approach?" check, and rejected strategies are regenerated.

Note: the plugin API exposes chat completions only, no embeddings. The spec's embedding threshold
(cosine > 0.88) therefore becomes a lexical threshold (default 0.55) backed by the axis-uniqueness rule.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import LLM, redact_for_model
from .util import extract_json, in_skipped_dir, is_secret_path, run, short

AXES = ("library-choice", "data-model", "logic-placement", "sync-vs-async", "do-less", "infrastructure", "api-contract", "build-vs-buy")
DEFAULT_THRESHOLD = 0.55
_STOP = set("the a an and or of to in on for with by from is are be this that it as at use using add new via into over under per not no".split())


# --------------------------------------------------------------------------- repo map

def _py_outline(src: str) -> str:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return ""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            names.append(f"class {node.name}")
    return ", ".join(names[:14])


def repo_map(root: str | Path, max_chars: int = 7000) -> str:
    """Compact map of the repo: source files with line counts and top-level symbols."""
    root = Path(root)
    res = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], root, timeout=30)
    files = sorted(f for f in res.out.split("\0") if f and not in_skipped_dir(f) and not is_secret_path(f)) if res.ok else []
    lines: list[str] = []
    for f in files:
        p = root / f
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > 400_000:
            lines.append(f"{f}  ({size // 1024} KB, skipped)")
            continue
        outline = ""
        n = 0
        try:
            if f.endswith(".py"):
                outline = _py_outline(p.read_text(encoding="utf-8", errors="replace"))
            n = sum(1 for _ in p.open("rb"))
        except OSError:
            pass
        lines.append(f"{f}  ({n} lines)" + (f"  defs: {outline}" if outline else ""))
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + f"\n… ({len(lines)} files total, map truncated)"


def read_excerpts(root: str | Path, paths: list[str], max_chars: int = 9000) -> str:
    """File contents for the planner, secrets excluded, bounded."""
    root = Path(root)
    out: list[str] = []
    budget = max_chars
    for rel in paths:
        rel = rel.lstrip("./")
        if is_secret_path(rel) or in_skipped_dir(rel):
            continue
        p = root / rel
        try:
            if not p.is_file() or p.stat().st_size > 200_000:
                continue
            body = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = f"### {rel}\n{body}"
        if len(chunk) > budget:
            chunk = chunk[: max(0, budget)] + "\n… (truncated)"
        out.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    return "\n\n".join(out)


# --------------------------------------------------------------------------- similarity

def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in _STOP]


def strategy_text(s: dict[str, Any]) -> str:
    return " ".join([str(s.get("axis", "")), str(s.get("thesis", "")), " ".join(map(str, s.get("key_decisions") or []))])


def _tfidf(docs: list[list[str]]) -> list[dict[str, float]]:
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    vecs: list[dict[str, float]] = []
    for d in docs:
        tf = Counter(d)
        vecs.append({t: (c / max(1, len(d))) * (math.log((1 + n) / (1 + df[t])) + 1.0) for t, c in tf.items()})
    return vecs


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {x.lstrip("./") for x in a}, {x.lstrip("./") for x in b}
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def similarity_matrix(strategies: list[dict[str, Any]]) -> list[list[float]]:
    vecs = _tfidf([tokenize(strategy_text(s)) for s in strategies])
    n = len(strategies)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cos = _cos(vecs[i], vecs[j])
            fj = _jaccard(list(strategies[i].get("predicted_files") or []), list(strategies[j].get("predicted_files") or []))
            m[i][j] = m[j][i] = round(0.8 * cos + 0.2 * fj, 4)
    return m


def find_duplicates(strategies: list[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> list[tuple[int, int, float, str]]:
    """Pairs (i, j, score, why) that are too alike (same axis, or similarity >= threshold)."""
    sim = similarity_matrix(strategies)
    out: list[tuple[int, int, float, str]] = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            ai = str(strategies[i].get("axis", "")).strip().lower()
            aj = str(strategies[j].get("axis", "")).strip().lower()
            if ai and ai == aj:
                out.append((i, j, max(sim[i][j], 1.0), f"same axis '{ai}'"))
            elif sim[i][j] >= threshold:
                out.append((i, j, sim[i][j], f"similarity {sim[i][j]:.2f} >= {threshold:.2f}"))
    return out


# --------------------------------------------------------------------------- generation

SYSTEM = (
    "You are a staff engineer exploring architecture options before any code is written. "
    "Propose build directions that are genuinely different in substance, not cosmetic variants. "
    "Each direction must sit on a DIFFERENT axis. Allowed axes: " + ", ".join(AXES) + ". "
    "Be concrete: name libraries, files and where the logic lives. One direction may be 'do-less' "
    "(the smallest change that meets the goal). Reply with JSON only."
)

SCHEMA_HINT = (
    '{"strategies":[{"axis":"<one allowed axis>","name":"<3-5 words>","thesis":"<one sentence: the bet this route makes>",'
    '"key_decisions":["<concrete decision>", "..."],"predicted_files":["path/to/file.py"],"trade_off":"<what it gives up>"}]}'
)


@dataclass
class DivergeResult:
    strategies: list[dict[str, Any]]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    warnings: list[str] = field(default_factory=list)
    similarity: list[list[float]] = field(default_factory=list)


def _clean(s: dict[str, Any], i: int) -> dict[str, Any]:
    return {
        "axis": str(s.get("axis") or "").strip().lower() or f"axis-{i + 1}",
        "name": short(s.get("name") or s.get("thesis") or f"Route {i + 1}", 60),
        "thesis": short(s.get("thesis") or "", 300),
        "key_decisions": [short(x, 200) for x in (s.get("key_decisions") or [])][:8],
        "predicted_files": [str(x) for x in (s.get("predicted_files") or [])][:15],
        "trade_off": short(s.get("trade_off") or "", 200),
    }


def _parse(text: str) -> list[dict[str, Any]]:
    data = extract_json(text)
    items = data.get("strategies") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("expected a list of strategies")
    return [x for x in items if isinstance(x, dict) and (x.get("thesis") or x.get("name"))]


def describe_baseline(llm: LLM, goal: str, diff_text: str, diff_stat: str) -> dict[str, Any]:
    """Summarise what the user actually built (route A) in the same shape as a strategy."""
    user = (f"GOAL:\n{redact_for_model(goal)}\n\nWHAT WAS BUILT (diff stat):\n{diff_stat}\n\nDIFF (truncated):\n{redact_for_model(diff_text[:9000])}\n\n"
            f"Describe this implementation as one build direction. Reply with JSON only: {SCHEMA_HINT} "
            "with exactly one strategy.")
    reply = llm.complete(SYSTEM, user, max_tokens=1200, purpose="baseline")
    items = _parse(reply.text)
    if not items:
        raise ValueError("baseline description came back empty")
    return _clean(items[0], 0)


def _distinct_check(llm: LLM, strategies: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    listing = "\n".join(f"{i}. [{s['axis']}] {s['thesis']} | decisions: {'; '.join(s['key_decisions'][:4])}" for i, s in enumerate(strategies))
    user = (f"Here are candidate build directions:\n{listing}\n\n"
            'List pairs that are substantially the SAME approach (a reviewer would call them variants of one idea). '
            'Reply with JSON only: {"duplicates":[{"a":0,"b":1,"why":"..."}]} — use an empty list if all are distinct.')
    try:
        reply = llm.complete("You are a strict reviewer of architecture proposals.", user, max_tokens=600, purpose="distinct")
        data = extract_json(reply.text)
        pairs = data.get("duplicates") if isinstance(data, dict) else None
        out = []
        for p in pairs or []:
            a, b = int(p["a"]), int(p["b"])
            if 0 <= a < len(strategies) and 0 <= b < len(strategies) and a != b:
                out.append((min(a, b), max(a, b), short(p.get("why") or "judged the same approach", 160)))
        return out
    except Exception:
        return []


def generate_strategies(llm: LLM, *, goal: str, n: int, context: str = "", repo_map_text: str = "",
                        baseline: dict[str, Any] | None = None, followups: list[str] | None = None,
                        threshold: float = DEFAULT_THRESHOLD, max_rounds: int = 2, llm_check: bool = True) -> DivergeResult:
    """Return ``n`` distinct strategies (``n - 1`` when a baseline occupies route A)."""
    want = n - 1 if baseline else n
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []
    rounds = 0
    pinned = [baseline] if baseline else []

    def prompt(need: int, avoid: list[dict[str, Any]], note: str = "") -> str:
        parts = [f"GOAL:\n{redact_for_model(goal)}"]
        if followups:
            parts.append("LATER REQUESTS IN THE SAME SESSION (the route must also satisfy these):\n" + "\n".join(f"- {short(redact_for_model(f), 240)}" for f in followups[:6]))
        if context:
            parts.append(f"SESSION CONTEXT BEFORE THE GOAL:\n{context}")
        parts.append(f"REPO MAP:\n{repo_map_text}")
        if baseline:
            parts.append(f"ROUTE A (already built by the user, do NOT repeat it):\naxis={baseline['axis']}; {baseline['thesis']}\ndecisions: {'; '.join(baseline['key_decisions'])}")
        if avoid:
            parts.append("ALREADY CHOSEN (new routes must differ from all of these, including their axes):\n" + "\n".join(f"- [{a['axis']}] {a['thesis']}" for a in avoid))
        if note:
            parts.append(note)
        parts.append(f"Propose exactly {need} build direction(s). Reply with JSON only: {SCHEMA_HINT}")
        return "\n\n".join(parts)

    need = want
    while need > 0 and rounds <= max_rounds:
        avoid = pinned + kept
        note = "The previous attempt returned duplicates; be bolder and pick axes not yet used." if rounds else ""
        reply = llm.complete(SYSTEM, prompt(need if rounds == 0 else need + 1, avoid, note), max_tokens=3500,
                             temperature=0.9 if rounds else None, purpose="diverge")
        try:
            cand = [_clean(x, len(kept) + i) for i, x in enumerate(_parse(reply.text))]
        except ValueError as exc:
            warnings.append(f"round {rounds + 1}: unparseable strategies ({short(exc, 80)})")
            rounds += 1
            continue
        for c in cand:
            trial = pinned + kept + [c]
            dups = [d for d in find_duplicates(trial, threshold) if d[1] == len(trial) - 1]
            if dups:
                c["rejected_because"] = dups[0][3] + f" (vs {trial[dups[0][0]]['name']})"
                rejected.append(c)
                continue
            if len(kept) < want:
                kept.append(c)
        if llm_check and len(pinned) + len(kept) > 1:
            drop: set[int] = set()
            for a, b, why in _distinct_check(llm, pinned + kept):
                victim = b - len(pinned)
                if 0 <= victim < len(kept) and victim not in drop:
                    drop.add(victim)
                    r = dict(kept[victim])
                    r["rejected_because"] = f"judged near-duplicate: {why}"
                    rejected.append(r)
            kept = [k for i, k in enumerate(kept) if i not in drop]
        need = want - len(kept)
        rounds += 1
    if need > 0:
        warnings.append(f"only {len(kept)} of {want} requested routes were distinct enough; the rest were rejected as near-duplicates")
    final = pinned + kept
    return DivergeResult(kept, rejected, rounds, warnings, similarity_matrix(final) if len(final) > 1 else [])
