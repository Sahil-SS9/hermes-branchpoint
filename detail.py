"""Everything the Decision Record view needs, assembled from a run directory (no model calls).

``build_detail`` returns the structured decision, every route ledger, all evidence snippets, and a ``comments``
map: evidence ref -> the justifications that cite it (judge sentences, reviewer flaws/rationales) plus facts the
tracer established for the file a diff hunk touches (public API change, schema, config, new dependency, failing
check). Nothing here is generated: a comment exists only because something in the run said it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .declog import read_entries, summarise
from .ledger import find_refs, parse_ref


def _read(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _clean(text: str) -> str:
    t = re.sub(r"\s*[\[(]`?[A-Z][0-9]?:[a-z]+:[\w\-/#.]+`?[\])]", "", text or "")
    return re.sub(r"\s{2,}", " ", t).strip()


def build_detail(run_dir: str | Path) -> dict[str, Any]:
    rd = Path(run_dir)
    decision = _read(rd / "decision.json", {}) or {}
    md = (rd / "decision.md").read_text(encoding="utf-8") if (rd / "decision.md").exists() else ""
    ledgers: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for p in sorted((rd / "routes").glob("*/ledger.json")):
        ledgers[p.parent.name] = _read(p, {})
    for p in sorted((rd / "routes").glob("*/evidence.json")):
        evidence.update(_read(p, {}) or {})

    comments: dict[str, list[dict[str, str]]] = {}

    def add(ref: str, source: str, text: str, kind: str = "reason") -> None:
        if not parse_ref(ref) or ref not in evidence:
            return
        text = _clean(text)
        if text and not any(c["text"] == text and c["source"] == source for c in comments.setdefault(ref, [])):
            comments[ref].append({"source": source, "text": text, "kind": kind})

    nar = decision.get("narrative") or {}
    win = decision.get("winner")
    for s in nar.get("winner_why") or []:
        for r in s.get("refs") or []:
            add(r, f"Judge · why {win} won", s.get("text", ""), "for")
    for rid, items in (nar.get("losers") or {}).items():
        for s in items or []:
            for r in s.get("refs") or []:
                add(r, f"Judge · why {rid} lost", s.get("text", ""), "against")
    for s in nar.get("risks_carried") or []:
        for r in s.get("refs") or []:
            add(r, "Judge · risk carried forward", s.get("text", ""), "risk")
    for s in nar.get("borrow") or []:
        for r in s.get("refs") or []:
            add(r, f"Judge · worth borrowing from {s.get('from')}", s.get("text", ""), "idea")
    reviews = []
    for p in sorted((rd / "council").glob("review-*.json")):
        rv = _read(p, {}) or {}
        reviews.append({"persona": rv.get("persona"), "flaws": rv.get("flaws") or {}, "rationale": rv.get("rationale") or {}})
        for rid, fl in (rv.get("flaws") or {}).items():
            for r in fl.get("refs") or []:
                add(r, f"{rv.get('persona')} · worst flaw of {rid}", fl.get("text", ""), "against")
        for rid, text in (rv.get("rationale") or {}).items():
            for r in find_refs(text or ""):
                add(r, f"{rv.get('persona')} · on {rid}", text, "reason")

    # facts the tracer / checks established, attached to the diff hunks of the file they are about
    for rid, led in ledgers.items():
        blast = led.get("blast") or {}
        facts: dict[str, list[tuple[str, str]]] = {}
        for label, items in (("Public API", blast.get("public_api")), ("Schema / migration", blast.get("schema")), ("Config / env", blast.get("config"))):
            for item in items or []:
                m = re.match(r"^([\w./\-]+?):\s*(.*)$", item)
                if m and "/" in m.group(1) or (m and "." in m.group(1)):
                    facts.setdefault(m.group(1), []).append((label, m.group(2) or item))
        for it in led.get("issues") or []:
            m = re.match(r"^([\w./\-]+?)(?::\d+)?[ :]", it.get("detail", ""))
            if m and it.get("kind") in ("test", "lint", "type") and it.get("severity") == "high":
                facts.setdefault(m.group(1), []).append((f"{it['kind'].title()} failure", it["detail"]))
        for ref, item in evidence.items():
            if not ref.startswith(f"{rid}:diff:"):
                continue
            path = (item.get("title") or "").split(" @@")[0]
            for label, text in facts.get(path, []):
                add(ref, f"Tracer · {label}", text, "fact")
        for dep in led.get("deps_added") or []:
            for ref, item in evidence.items():
                if ref.startswith(f"{rid}:diff:") and re.search(r"(requirements|pyproject|package\.json)", item.get("title", "")):
                    add(ref, "Tracer · new dependency", f"adds {dep}", "fact")
    log = read_entries(rd)
    meta = _read(rd / "run.json", {}) or {}
    return {"decision": {"markdown": md, "data": decision}, "ledgers": ledgers, "evidence": evidence, "comments": comments, "reviews": reviews,
            "log": {"entries": log, "summary": summarise(log, meta)},
            "settings": {k: (meta.get("settings") or {}).get(k) for k in ("n", "approach", "models", "judge_model", "reviewer_model")},
            "converged": meta.get("converged") or None}
