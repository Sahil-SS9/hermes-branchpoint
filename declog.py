"""The decision log: what was decided, at each decision point, and why.

One append-only file per run (``decision_log.jsonl``). Entries are written by the code that makes the decision,
from facts it already holds (a similarity score, a ranking, a failing check, a budget stop, the agent's own stated
plan), so the "why" is never invented after the fact. ``summarise`` turns the entries into a short plain-English
account; ``render_markdown`` writes the full log as ``decision-log.md``.

Kinds: anchor, agents, plan, duplicate, select, build, council, converge, adopt, note.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import iso, now, short

KINDS = ("anchor", "agents", "plan", "duplicate", "select", "build", "council", "converge", "adopt", "note")
PHASES = {"anchor": "Where we started", "agents": "Who works on it", "plan": "How each agent plans it", "duplicate": "Plans that matched",
          "select": "What gets built", "build": "What got built", "council": "The verdict", "converge": "Converging the best ideas",
          "adopt": "Adopting a result", "note": "Other notes"}


class DecisionLog:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "decision_log.jsonl"
        self.md_path = Path(run_dir) / "decision-log.md"
        self._lock = threading.Lock()

    def record(self, kind: str, title: str, what: str, why: str, *, route: str = "", thinking: str = "",
               refs: list[str] | None = None, data: dict[str, Any] | None = None) -> dict[str, Any]:
        kind = kind if kind in KINDS else "note"
        with self._lock:
            n = len(self.entries()) + 1
            entry = {"n": n, "ts": now(), "iso": iso(), "kind": kind, "route": route, "title": short(title, 140),
                     "what": short(what, 600), "why": short(why, 900), "thinking": short(thinking, 1200),
                     "refs": [r for r in (refs or []) if r], "data": data or {}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def write_markdown(self, meta: dict[str, Any] | None = None) -> Path:
        self.md_path.write_text(render_markdown(self.entries(), meta or {}), encoding="utf-8")
        return self.md_path


def read_entries(run_dir: str | Path) -> list[dict[str, Any]]:
    return DecisionLog(run_dir).entries()


def _of(entries: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in entries if e.get("kind") == kind]


def summarise(entries: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> str:
    """A short, factual account built only from the logged entries."""
    meta = meta or {}
    parts: list[str] = []
    a = _of(entries, "anchor")
    if a:
        parts.append(a[0]["what"].rstrip(".") + ".")
    g = _of(entries, "agents")
    if g:
        parts.append(g[-1]["what"].rstrip(".") + ".")
    d = _of(entries, "duplicate")
    if d:
        parts.append(f"{len(d)} plan{'s' if len(d) != 1 else ''} matched an earlier one closely, so each was built only once.")
    s = _of(entries, "select")
    if s:
        parts.append(s[-1]["what"].rstrip(".") + ".")
    b = _of(entries, "build")
    if b:
        bad = [e for e in b if e.get("data", {}).get("failing")]
        parts.append(f"{len(b)} route{'s were' if len(b) != 1 else ' was'} built" + (f"; {len(bad)} ended with failing checks." if bad else ", all with checks passing."))
    c = [e for e in _of(entries, "council") if e.get("title") == "The council's pick"]
    if c:
        parts.append(c[0]["what"].rstrip(".") + ". " + c[0]["why"].split(". Three reviewers")[0].rstrip(".") + ".")
    v = [e for e in _of(entries, "converge") if e.get("title") == "Result of converging"]
    if v:
        parts.append(v[-1]["what"].rstrip(".") + ".")
    ad = _of(entries, "adopt")
    if ad:
        parts.append(ad[-1]["what"].rstrip(".") + ".")
    return " ".join(parts)


def render_markdown(entries: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# Decision log", ""]
    if meta.get("goal"):
        lines += [f"**Task:** {meta['goal']}", ""]
    lines += ["## In short", "", summarise(entries, meta) or "No decisions were recorded.", ""]
    seen: set[str] = set()
    for kind in PHASES:
        group = _of(entries, kind)
        if not group:
            continue
        seen.add(kind)
        lines += [f"## {PHASES[kind]}", ""]
        for e in group:
            tag = f"Route {e['route']}: " if e.get("route") else ""
            lines += [f"### {tag}{e['title']}", "", f"- **Decided:** {e['what']}", f"- **Because:** {e['why']}"]
            if e.get("thinking"):
                lines.append(f"- **The agent's own reasoning:** {e['thinking']}")
            if e.get("refs"):
                lines.append("- **Evidence:** " + ", ".join(f"`{r}`" for r in e["refs"]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
