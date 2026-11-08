"""Route ledger: the single evidence record the council reasons over.

One ``ledger.json`` per route plus an ``evidence.json`` that backs every ``evidence_ref``.
A ref looks like ``B:diff:3`` (route B, kind ``diff``, id ``3``) and always resolves to a stored
snippet (diff hunk, test output, blast trace, ...), so a claim that cites it can be checked.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .util import iso, now, read_json, short, tail, write_json

EVIDENCE_KINDS = ("diff", "test", "lint", "type", "trace", "plan", "issue", "event", "stat", "dep", "runtime")
REF_RE = re.compile(r"^([A-Z][0-9]?):(" + "|".join(EVIDENCE_KINDS) + r"):([A-Za-z0-9_.\-/#]+)$")
REF_FIND_RE = re.compile(r"\b[A-Z][0-9]?:(?:" + "|".join(EVIDENCE_KINDS) + r"):[A-Za-z0-9_\-/#]+(?:\.[A-Za-z0-9_\-/#]+)*")

ISSUE_KINDS = ("test", "lint", "type", "runtime", "retry", "predicted")
STATUSES = ("queued", "planning", "simulating", "executing", "checking", "done", "failed", "aborted", "skipped")


def parse_ref(ref: str) -> tuple[str, str, str] | None:
    m = REF_RE.match((ref or "").strip().rstrip(".,;:)]`"))
    return (m.group(1), m.group(2), m.group(3)) if m else None


def find_refs(text: str) -> list[str]:
    return REF_FIND_RE.findall(text or "")


@dataclass
class Issue:
    kind: str
    detail: str
    evidence_ref: str = ""
    severity: str = "medium"


@dataclass
class Blast:
    dependents: int = 0
    dependents_list: list[str] = field(default_factory=list)
    public_api: list[str] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)
    config: list[str] = field(default_factory=list)
    score: int = 0
    basis: str = ""
    evidence_ref: str = ""


@dataclass
class Cost:
    tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    usd: float | None = None
    estimated: bool = False


@dataclass
class Ledger:
    route: str
    strategy: dict[str, Any]
    fidelity: str = "simulate"          # simulate | execute | baseline
    status: str = "queued"
    summary: str = ""
    diff_stats: dict[str, Any] = field(default_factory=lambda: {"files": 0, "insertions": 0, "deletions": 0, "predicted": True})
    files: list[dict[str, Any]] = field(default_factory=list)
    blast: Blast = field(default_factory=Blast)
    issues: list[Issue] = field(default_factory=list)
    deps_added: list[str] = field(default_factory=list)
    cost: Cost = field(default_factory=Cost)
    reversibility: int = 3
    reversibility_notes: str = ""
    base_sha: str = ""
    head_sha: str = ""
    branch: str = ""
    worktree: str = ""
    notes: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)

    # ---- (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at_iso"] = iso(self.started_at)
        d["updated_at_iso"] = iso(self.updated_at)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ledger":
        d = dict(d)
        d.pop("started_at_iso", None)
        d.pop("updated_at_iso", None)
        d["blast"] = Blast(**{k: v for k, v in (d.get("blast") or {}).items() if k in Blast.__dataclass_fields__})
        d["cost"] = Cost(**{k: v for k, v in (d.get("cost") or {}).items() if k in Cost.__dataclass_fields__})
        d["issues"] = [Issue(**{k: v for k, v in i.items() if k in Issue.__dataclass_fields__}) for i in d.get("issues") or []]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def is_complete(self) -> bool:
        """Every spec'd field is populated enough for the council to judge (see ``completeness``)."""
        return not self.completeness()

    def completeness(self) -> list[str]:
        """Names of required ledger fields that are still empty/invalid (empty list == complete)."""
        gaps: list[str] = []
        if not self.strategy.get("thesis"):
            gaps.append("strategy")
        if not self.summary:
            gaps.append("summary")
        if self.status not in ("done",):
            gaps.append(f"status={self.status}")
        if self.fidelity != "baseline" and not self.files and self.diff_stats.get("files", 0) == 0:
            gaps.append("files")
        if not self.blast.basis:
            gaps.append("blast")
        if not (1 <= int(self.reversibility) <= 5):
            gaps.append("reversibility")
        if self.cost.seconds <= 0 and self.cost.tokens <= 0 and not self.cost.estimated:
            gaps.append("cost")
        return gaps


class EvidenceStore:
    """Per-route evidence snippets, addressable by ref."""

    MAX_TEXT = 6000

    def __init__(self, route_dir: Path, route_id: str):
        self.path = Path(route_dir) / "evidence.json"
        self.route_id = route_id
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = read_json(self.path, {}) or {}

    def put(self, kind: str, title: str, text: str, key: str | None = None) -> str:
        if kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {kind!r}")
        with self._lock:
            if key is None:
                n = 1 + sum(1 for k in self._items if k.split(":")[1] == kind)
                key = str(n)
            ref = f"{self.route_id}:{kind}:{key}"
            self._items[ref] = {"kind": kind, "title": short(title, 160), "text": tail(text, self.MAX_TEXT) if kind in ("test", "lint", "type", "runtime") else text[: self.MAX_TEXT]}
            write_json(self.path, self._items)
            return ref

    def get(self, ref: str) -> dict[str, Any] | None:
        with self._lock:
            return self._items.get(ref)

    def refs(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def reload(self) -> None:
        with self._lock:
            self._items = read_json(self.path, {}) or {}


class RunEvidence:
    """Resolve any ref across all routes of a run (used by the council's citation checker)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def resolve(self, ref: str) -> dict[str, Any] | None:
        parsed = parse_ref(ref)
        if not parsed:
            return None
        route = parsed[0]
        items = read_json(self.run_dir / "routes" / route / "evidence.json", {}) or {}
        return items.get(ref)

    def exists(self, ref: str) -> bool:
        return self.resolve(ref) is not None


def load_ledger(route_dir: Path) -> Ledger | None:
    d = read_json(Path(route_dir) / "ledger.json")
    return Ledger.from_dict(d) if d else None


def save_ledger(route_dir: Path, ledger: Ledger) -> None:
    ledger.updated_at = now()
    write_json(Path(route_dir) / "ledger.json", ledger.to_dict())


# --------------------------------------------------------------------------- scoring helpers

def reversibility_from_facts(*, schema: bool, public_api: bool, new_deps: int, config: int,
                             infra: bool, files: int) -> tuple[int, str]:
    """Measured reversibility 1 (hard to undo) .. 5 (a clean ``git revert``)."""
    score = 5
    why: list[str] = []
    if schema:
        score -= 2
        why.append("schema/migration change (data may need a down-migration)")
    if public_api:
        score -= 1
        why.append("public API/contract change (clients may depend on it)")
    if infra:
        score -= 1
        why.append("new runtime service/infrastructure to operate")
    elif new_deps:
        score -= 1
        why.append(f"{new_deps} new dependenc{'y' if new_deps == 1 else 'ies'}")
    if config and not infra:
        score -= 0  # config alone is easy to revert; counted via blast
    if files >= 8:
        score -= 1
        why.append(f"{files} files touched")
    score = max(1, min(5, score))
    return score, "; ".join(why) or "self-contained code change; a plain git revert undoes it"
