"""Append-only event stream for a run (JSONL on disk + in-memory for the live view)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import now


class EventBus:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        self.listeners: list = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._events.append(json.loads(line))
                except ValueError:
                    continue

    def emit(self, type: str, **data: Any) -> dict[str, Any]:
        with self._cond:
            ev = {"seq": len(self._events) + 1, "ts": round(now(), 3), "type": type, **data}
            self._events.append(ev)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(ev, default=str) + "\n")
            self._cond.notify_all()
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:
                pass
        return ev

    def since(self, seq: int = 0) -> list[dict[str, Any]]:
        with self._cond:
            return [e for e in self._events if e["seq"] > seq]

    def wait(self, seq: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        with self._cond:
            if len(self._events) <= seq:
                self._cond.wait(timeout)
            return [e for e in self._events if e["seq"] > seq]

    def all(self) -> list[dict[str, Any]]:
        return self.since(0)
