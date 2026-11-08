"""The small contract between the parent session and a route's headless build agent.

The parent writes ``branchpoint-route.json`` into the worktree's private git dir (so it never shows up
in the diff). The same plugin, loaded inside the child ``hermes`` process, finds it from the child's
working directory and then (a) confines file writes to the worktree, (b) caps tool calls and tokens,
(c) logs every tool event for the ledger and (d) feeds the project's check results back to the agent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .util import within

MARKER = "branchpoint-route.json"
ALLOWED_TOOLS = frozenset({"read_file", "write_file", "patch", "search_files"})
WRITE_TOOLS = frozenset({"write_file", "patch"})
_PATH_KEYS = ("path", "file_path", "filepath", "filename", "target", "destination", "dest")
_V4A_RE = re.compile(r"^\*\*\*\s+(?:Update|Add|Delete|Move to)(?:\s+File)?:\s*(.+?)\s*$", re.M)


def gitdir_of(worktree: str | Path) -> Path | None:
    """The private git dir of a linked worktree (``<repo>/.git/worktrees/<name>``), else None."""
    dotgit = Path(worktree) / ".git"
    try:
        if dotgit.is_file():
            text = dotgit.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gd = Path(text.split(":", 1)[1].strip())
                if not gd.is_absolute():
                    gd = (Path(worktree) / gd).resolve()
                return gd if "worktrees" in gd.parts else None
    except OSError:
        pass
    return None


def marker_path(worktree: str | Path) -> Path | None:
    gd = gitdir_of(worktree)
    return gd / MARKER if gd else None


def write_marker(worktree: str | Path, data: dict[str, Any]) -> Path:
    mp = marker_path(worktree)
    if mp is None:
        raise ValueError(f"{worktree} is not a linked git worktree")
    mp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return mp


def read_marker(cwd: str | Path) -> dict[str, Any] | None:
    """Marker for the worktree that contains ``cwd`` (walks up to the worktree root)."""
    p = Path(cwd).resolve()
    for cand in [p, *p.parents][:12]:
        mp = marker_path(cand)
        if mp is not None:
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            data.setdefault("worktree", str(cand))
            return data
    return None


def extract_paths(tool_name: str, args: dict[str, Any] | None) -> list[str]:
    """Every filesystem path a file-tool call would touch (including V4A multi-file patches)."""
    args = args or {}
    out: list[str] = []
    for k in _PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    patch = args.get("patch")
    if isinstance(patch, str):
        out.extend(m.group(1) for m in _V4A_RE.finditer(patch))
    return out


def outside_root(paths: list[str], root: str | Path, cwd: str | Path) -> str | None:
    """First path that resolves outside ``root`` (relative paths resolve against ``cwd``), else None."""
    for raw in paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(cwd) / p
        if not within(p, root):
            return raw
    return None


def tokens_from_usage(usage: Any) -> int:
    """Total tokens from a provider usage payload of unknown shape."""
    if isinstance(usage, dict):
        for k in ("total_tokens", "total"):
            if isinstance(usage.get(k), (int, float)) and usage[k]:
                return int(usage[k])
        tin = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        tout = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        return int(tin) + int(tout)
    for attr in ("total_tokens",):
        v = getattr(usage, attr, None)
        if isinstance(v, (int, float)) and v:
            return int(v)
    tin = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
    tout = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0
    return int(tin) + int(tout)
