"""The small contract between the parent session and a route's headless build agent.

The parent writes ``branchpoint-route.json`` into the worktree's private git dir (so it never shows up
in the diff). The same plugin, loaded inside the child ``hermes`` process, finds it from the child's
working directory and then (a) confines file writes to the worktree, (b) caps tool calls and tokens,
(c) logs every tool event for the ledger and (d) feeds the project's check results back to the agent.

The tool guard is **fail-closed**: a call is allowed only when this module can fully account for it —
the tool is one of the four file tools, every argument key is one the tool's schema names, every path
it names stays inside the worktree, and a patch body parses into headers this module recognises.
Anything else is refused. A guard that allows what it cannot read is not a guard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .util import within

MARKER = "branchpoint-route.json"
ALLOWED_TOOLS = frozenset({"read_file", "write_file", "patch", "search_files"})
WRITE_TOOLS = frozenset({"write_file", "patch"})

# The argument keys each tool's schema names, taken from the host's own tool schemas. A key outside its
# tool's set is refused: an unparsed argument is an unchecked argument, which is how ``{"file": "/etc/x"}``
# and OpenCode's ``filePath`` used to pass this guard. ``guard_extra_args`` in settings extends a tool's set
# for a host that adds a key, which is a deliberate, recorded choice rather than a silent allow.
TOOL_ARG_KEYS: dict[str, frozenset[str]] = {
    "read_file": frozenset({"path", "offset", "limit"}),
    "write_file": frozenset({"path", "content", "cross_profile"}),
    "patch": frozenset({"path", "old_string", "new_string", "replace_all", "mode", "patch", "cross_profile"}),
    "search_files": frozenset({"pattern", "target", "path", "file_glob", "limit", "offset", "order",
                               "output_mode", "context"}),
}

# Keys whose value is a filesystem path, and keys whose value is a list of them.
PATH_KEYS = ("path", "file_path", "filepath", "filename", "target", "destination", "dest", "file",
             "filePath", "filepath_", "source", "src")
PATH_LIST_KEYS = ("paths", "files", "edits")

# Paths a route may not write at all. These are the files whose *content* is executed or that redirect
# git and harness behaviour, so a route that edits them is not a code change — it is a code-execution
# change, and it must be reviewed by a human, not by the council. Configurable via ``protected_paths``.
PROTECTED_PREFIXES = (
    ".git", ".gitattributes", ".gitmodules", ".gitconfig",
    ".claude", ".codex", ".opencode", ".pi", ".hermes", ".commandcode", ".cursor", ".windsurf",
    ".github/workflows", ".gitlab-ci.yml", ".envrc", ".vscode",
)

_V4A_RE = re.compile(r"^\*\*\*\s+(?:Update|Add|Delete|Move to)(?:\s+File)?:\s*(.+?)\s*$", re.M)
_DIFF_GIT_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)
_UNIFIED_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:[ab]/)?(\S+?)(?:\t.*)?$", re.M)
_RENAME_RE = re.compile(r"^(?:rename (?:from|to)|copy (?:from|to))\s+(\S+)$", re.M)


class PatchUnparsed(Exception):
    """A patch body whose file headers this module cannot read, so its targets cannot be confined."""


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


def patch_targets(patch: str) -> list[str]:
    """Every file a patch body names, from V4A and unified-diff headers alike.

    Raises ``PatchUnparsed`` when the body carries no header this module can read: a patch whose
    targets cannot be extracted cannot be confined, so it is refused rather than applied blind.
    """
    found: list[str] = []
    for regex in (_V4A_RE, _DIFF_GIT_RE, _UNIFIED_RE, _RENAME_RE):
        for match in regex.finditer(patch):
            for group in match.groups():
                if group:
                    found.append(group.strip())
    out = [p for p in dict.fromkeys(found) if p and p != "/dev/null"]
    if not out:
        raise PatchUnparsed("the patch body has no file header this guard can read (expected a "
                            "`*** Update File:` or `---`/`+++`/`diff --git` header)")
    return out


def extract_paths(tool_name: str, args: dict[str, Any] | None) -> list[str]:
    """Every filesystem path a file-tool call would touch. Raises ``PatchUnparsed`` on an unreadable patch."""
    args = args or {}
    out: list[str] = []
    for key in PATH_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    for key in PATH_LIST_KEYS:
        value = args.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
                elif isinstance(item, dict):
                    for nested in PATH_KEYS:
                        inner = item.get(nested)
                        if isinstance(inner, str) and inner.strip():
                            out.append(inner.strip())
    patch = args.get("patch")
    if isinstance(patch, str) and patch.strip():
        out.extend(patch_targets(patch))
    return out


def unknown_keys(tool_name: str, args: dict[str, Any] | None, extra: frozenset[str] | set[str] | None = None) -> list[str]:
    """Argument keys this tool's schema does not name (they would go unchecked)."""
    allowed = TOOL_ARG_KEYS.get(tool_name, frozenset()) | frozenset(extra or ())
    return [k for k in (args or {}) if k not in allowed]


def normalise_rel(path: str) -> str:
    """A repo-relative path with every ``./`` removed and no leading slash.

    ``str.lstrip("./")`` strips *characters*, not a prefix: it turned ``.github/ci.yml`` into
    ``github/ci.yml``, losing the dot and every plan's blast radius with it.
    """
    text = str(path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    return str(PurePosixPath(text)) if text else ""


def is_protected(path: str, extra: tuple[str, ...] | list[str] | None = None) -> str | None:
    """The protected entry this path matches, else None.

    Matched on whole path components so ``.github/workflows`` blocks ``.github/workflows/ci.yml`` but not
    ``docs/.githubnotes.md``.
    """
    rel = normalise_rel(path)
    if not rel:
        return None
    parts = PurePosixPath(rel).parts
    for entry in (*PROTECTED_PREFIXES, *(extra or ())):
        candidate = normalise_rel(entry)
        if not candidate:
            continue
        cparts = PurePosixPath(candidate).parts
        if parts[: len(cparts)] == cparts:
            return candidate
    return None


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
