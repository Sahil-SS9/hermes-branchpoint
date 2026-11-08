"""Anchor: map a rewind point ("prompt N") to a git commit, plus the transcript up to that point.

Resolution ladder (first hit wins, each recorded in ``Anchor.source`` so the user can see how sure we are):

1. ``--base <rev>``               explicit override.
2. turn snapshot                  Branchpoint's own snapshot taken when prompt N was submitted
                                  (a dangling commit made through a throw-away index; the user's
                                  index, HEAD and files are never touched).
3. Hermes checkpoint              the shadow-store checkpoint taken at or after prompt N's timestamp,
                                  overlaid on HEAD (read-only; skipped when the layout is not recognised).
4. git history                    the last commit made before prompt N's timestamp.
5. refuse                         ask for ``--base``.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .util import (GIT_IDENTITY, BranchpointError, git, git_ok, head_sha, iso, now, repo_root, run,
                   short, snapshot_tree, tree_of)

ANCHOR_REF_PREFIX = "refs/branchpoint/anchors"
MAX_KEPT_ANCHORS = 300
MAX_INDEX_BYTES = 30_000_000   # ~200k tracked files


@dataclass
class Anchor:
    repo: str
    base_sha: str
    head_sha: str                   # snapshot of the *current* working tree (= route A's end state)
    source: str
    prompt_n: int | None = None
    session_id: str = ""
    prompt_ts: float | None = None
    goal: str = ""
    followups: list[str] = field(default_factory=list)
    context: str = ""               # transcript before prompt N, trimmed
    has_baseline: bool = True       # False when base and head trees are identical
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_key(repo: str | Path) -> str:
    return hashlib.sha256(str(Path(repo).resolve()).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- recording (hook side)

def _turn_log(data_dir: Path, repo: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:80] or "session"
    return Path(data_dir) / "anchors" / repo_key(repo) / f"{safe}.jsonl"


def load_turns(data_dir: Path, repo: Path, session_id: str) -> list[dict[str, Any]]:
    path = _turn_log(data_dir, repo, session_id)
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(__import__("json").loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def record_turn(data_dir: Path, repo: str | Path, session_id: str, n: int, prompt: str,
                timeout: float = 25.0) -> dict[str, Any] | None:
    """Snapshot the working tree as prompt ``n`` is submitted. Never raises."""
    try:
        repo = Path(repo)
        if not (repo / ".git").exists() and not git_ok(["rev-parse", "--git-dir"], repo, timeout=10):
            return None
        gitdir = Path(git(["rev-parse", "--absolute-git-dir"], repo, timeout=10).strip())
        try:
            if (gitdir / "index").stat().st_size > MAX_INDEX_BYTES:
                return None       # a very large repo: per-prompt snapshots would cost more than they are worth
        except OSError:
            pass
        existing = load_turns(data_dir, repo, session_id)
        if any(t.get("n") == n and t.get("prompt_hash") == hashlib.sha1(prompt.encode()).hexdigest()[:12] for t in existing):
            return None
        sha = snapshot_tree(repo, parent=head_sha(repo), message=f"branchpoint anchor: {session_id[:12]} prompt {n}", timeout=timeout)
        ref = f"{ANCHOR_REF_PREFIX}/{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id)[:40]}/{n}"
        git(["update-ref", ref, sha], repo, timeout=15)
        entry = {"n": n, "ts": now(), "sha": sha, "ref": ref, "prompt": prompt[:2000],
                 "prompt_hash": hashlib.sha1(prompt.encode()).hexdigest()[:12], "head": head_sha(repo)}
        path = _turn_log(data_dir, repo, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(__import__("json").dumps(entry) + "\n")
        # keep the ref namespace bounded
        if len(existing) + 1 > MAX_KEPT_ANCHORS:
            old = existing[0]
            git(["update-ref", "-d", old.get("ref", "")], repo, check=False)
        return entry
    except Exception:  # hooks must never break a turn
        return None


# --------------------------------------------------------------------------- transcript

@dataclass
class Prompt:
    n: int
    text: str
    ts: float | None


def _row_ts(row: dict[str, Any]) -> float | None:
    for k in ("timestamp", "created_at", "ts", "time"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                try:
                    return time.mktime(time.strptime(v[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
                except ValueError:
                    continue
    return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def load_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Read a session transcript through the public ``hermes_state`` module. [] when unavailable."""
    if not session_id:
        return []
    try:
        from hermes_state import SessionDB  # public plugin API surface
        db = SessionDB()
        try:
            rows = db.get_messages(session_id)
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()
        return [dict(r) for r in rows or []]
    except Exception:
        return []


def user_prompts(messages: list[dict[str, Any]]) -> list[Prompt]:
    prompts: list[Prompt] = []
    for row in messages:
        if row.get("role") != "user":
            continue
        text = _content_text(row.get("content")).strip()
        if not text or text.startswith("[CONTEXT COMPACTION") or text.startswith("[Branchpoint"):
            continue
        prompts.append(Prompt(len(prompts) + 1, text, _row_ts(row)))
    return prompts


def build_context(messages: list[dict[str, Any]], before_n: int, redact=None, max_chars: int = 6000) -> str:
    """Condensed transcript before prompt ``before_n``: user asks and short assistant replies only."""
    lines: list[str] = []
    seen_users = 0
    for row in messages:
        role = row.get("role")
        text = _content_text(row.get("content")).strip()
        if role == "user" and text:
            seen_users += 1
            if seen_users >= before_n:
                break
            lines.append(f"USER: {short(text, 600)}")
        elif role == "assistant" and text and seen_users:
            lines.append(f"ASSISTANT: {short(text, 300)}")
    ctx = "\n".join(lines)[-max_chars:]
    if redact:
        try:
            ctx = redact(ctx)
        except Exception:
            ctx = "[transcript withheld: redaction unavailable]"
    return ctx


# --------------------------------------------------------------------------- Hermes checkpoint fallback

def _checkpoint_store(hermes_home: Path) -> Path | None:
    store = Path(hermes_home) / "checkpoints" / "store"
    return store if (store / "HEAD").exists() else None


def find_hermes_checkpoint(hermes_home: Path, repo: Path, since_ts: float) -> tuple[Path, str, str, float] | None:
    """(store, ref, commit, taken_at) of the earliest checkpoint taken at/after ``since_ts``."""
    store = _checkpoint_store(hermes_home)
    if not store:
        return None
    dirhash = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:16]
    ref = f"refs/hermes/{dirhash}"
    res = run(["git", "--git-dir", str(store), "log", ref, "--format=%H%x1f%at%x1f%B%x1e", "-n", "60"], repo, timeout=30)
    if not res.ok:
        return None
    best: tuple[str, float] | None = None
    for rec in res.out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f")
        if len(parts) < 3:
            continue
        sha, at, body = parts[0].strip(), parts[1], parts[2]
        taken = float(at) if at.isdigit() else 0.0
        m = re.search(r"Hermes-Taken-At:\s*([0-9.]+)", body)
        if m:
            taken = float(m.group(1))
        if taken >= since_ts - 1 and (best is None or taken < best[1]):
            best = (sha, taken)
    return (store, ref, best[0], best[1]) if best else None


def overlay_checkpoint(repo: Path, store: Path, ref: str, commit: str) -> str:
    """Create a commit in ``repo`` = HEAD tree with every file of the checkpoint laid over it."""
    dirhash = ref.rsplit("/", 1)[-1]
    local_ref = f"refs/branchpoint/imported/{dirhash}"
    git(["fetch", "--no-tags", "--quiet", str(store), f"+{ref}:{local_ref}"], repo, timeout=120)
    tmp = tempfile.mkdtemp(prefix="bp-ckpt-")
    idx = os.path.join(tmp, "index")
    try:
        env = {"GIT_INDEX_FILE": idx, **GIT_IDENTITY}
        parent = head_sha(repo)
        if parent:
            git(["read-tree", parent], repo, env=env)
        listing = git(["ls-tree", "-r", "-z", commit], repo)
        info: list[str] = []
        for ent in listing.split("\0"):
            if not ent:
                continue
            meta, path = ent.split("\t", 1)
            mode, typ, sha = meta.split()
            if typ == "blob":
                info.append(f"{mode} {sha}\t{path}")
        if info:
            git(["update-index", "--add", "-z", "--index-info"], repo, env=env, stdin="\0".join(info) + "\0")
        tree = git(["write-tree"], repo, env=env).strip()
        args = ["commit-tree", tree, "-m", f"branchpoint: checkpoint {commit[:10]} overlaid on HEAD"]
        if parent:
            args[2:2] = ["-p", parent]
        return git(args, repo, env=env).strip()
    finally:
        try:
            for f in Path(tmp).iterdir():
                f.unlink()
            os.rmdir(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------- resolution

def resolve_rewind(*, cwd: str | Path, data_dir: Path, hermes_home: Path | None, session_id: str, n: int,
                   base_override: str | None = None, goal_override: str | None = None,
                   messages: list[dict[str, Any]] | None = None, redact=None) -> Anchor:
    repo = repo_root(cwd)
    messages = messages if messages is not None else load_session_messages(session_id)
    prompts = user_prompts(messages)
    turns = load_turns(data_dir, repo, session_id)
    notes: list[str] = []

    prompt: Prompt | None = prompts[n - 1] if 0 < n <= len(prompts) else None
    if prompt is None and turns:
        t = next((t for t in turns if t.get("n") == n), None)
        if t:
            prompt = Prompt(n, t.get("prompt", ""), t.get("ts"))
            notes.append("transcript unavailable; using Branchpoint's own turn log")
    if prompt is None and not base_override:
        have = len(prompts) or len(turns)
        raise BranchpointError(
            f"Prompt {n} not found in this session ({have} prompt(s) known). "
            "Pass --base <commit> to name the starting point explicitly.")

    head_snap = snapshot_tree(repo, parent=head_sha(repo), message="branchpoint: current working state (route A)")
    base: str | None = None
    source = ""
    if base_override:
        res = run(["git", "rev-parse", "--verify", "-q", f"{base_override}^{{commit}}"], repo, timeout=15)
        if not res.ok:
            raise BranchpointError(f"--base {base_override!r} is not a commit in this repository.")
        base, source = res.out.strip(), "override"
    if base is None:
        ph = hashlib.sha1(prompt.text.encode()).hexdigest()[:12] if prompt else ""
        t = next((t for t in reversed(turns) if ph and t.get("prompt_hash") == ph), None) or \
            next((t for t in reversed(turns) if t.get("n") == n), None)
        if t and git_ok(["cat-file", "-e", f"{t['sha']}^{{commit}}"], repo, timeout=15):
            base, source = t["sha"], "turn-snapshot"
    if base is None and prompt and prompt.ts and hermes_home:
        found = find_hermes_checkpoint(hermes_home, repo, prompt.ts)
        if found:
            store, ref, commit, taken = found
            try:
                base = overlay_checkpoint(repo, store, ref, commit)
                source = "hermes-checkpoint"
                notes.append(f"Hermes checkpoint {commit[:10]} taken {iso(taken)} overlaid on HEAD")
            except BranchpointError as exc:
                notes.append(f"checkpoint import failed: {short(exc, 160)}")
    if base is None and prompt and prompt.ts:
        res = run(["git", "rev-list", "-1", f"--before={int(prompt.ts)}", "HEAD"], repo, timeout=30)
        if res.ok and res.out.strip():
            base, source = res.out.strip(), "git-history"
            notes.append("base is the last commit before the prompt: uncommitted work from before the prompt is not in it")
    if base is None:
        raise BranchpointError(
            f"Could not find a code state for prompt {n} (no turn snapshot, Hermes checkpoint or commit history). "
            "Pass --base <commit> to choose the starting point.")

    goal = goal_override or (prompt.text if prompt else "")
    followups = [p.text for p in prompts[n:]] if prompt else []
    has_baseline = tree_of(repo, base) != tree_of(repo, head_snap)
    if not has_baseline:
        notes.append("working tree equals the base: nothing was built since that point, so there is no baseline route")
    return Anchor(
        repo=str(repo), base_sha=base, head_sha=head_snap, source=source, prompt_n=n, session_id=session_id,
        prompt_ts=prompt.ts if prompt else None, goal=goal, followups=followups,
        context=build_context(messages, n, redact) if messages else "", has_baseline=has_baseline, notes=notes,
    )


def resolve_fork(*, cwd: str | Path, goal: str, session_id: str = "", messages: list[dict[str, Any]] | None = None,
                 redact=None) -> Anchor:
    """Prospective mode: every route starts from the working tree as it is now."""
    repo = repo_root(cwd)
    snap = snapshot_tree(repo, parent=head_sha(repo), message="branchpoint: fork point")
    messages = messages if messages is not None else load_session_messages(session_id)
    n_users = len(user_prompts(messages))
    return Anchor(
        repo=str(repo), base_sha=snap, head_sha=snap, source="working-tree", prompt_n=None, session_id=session_id,
        goal=goal, context=build_context(messages, n_users + 1, redact) if messages else "", has_baseline=False,
    )


def baseline_stats(messages: list[dict[str, Any]], n: int) -> dict[str, Any]:
    """What building route A cost, read from the transcript after prompt ``n`` (tokens are an estimate)."""
    seen = 0
    tool_calls = 0
    chars = 0
    first_ts: float | None = None
    last_ts: float | None = None
    for row in messages:
        if row.get("role") == "user" and _content_text(row.get("content")).strip():
            seen += 1
        if seen < n:
            continue
        ts = _row_ts(row)
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        text = _content_text(row.get("content"))
        chars += len(text)
        if row.get("role") == "tool":
            tool_calls += 1
        calls = row.get("tool_calls")
        if isinstance(calls, list) and row.get("role") == "assistant":
            chars += len(str(calls))
    return {"tool_calls": tool_calls, "tokens": chars // 4, "seconds": (last_ts - first_ts) if (first_ts and last_ts) else 0.0}
