"""Adopt: land the winner (or tidy up). Nothing here runs unless the user types it.

merge    make the working tree equal route X's tree (a diff from your tree *as it is now* to X is applied),
         committing it as one commit when your tree was clean; a safety snapshot is kept for ``undo``.
replay   write the winning plan as a ready-to-paste prompt (and optionally inject it).
adr      write the Decision Record to docs/decisions/NNNN-<slug>.md.
discard  drop every worktree, branch and ref of the run (ledgers stay for audit).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable

from .checks import detect_checks, run_checks
from .ledger import Ledger, load_ledger
from .route import delete_branch, worktree_remove
from .util import (GIT_IDENTITY, BranchpointError, atomic_write, git, run, short, slugify, snapshot_tree, tree_of,
                   head_sha)


def _ledger(run_dir: Path, rid: str) -> Ledger:
    led = load_ledger(run_dir / "routes" / rid)
    if led is None:
        raise BranchpointError(f"Route {rid} has no ledger in this run.")
    return led


def cleanup_run(repo: Path, run_dir: Path, run_id: str, wt_root: Path, route_ids: list[str], keep: bool = False,
                keep_routes: list[str] | None = None) -> list[str]:
    """Remove worktrees, branches and refs (``keep`` leaves the worktrees for inspection)."""
    removed: list[str] = []
    if keep:
        return removed
    for rid in ["base", *route_ids]:
        if keep_routes and rid in keep_routes:
            continue
        p = wt_root / rid
        if p.exists() or (rid != "base"):
            worktree_remove(repo, p)
            removed.append(rid)
        if rid != "base":
            delete_branch(repo, f"branchpoint/{run_id}/{rid}")
    if not keep_routes:
        shutil.rmtree(wt_root, ignore_errors=True)
    return removed


def adopt_merge(repo: Path, run_dir: Path, meta: dict[str, Any], rid: str) -> str:
    led = _ledger(run_dir, rid)
    if led.status != "done" or not led.head_sha:
        raise BranchpointError(f"Route {rid} did not finish a build (status: {led.status}); there is nothing to merge. "
                               "Simulated routes can be adopted with 'replay' (paste the plan) instead.")
    run_id = meta["run_id"]
    if led.fidelity == "baseline":
        return (f"Route {rid} is what is already in your working tree: nothing to change.\n"
                "Use 'discard' to drop the other routes, or 'adr' to record the decision.")
    before = snapshot_tree(repo, parent=head_sha(repo), message="branchpoint: pre-adopt snapshot")
    git(["update-ref", f"refs/branchpoint/{run_id}/pre-adopt", before], repo)
    was_clean = not git(["status", "--porcelain"], repo).strip()
    target = led.head_sha
    if tree_of(repo, before) == tree_of(repo, target):
        return f"Your working tree already matches route {rid}."
    patch = git(["diff", "--binary", "--full-index", before, target], repo, timeout=120)
    res = run(["git", "apply", "--binary", "--whitespace=nowarn", "-"], repo, timeout=120, stdin=patch)
    if not res.ok:
        raise BranchpointError("Could not apply the route to your working tree: " + short(res.err or res.out, 300)
                               + "\nNothing was changed. Commit or stash your work and try again.")
    after = snapshot_tree(repo, parent=head_sha(repo), message="branchpoint: post-adopt check")
    exact = tree_of(repo, after) == tree_of(repo, target)
    lines = [f"Route {rid} ({led.strategy.get('name')}) applied: {len(led.files)} file(s), "
             f"+{led.diff_stats.get('insertions')} -{led.diff_stats.get('deletions')}."]
    if not exact:
        lines.append("Warning: the working tree differs from the route's tree in ignored/untracked paths; review with `git status`.")
    meta["adopted"] = rid
    meta["pre_head"] = head_sha(repo) or ""
    if was_clean:
        git(["add", "-A"], repo)
        msg = f"Adopt Branchpoint route {rid}: {led.strategy.get('name')}\n\n{led.strategy.get('thesis')}\n\nBranchpoint run {run_id}; decision record in docs/decisions or the run directory."
        git(["commit", "-q", "--no-verify", "--no-gpg-sign", "-m", msg], repo, env=GIT_IDENTITY)
        meta["adopt_commit"] = head_sha(repo) or ""
        lines.append("Committed as one commit on your current branch (your tree was clean).")
    else:
        lines.append("Your tree had uncommitted work, so the result is left uncommitted for you to review and commit.")
    lines.append(f"Undo: /branchpoint undo {run_id}")
    specs = [s for s in detect_checks(repo, meta.get("settings") or {}) if s.kind == "test"]
    if specs:
        r = run_checks(repo, specs)[0]
        lines.append(f"Tests after adopt: {'PASS' if r.passed else 'FAIL'} — {r.summary} ({r.seconds:.1f}s)")
    return "\n".join(lines)


def undo_adopt(repo: Path, run_dir: Path, meta: dict[str, Any]) -> str:
    run_id = meta["run_id"]
    res = run(["git", "rev-parse", "--verify", "-q", f"refs/branchpoint/{run_id}/pre-adopt^{{commit}}"], repo, timeout=15)
    if not res.ok:
        raise BranchpointError("No pre-adopt snapshot for this run: nothing to undo.")
    before = res.out.strip()
    winner = meta.get("adopted")
    if not winner:
        raise BranchpointError("This run has not adopted a route.")
    led = _ledger(run_dir, winner)
    now_snap = snapshot_tree(repo, parent=head_sha(repo), message="branchpoint: pre-undo snapshot")
    patch = git(["diff", "--binary", "--full-index", now_snap, before], repo, timeout=120)
    r = run(["git", "apply", "--binary", "--whitespace=nowarn", "-"], repo, timeout=120, stdin=patch)
    if not r.ok:
        raise BranchpointError("Could not undo cleanly: " + short(r.err or r.out, 240))
    note = ""
    adopt_commit, pre_head = meta.get("adopt_commit"), meta.get("pre_head")
    if adopt_commit and pre_head and head_sha(repo) == adopt_commit:
        git(["reset", "--mixed", "-q", pre_head], repo)       # the adopt commit was the tip: take it back out of history
        note = " The adopt commit was removed from your branch."
    elif adopt_commit:
        note = f"\nThe adopt commit {adopt_commit[:10]} is still in your history (you have committed since): revert it with `git revert {adopt_commit[:10]}` if you want it gone."
    meta["adopted"] = None
    return f"Working tree restored to its state before adopting route {winner} ({led.strategy.get('name')}).{note}"


def replay_prompt(run_dir: Path, meta: dict[str, Any], rid: str) -> tuple[str, Path]:
    led = _ledger(run_dir, rid)
    plan = ""
    pj = run_dir / "routes" / rid / "evidence.json"
    if pj.exists():
        import json
        items = json.loads(pj.read_text(encoding="utf-8"))
        plan = (items.get(f"{rid}:plan:md") or {}).get("text", "")
    s = led.strategy
    text = [f"Implement this goal using the approach below (chosen by a Branchpoint council, run {meta['run_id']}).", "",
            f"GOAL: {meta['goal']}", "",
            f"APPROACH — {s.get('name')} (axis: {s.get('axis')}): {s.get('thesis')}", "",
            "Key decisions:"] + [f"- {d}" for d in s.get("key_decisions") or []]
    if plan:
        text += ["", "Plan:", plan]
    elif led.files:
        text += ["", "Files touched by the winning build: " + ", ".join(f["path"] for f in led.files[:20])]
    risks = [i.detail for i in led.issues if i.severity != "low"][:5]
    if risks:
        text += ["", "Known risks to handle:"] + [f"- {r}" for r in risks]
    out = "\n".join(text).rstrip() + "\n"
    path = run_dir / "replay_prompt.md"
    atomic_write(path, out)
    return out, path


def adopt_replay(run_dir: Path, meta: dict[str, Any], rid: str, inject: Callable[[str], Any] | None = None) -> str:
    text, path = replay_prompt(run_dir, meta, rid)
    n = meta.get("prompt_n")
    lines = []
    if meta.get("mode") == "rewind" and n:
        lines.append(f"1. In the Hermes TUI/CLI run:  /rewind fork {n}   (forks a clean session from just before prompt {n}; your original session stays intact)")
        lines.append("2. Paste the prompt below, or send it from the file:")
    else:
        lines.append("Paste the prompt below into the session (or send it from the file):")
    lines.append(f"   {path}")
    if inject is not None:
        try:
            inject(text)
            lines.append("The prompt was also queued into the current session (Hermes accepted the injection).")
        except Exception as exc:
            lines.append(f"(could not inject automatically: {short(exc, 100)})")
    return "\n".join(lines) + "\n\n----- prompt -----\n" + text


def next_adr_number(dir_: Path) -> int:
    nums = [int(m.group(1)) for p in dir_.glob("*.md") if (m := re.match(r"(\d{4})-", p.name))] if dir_.exists() else []
    return (max(nums) + 1) if nums else 1


def adopt_adr(repo: Path, run_dir: Path, meta: dict[str, Any], rid: str | None = None) -> str:
    src = run_dir / "decision.md"
    if not src.exists():
        raise BranchpointError("No Decision Record yet: run /branchpoint council first.")
    dest_dir = repo / "docs" / "decisions"
    n = next_adr_number(dest_dir)
    body = src.read_text(encoding="utf-8")
    title = short(meta.get("goal") or "decision", 80)
    adopted = f"Route {rid}" if rid else "see below"
    header = (f"# {n:04d}. {title}\n\n- Status: Accepted ({adopted})\n- Date: {meta.get('created_iso', '')[:10]}\n"
              f"- Generated by Branchpoint run `{meta['run_id']}`\n\n")
    body = re.sub(r"^# .*\n", "", body, count=1)
    dest = dest_dir / f"{n:04d}-{slugify(title)}.md"
    atomic_write(dest, header + body.lstrip())
    return f"Decision Record written to {dest.relative_to(repo)} (not committed)."
