---
name: branchpoint
description: Use when the user is at an architecture fork or just finished a build and wants to know if another route would have been better.
---

# Branchpoint

Branchpoint takes `/rewind` from undo-and-retry to undo-and-explore. It builds 3-5 genuinely different
routes in isolated git worktrees, logs every issue and blast-radius impact in a ledger, and a blind council
(Maintainer, Skeptic, User Advocate + judge) writes a Decision Record whose every claim cites ledger evidence.

## When to suggest it

- The user asks "Redis or in-process cache?", "trigger or app-level job?", "sync or queue?" before building.
- The user just finished a build and asks whether a different route would have been better.
- Do NOT suggest it for small, obvious edits. A run costs tokens (default cap 600k).

## What you can do yourself

- `branchpoint_fork(goal, n)`: simulate-only exploration (plans + static blast radius, no builds, 200k token cap).
- `branchpoint_status` / `branchpoint_decision`: read results.

Real builds, `rewind` and `adopt` are the user's decisions: tell them to type the slash command:

```
/branchpoint fork "add caching to the slow /report endpoint" --n 4
/branchpoint rewind 3
/branchpoint status
/branchpoint adopt C merge      # or replay | adr | discard
```

## Safety facts you can rely on

- Routes run in worktrees outside the repository; the main working tree changes only on `adopt`.
- Build agents get file tools only (no shell, no network). Branchpoint itself runs the project's tests/lint/type checks.
- `adopt merge` keeps a safety snapshot; `/branchpoint undo` reverses it.
