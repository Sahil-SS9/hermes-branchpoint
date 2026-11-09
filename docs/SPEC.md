# Branchpoint: design notes

How the PRD became code, what the open questions resolved to (checked against Hermes upstream `main`, 8 Nov 2026),
and every place the implementation deliberately differs from the written spec.

## Open questions, resolved

| Question | Answer | Consequence |
|---|---|---|
| Can a plugin fork a session from code? | **No.** `/rewind fork` is internal to the CLI (`_backtrack_fork` = `/branch` + rewind); no `ctx` API exposes it. | Branchpoint does its own branching (worktrees + headless runs). `adopt replay` writes the paste-ready prompt and tells you to run `/rewind fork N`. |
| How do we map prompt N to a code state? | Checkpoints are a shared bare repo (`<home>/checkpoints/store`, ref `refs/hermes/<sha256(cwd)[:16]>`, a `Hermes-Taken-At` trailer on each commit) with **no prompt index**. | Ladder: Branchpoint's own per-prompt snapshot (exact) → checkpoint by timestamp, overlaid on HEAD → last commit before the prompt → `--base`. The anchor reports which rung it used. |
| Subagent tool name/shape for `dispatch_tool`? | `delegate_task(tasks=[{goal, context}])`; **no per-task cwd**, so it cannot target a worktree. | Builds use a headless `hermes chat -Q -q … --in <worktree> -t file --max-turns --run-budget` subprocess. |

## Deviations from the PRD (and why)

1. **Build agents have file tools only; the plugin runs the checks.** A `-q` child can auto-approve shell commands, and
   catalog rule 12 forbids spawning Hermes children that inherit non-interactive/YOLO mode to run unapproved
   commands. The PRD already wanted tests/lint/types "every 5 tool calls": the plugin does that (a `post_tool_call`
   counter + background run + `transform_tool_result` to show the agent the outcome). The agent never gets a shell.
2. **Worktrees live in the plugin-data dir, not `.branchpoint/` in the repo.** `git status` stays clean and nothing
   appears in the working tree before `adopt`.
3. **Diversity check is lexical + a model check, not embeddings.** The plugin API has completions, not embeddings
   (threshold 0.88 over embeddings becomes 0.55 over TF-IDF, plus unique axes plus an LLM "same diff, same
   footprint?" check).
4. **Stdlib `http.server` SSE instead of aiohttp.** `aiohttp` is not guaranteed in a Hermes install; the plugin has
   zero dependencies.
5. **Blast score maximum is 9, not 10.** The PRD's rule adds up to 2+2+2+1+1+1 = 9; the score is clamped to 0-10 and
   the README says 9.
6. **Council arithmetic is code, not prose.** Reviewers score; weights, ties (smaller blast radius wins) and the ranking
   are computed deterministically; the judge only explains. Reviewer order is shuffled per reviewer, not once.
7. **Citations are enforced, not requested.** The judge's narrative is validated sentence by sentence against the real
   evidence store; unsupported sentences are dropped and counted in an "integrity notes" section. The rendered
   Decision Record is produced from the validated structure, so an uncited claim cannot appear.
8. **Confidence is computed** from score margin, reviewer agreement and fidelity (a simulated winner can never exceed
   "medium"), not taken from the judge.
9. **Adopt `merge` is "make my tree equal route X"** (apply the diff from my tree *as it is now* to X). It works for both
   modes, keeps a safety snapshot, and `undo` reverses it. A literal `git merge` would drag the snapshot commits
   into history and cannot express "replace the baseline I already have in my working tree".
10. **The baseline (route A) is judged on the same evidence as everything else**: its snapshot gets a worktree and the
    project's checks run on it. Its cost comes from the transcript and is labelled an estimate (or "unknown" with no
    transcript).
11. **Simulated correctness is capped at 6/10** in code; a PRD risk ("the council rubber-stamping") needs a mechanical
    guard, not an instruction.

## Added in 0.2.0 (owner direction, 8 Nov 2026)

12. **Routes are agents, not assigned strategies, by default.** The idea is to give the same task to X, Y and Z and let
    each plan and build it its own way (`approach: independent`). The PRD's axis-assigned mode is kept as
    `approach: assigned`. In independent mode there is no diverge step: each agent states its own approach in its plan,
    and two agents that land on the same idea are built once (the diversity check runs on the finished plans instead of
    on proposals).
13. **2 to 8 routes, default 4 (A-D).** `--n`/`default_routes`. Agents take models in order from `--models`/`models`
    (repeating if fewer); the reviewers and the judge have their own `reviewer_model` / `judge_model`.
14. **Decision log** (`declog.py`): entries are written at the point of decision from facts the code holds, never
    reconstructed afterwards; the summary is deterministic text built from them, with no extra model call.
15. **Converge** (`Run.converge`), which the PRD listed as out of scope ("automatically merging two executed routes"):
    one more agent starts from the winner's code with the judge's borrow/risk notes, is judged blind with every other
    route, and may lose. It is a build by one agent, not a textual merge.

## Where each PRD file went

| PRD | Here |
|---|---|
| `anchor.py` | `anchor.py` (+ hooks record per-prompt snapshots) |
| `diverge.py` | `diverge.py` |
| `route.py` | `route.py`, `agent.py` (headless runner), `checks.py`, `protocol.py` (parent↔child contract) |
| `ledger.py` | `ledger.py` (ledger + evidence store + ref grammar) |
| `blast.py` | `blast.py` |
| `council.py` | `council.py` |
| `adopt.py` | `adopt.py` |
| `view/index.html` | `view/index.html` + `server.py` |
| `skills/branchpoint/SKILL.md` | same |
| `demo/` | `demo/service` (FastAPI + pytest), `demo/routes` and `demo/baseline_ttl` (the four routes), `demo/prepare_demo.py` |
| (new) | `run.py` (orchestration), `commands.py`, `hooks.py`, `llm.py`, `scripts/branchpoint` (standalone), `scripts/offline_demo.py` |

## What is verified, and how

* 64 offline tests (pipeline on real git worktrees with a scripted model, council arithmetic and citation checks, blast
  tracer, anchor ladder incl. a synthetic Hermes checkpoint store, hooks, server security, packaging rules).
* 2 real-process tests against a real Hermes host with a scripted model endpoint (no credentials): a headless build
  agent with the plugin loaded (confinement, file-tools-only, usage metering, mid-build checks fed back), and the
  full `/branchpoint fork` slash command inside a real host.
* `hermes plugins validate` against a clean upstream checkout: all checks pass.
* Real-model planning runs (Haiku 4.5) on the demo service in both modes: strategies, simulation, blind council and a
  fully cited Decision Record. Real-model *builds* depend on the user's model and are exercised by the user's own run.
