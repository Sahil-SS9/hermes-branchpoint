# Changelog

## 0.2.3

Second confinement review (catalog review of 0.2.2, NousResearch/hermes-agent#152959).

- **The sandbox masks the real home.** 0.2.2 put its `--tmpfs` over the sandbox's own fresh temp home, so after
  `--ro-bind / /` a check could still read `~/.hermes/.env`, `~/.ssh` and other profiles, and failing-test output goes
  back to the build model. The mask now covers `Path.home()` and the Hermes home when it lives outside it; the linked
  `.venv` / `venv` / `node_modules` and the check's interpreter install (with the symlinks leading to them) come back
  read-only, the sandbox's own home and the worktree come back writable. macOS gets
  `(deny file-read* (subpath <home>))` with the same paths re-allowed.
- **The probe plants its canary under the real home**, and judges writes on the host afterwards. The old write test
  touched the tmpfs it had just mounted, so on a host whose temp dir sits under the home the probe failed and Branchpoint
  treated a working bubblewrap as absent.
- **No unsandboxed mypy probe.** Detecting checks no longer runs `python -m mypy --version` with the worktree as cwd
  (an agent-written `mypy.py` would have run on the host with every key); the sandboxed `mypy .` reports a missing mypy.
- **Run ids are validated** against the `new_run_id` shape (`^\d{4}-\d{6}-[0-9a-f]{4}$`) before they become paths.
- **`--allow-sensitive` is gone** from the usage text and the parser: it was parsed and never read. The adopt-time gate it
  promised is not built.
- A command the sandbox cannot find is reported as "command not found", as it is unsandboxed.
- `__version__` matches `plugin.yaml`.

## 0.2.2

Confinement hardening. Two paths let a build agent run arbitrary code as the user, both outside the hook guard;
both are closed, and the confinement claim no longer rests on the harness's hooks.

- **Checks run in an OS sandbox.** `pytest`/`npm test`/ruff/mypy execute code from inside the worktree, which the
  build agent wrote — a two-line `tests/conftest.py` was enough to write outside the worktree and read a provider key
  from the environment. Every check now runs under `bwrap` (Linux) or `sandbox-exec` (macOS): worktree writable,
  everything else read-only, home directory masked, no network, allowlisted environment (no denylist of four variables).
  The backend is probe-verified (positive control + three denials) before a build, because a bwrap that cannot create a
  user namespace passes every denial for free. With no passing backend, `--execute` refuses; `--unsafe-checks` accepts
  unsandboxed checks explicitly and is recorded in the Decision Record.
- **Parent git calls are pinned and hardened.** A linked worktree's `.git` is a one-line file inside the confinement
  root; pointing it at an agent-built git dir carrying `core.fsmonitor` made the parent's `git add` execute that command
  (`--no-verify` does not help — fsmonitor is not a hook). Commits now use explicit `--git-dir`/`--work-tree`, hardened
  `-c` flags, `GIT_CONFIG_NOSYSTEM`/`GIT_CONFIG_GLOBAL`, and refuse when `<wt>/.git` no longer matches the pointer git wrote.
- **The tool guard fails closed.** Per-tool argument schemas (unknown keys refused, so `{"file": …}`, `{"paths": [...]}` and
  other harnesses' `filePath`/`edits[]` no longer pass), unified-diff and rename headers parsed, and a patch body with no
  readable header refused. New protected-path list: `.git*`, `.gitattributes`, `.gitmodules`, `.envrc`, `.vscode`,
  `.claude/`, `.codex/`, `.opencode/`, `.pi/`, `.hermes/`, `.commandcode/`, `.github/workflows/`.
- **The fingerprint covers every shipped module**, not three, so a stale `util.py` or `checks.py` can no longer pass the
  hook proof.
- **The secret-file list is broader**: `.envrc`, `.git-credentials`, `hosts.yml`, `kubeconfig`, `*.tfvars`,
  `terraform.tfstate*`, `.docker/config.json`, `service-account*.json`.
- **Dotfile paths survive normalisation.** `lstrip("./")` strips characters, so `.github/ci.yml` became `github/ci.yml`
  and plans for dotfiles lost their blast radius and excerpts.
- **Bounded anchor log.** Pruning deleted the same ref on every turn, so the turn log grew without limit; it is now
  rewritten to the cap. Timestamps parse their own offset instead of assuming local time (a DST change could rewind to
  the wrong checkpoint).
- **No worktree-local interpreter.** A `.venv/bin/python` that is a file *inside* the worktree is the agent's, not the
  user's linked venv, so it is no longer chosen.
- **Failed runs stop their builds.** The pipeline's thread pool is a context manager, every failure path sets `abort`,
  pending futures are cancelled, and `meta["warnings"]` has one locked writer.
- **The view server validates the run id** and compares the token in constant time (`__RUN_ID__` is substituted into the
  page and passed to `Run.load`, so an unchecked id was reflected injection and a `../` path out of `runs/`).
- **`python -m pytest` is the offline suite.** e2e tests skip when no `hermes` binary is present instead of erroring.

Not in this release (the review's later phases, unchanged behaviour, still documented as limits): the core/adapter split
behind the five ports, the CLI/`branchpoint-guard`/MCP front doors, per-harness adapters, a detached worker process with
heartbeat and resume, per-route budget reservation, adopt-time execution-sensitive-path gating (`--allow-sensitive` was
parsed but the gate is not wired; removed in 0.2.3), council fencing of agent-authored text, and claim-type/ref-kind matching. Confinement
is now stronger than the hook handshake it replaces, but it is still the sandbox's boundary, not a claim of immunity to
kernel bugs; a sandbox that fails its probe is treated as absent.

## 0.2.1

Safety and evidence corrections:

- Stop if Hermes refuses a chosen planning, reviewer or judge model. Never silently switch models.
- Treat crashed, timed-out and empty-diff builds as failed routes, not successful candidates.
- Refuse real builds when Branchpoint is missing, disabled or stale in the child's Hermes home. Verify the installed hook code with a no-model native preflight and require a report from each build child.
- Protect secret-looking files from reads, writes, symlink aliases and broad content searches. Scrub common credential patterns without private host imports.
- Limit standalone planning and council chats to session-local TODO bookkeeping rather than the broader `safe` toolset.
- Preserve the previous verdict when converge fails. Retrying in the same session asks the chosen model again after permission is granted, without adding duplicate Z routes.
- Respect zero token caps; report missing child usage as estimated rather than measured. Return a nonzero standalone exit code on failure.
- Serialize Git worktree metadata mutations per repository (including the hook preflight), preventing parallel routes from seeing a sibling's partially published `commondir`. Builds and checks remain parallel.
- Prepare isolated real-host test environments separately from product assertions, so a cold bootstrap cannot masquerade as a product timeout.

Limits are documented in the README: plugin-level confinement is not an OS sandbox, budget accounting is not a live cross-process meter, inherited failing checks remain visible, and a council recommendation is not release approval.

The published 0.2.0 tag is unchanged. This correction is a new patch release.
