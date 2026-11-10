# Changelog

## 0.2.1

Safety and evidence corrections:

- Stop if Hermes refuses a chosen planning, reviewer or judge model. Never silently switch models.
- Treat crashed, timed-out and empty-diff builds as failed routes, not successful candidates.
- Refuse real builds when Branchpoint is missing, disabled or stale in the child's Hermes home. Verify the installed hook code with a no-model native preflight and require a report from each build child.
- Protect secret-looking files from reads, writes, symlink aliases and broad content searches. Scrub common credential patterns without private host imports.
- Limit standalone planning and council chats to session-local TODO bookkeeping rather than the broader `safe` toolset.
- Preserve the previous verdict when converge fails. Retrying in the same session asks the chosen model again after permission is granted, without adding duplicate Z routes.
- Respect zero token caps; report missing child usage as estimated rather than measured. Return a nonzero standalone exit code on failure.
- Prepare isolated real-host test environments separately from product assertions, so a cold bootstrap cannot masquerade as a product timeout.

Limits are documented in the README: plugin-level confinement is not an OS sandbox, budget accounting is not a live cross-process meter, inherited failing checks remain visible, and a council recommendation is not release approval.

The published 0.2.0 tag is unchanged. This correction is a new patch release.
