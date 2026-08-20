# Agent instructions

Read `STATUS.md` before substantial work and update it before every handoff or pull request.

## STATUS.md policy

`STATUS.md` is a compact current-state ledger, not an append-only journal.

- Target under 200 lines and never exceed 300 lines; compact stale material instead of appending indefinitely.
- Record the current milestone, implemented capabilities, exact validation results, blockers, downstream/upstream contracts, and next concrete work.
- Never label code verified unless the relevant tests actually ran successfully.
- Keep historical narrative in commits, PRs, ADRs, or design documents.
- Record exact external dependency pins if any are introduced; the core crate should remain independent.

Malleus must remain a local/pointwise compiler. Do not absorb mesh topology, basis traversal, global assembly, solver strategy, or simulation history state.
