# Malleus status

Updated: 2026-08-16
Branch: `agent/r13-r20-wave-a-f`
Milestone: Waves B-F / R13-R18 local-kernel responsibilities

## Current role

Malleus compiles finite-precision pointwise/local programs. Resolvent owns scientific/discrete meaning; Sinbad owns field/topology/runtime state; Solverang owns numerical algorithms. Malleus must not absorb mesh traversal, global assembly, coupling orchestration, or simulation history.

## Implemented on this branch

- R13: `CompiledKernelBundle` for primal/JVP/VJP/parameter-derivative pointwise programs with stable binding layouts.
- R15: property kernels for constants, linear expressions, 1-D tables, explicit physical/validity guards, derivatives, and external-provider boundaries.
- R16: constitutive-kernel metadata for primal outputs, tangents, parameter derivatives, and stateful/local distinction.
- R17: element-kernel binding contract exposes field values/gradients, geometry, quadrature weights, properties, and constitutive responses without importing global topology.
- R18: block identities and deterministic cross-block shared-evaluation planning for property/constitutive CSE.
- Existing direct Resolvent execution-plan lowering remains the finite-precision scalar compilation path.
- Permanent Rust CI now runs rustfmt, clippy with warnings denied, and all-feature tests.

## Validation state

The initial CI cycle reached all new tests: rustfmt and clippy passed and 14/15 tests passed. The sole failure was a test expecting construction order instead of deterministic `BTreeMap` order; that assertion has been corrected and the branch was rustfmt-normalized. This user-authored status commit retriggers normal CI after GitHub marked the formatter-bot commit `action_required`.

Do not mark the branch verified until the retriggered CI is green.

## Cross-repository contract

After Resolvent PR #9 has a green final Wave revision, update Malleus's exact Resolvent git revision and run CI again. Sinbad's federation lock will record that passing tuple.

## Remaining before merge

1. Confirm current Malleus-only CI is green.
2. Pin the final passing Resolvent Wave revision.
3. Re-run CI after the pin change.
4. Update this file with the exact green dependency revision.
