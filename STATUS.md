# Malleus status

Updated: 2026-08-16
Branch: `agent/r13-r20-wave-a-f`
Milestone: Waves B-F / R13-R18 local-kernel responsibilities

## Current role

Malleus compiles finite-precision pointwise/local programs. Resolvent owns scientific and discrete semantics; Sinbad owns field/topology/runtime state; Solverang owns numerical algorithms.

## Implemented on this branch

- R13: `CompiledKernelBundle` contract for primal/JVP/VJP/parameter-derivative pointwise programs with stable binding layouts.
- R15: property kernel contract for constants, linear expressions, 1-D tables, explicit validity/physical guards, derivatives, and external-provider boundaries.
- R16: constitutive-kernel metadata for primal outputs, tangents, parameter derivatives, and stateful/local distinction.
- R17: element-kernel binding contract exposes field values/gradients, geometry, quadrature weights, properties, and constitutive responses without importing global topology.
- R18: block identities and cross-block shared-evaluation planning for property/constitutive CSE.
- Existing R9 direct Resolvent execution-plan lowering remains the scalar compilation path.

## Validation state

Local Rust validation is unavailable in the execution sandbox because rustup cannot reach its download service. GitHub Actions must establish format/clippy/test status. Do not consider this branch verified until CI is green.

## Cross-repository contract

The final branch must pin the exact passing Resolvent Wave A-F commit. Until that synchronization step, the existing Resolvent R0-R9 pin remains in `Cargo.toml` and the new kernel-contract module deliberately depends only on already-stable execution-plan types.

## Next

1. Run/fix CI for this branch.
2. Update the Resolvent git revision to the final passing Wave A-F commit.
3. Re-run CI after pin synchronization.
4. Consume these bundles from Sinbad's generic element/coupled runtime.
