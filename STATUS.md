# Malleus status

Updated: 2026-08-21
Branch: `master`
Milestone: FC11 complete structured-IR serialization

## Current role

Malleus owns backend-independent, finite-precision local kernel IR, structural
validation, scheduling contracts, and reference execution. It does not own a
scientific language, mesh topology, finite-element spaces, global assembly,
solver policy, coupled state, or simulation history.

## Implemented

- One `malleus` crate with Serde as its only runtime dependency.
- `StructuredKernel` and `StructuredModule` with fixed iteration domains,
  affine indexing maps, explicit buffer regions and dense layouts, ordered SSA-like locals, scalar
  expressions, predicates, stores, and reductions.
- Explicit numeric policies and backend-neutral JVP/VJP/Jacobian request types;
  schedule-independent structured forward and reverse AD emits new validated IR.
- Deterministic validation of operand names/maps, ranks, layout permutations,
  affine bounds, write effects/aliasing, axes, access modes, local definition
  order, derivative requests, and module kernel names.
- `KernelSchedule`, `Executable`, and `ExecutableModule::reference` as the
  backend boundary.
- Deterministic sequential `Interpreter::run` over caller-owned buffers with
  row/column-major layouts, canonical reductions, and declared f32/f64 operation
  precision.
- Complete Serde representations for schedule-independent modules, kernels, operands, indexing,
  expressions, effects, numeric policy, and derivative products. Consumers re-run structural
  validation after decoding before constructing executables.
- Focused tests for module compilation, pointwise execution, reductions, f32
  precision, forward finite differences, reverse adjoint dot products, parameter
  selections, and malformed locals/indexes/layouts/effects.

The former scalar opcode stream, Cranelift JIT, compiled Newton step, Resolvent
bridge, scientific compatibility metadata, property layer, and JIT tests have
been removed. Git history is the archive; there is no compatibility surface.

## Validation

Passed locally on 2026-08-21:

- `cargo fmt --all -- --check`
- `cargo check --all-targets --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-features` — 9 integration tests and 0 doctests passed
- `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`
- `cargo test --doc`
- `git diff --check`

## Current limits and next work

- The reference interpreter uses `f64` storage while rounding loads and every
  operation according to the declared f32/f64 policy.
- Tile, vectorization, and parallel decisions are validated metadata; the
  reference interpreter intentionally serializes execution.
- Structured JVP and VJP are implemented; materialized Jacobians and
  differentiation through read-write state operands remain explicit refusals.
- Bounds and simple injective-write maps are proved conservatively; general
  affine injectivity and overlapping external-region proofs remain future work
  before production parallel schedules.
