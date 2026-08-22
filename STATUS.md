# Malleus status

Updated: 2026-08-21
Branch: `master`
Milestone: SV0-B2 reusable local differential campaigns

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
- A module-complete `run_local_differential_campaign` API over caller-supplied local buffers. It
  compares primal, JVP, VJP, and parameter-only JVP execution between the deterministic
  interpreter and a distinct version-identified in-process `LocalExecutableRunner`, checks
  centered differences and the JVP/VJP adjoint identity, and returns explicit per-check
  tolerances/errors. Reference-interpreter self-comparison is refused.
- Campaign validation requires exactly one case per module kernel, finite and sufficiently sized
  operand buffers, disjoint state/parameter directions, seeds for every writable dependent, a
  positive finite step, and nonnegative finite componentwise absolute-or-relative tolerances.
  Missing coverage, ambiguous roles, backend refusal, non-finite output, and
  completed-but-mismatching execution remain distinct outcomes.
- Retained numeric-policy mutation fixtures cover f64-to-f32 demotion, f32-to-f64 promotion, and
  reduction-order toggles. Reduction-order mutation executes a deterministic reversed loop order;
  a mutation check passes only when at least one local output component leaves both declared
  tolerances. Inapplicable mutations are refused.

The former scalar opcode stream, Cranelift JIT, compiled Newton step, Resolvent
bridge, scientific compatibility metadata, property layer, and JIT tests have
been removed. Git history is the archive; there is no compatibility surface.

## Validation

Passed locally on 2026-08-21:

- `cargo fmt --all -- --check`
- `cargo check --locked --all-targets --all-features`
- `cargo clippy --locked --all-targets --all-features -- -D warnings`
- `cargo test --locked --all-features` — 16 tests (1 unit, 15 integration) and 0 doctests passed
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps`
- `cargo test --locked --doc`
- `git diff --check`

## Current limits and next work

- The reference interpreter uses `f64` storage while rounding loads and every
  operation according to the declared f32/f64 policy.
- Tile, vectorization, and parallel decisions are validated metadata; the
  reference interpreter intentionally serializes execution.
- Structured JVP and VJP are implemented; materialized Jacobians and
  differentiation through read-write state operands remain explicit refusals.
- SV0-B2 campaigns are local conformance checks, not scientific verification or support-promotion
  evidence by themselves. They do not select scientific objectives, norms, operating envelopes,
  meshes, global operators, solvers, or histories. The executable runner is an in-process
  interface; external process/tool lineage belongs to Sinbad/Outboard campaign infrastructure.
- Bounds and simple injective-write maps are proved conservatively; general
  affine injectivity and overlapping external-region proofs remain future work
  before production parallel schedules.
