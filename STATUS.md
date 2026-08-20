# Malleus status

Updated: 2026-08-20
Branch: `master`
Milestone: structured local-kernel compiler reset

## Current role

Malleus owns backend-independent, finite-precision local kernel IR, structural
validation, scheduling contracts, and reference execution. It does not own a
scientific language, mesh topology, finite-element spaces, global assembly,
solver policy, coupled state, or simulation history.

## Implemented

- One dependency-free `malleus` crate.
- `StructuredKernel` and `StructuredModule` with fixed iteration domains,
  affine indexing maps, ordered SSA-like locals, scalar expressions,
  predicates, stores, and reductions.
- Explicit numeric policies and backend-neutral JVP/VJP/Jacobian request types.
- Deterministic validation of operand names/maps, ranks, axes, access modes,
  local definition order, and module kernel names.
- `KernelSchedule`, `Executable`, and `ExecutableModule::reference` as the
  backend boundary.
- Deterministic sequential `Interpreter::run` over caller-owned buffers.
- Focused tests for module compilation, pointwise execution, reductions, and
  malformed local order.

The former scalar opcode stream, Cranelift JIT, compiled Newton step, Resolvent
bridge, scientific compatibility metadata, property layer, and JIT tests have
been removed. Git history is the archive; there is no compatibility surface.

## Validation

Passed locally on 2026-08-20:

- `cargo fmt --all -- --check`
- `cargo check --all-targets --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all-features` — 3 integration tests and 0 doctests passed

## Current limits and next work

- The reference interpreter stores `f64`; target backends must implement the
  declared scalar/numeric policy exactly.
- Tile, vectorization, and parallel decisions are validated metadata; the
  reference interpreter intentionally serializes execution.
- Derivative requests are contracts only. Implement structured differentiation
  as an IR-to-IR pass before adding optimized native backends.
- Add conservative write-alias and affine-bound proofs before enabling parallel
  schedules in a production backend.
