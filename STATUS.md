# Malleus status

Updated: 2026-08-20
Branch: `agent/fc0-fc1-v2-form-audit`
Milestone: Physics Factory FC0-FC1

## Current role

Malleus compiles finite-precision pointwise/local programs. Resolvent owns formulation and typed
variational meaning; Sinbad owns mesh, basis/quadrature, assembly, and field state; Solverang owns
numerical algorithms. Malleus must not absorb global topology, assembly policy, or solver state.

## Implemented on this branch

- Pinned Resolvent FC0-FC1 revision
  `b7f6b9f9cffd00d62447c94c2ea0414102db54a0`.
- Added a V2 artifact audit boundary that verifies Resolvent content digests and receipts before any
  local compiler consumes the form.
- Inventories scalar/tensor, complex, gradient, time, contraction, conjugation, transpose, trace,
  and facet/interface requirements per integral and for the whole form.
- Exposes the digest-bound scalar-H1 compatibility oracle when present.
- Reports structured TensorIR/QFunction generation as explicitly deferred to FC4; FC0-FC1 do not
  claim a generated local kernel.
- Confirms assembly level is not part of form identity and carries Resolvent's truthful derivative
  artifact and operator-claim state through inspection.
- Added tests for a scalar transient diffusion form and tampered-artifact rejection with stable
  diagnostics.

## Validation state

Pending the branch CI tuple:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Cross-repository contract

Sinbad must pin this exact Malleus revision and the exact Resolvent revision above. The first runtime
slice may execute only through the retained V1 scalar-H1 oracle after confirming the V2 artifact and
Malleus audit agree on digests. Structured kernel generation remains FC4+.

## Next work

FC4 consumes indexed TensorIR/QFunctionIR and emits actual structured local kernels. No FC2-FC3
mesh, element, quadrature, or assembly ownership moves into Malleus.
