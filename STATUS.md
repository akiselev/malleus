# Malleus status

Updated: 2026-08-20
Branch: `agent/fc0-fc1-v2-form-audit`
Milestone: Physics Factory FC0-FC1

## Current role

Malleus compiles finite-precision pointwise/local programs. Resolvent owns formulation and typed
variational meaning; Sinbad owns mesh, basis/quadrature, assembly, and field state; Solverang owns
numerical algorithms. Malleus must not absorb global topology, assembly policy, or solver state.

## Implemented on this branch

- Pins Resolvent FC0-FC1 revision
  `ba0be14061afe8057e8bdc86eec93873c10212ea`.
- Adds a versioned V2 artifact audit boundary that verifies Resolvent content digests, artifact
  stage, payload schema, and form validation before any local compiler consumes the form.
- Inventories scalar/tensor, complex, gradient, time, contraction, conjugation, transpose, trace,
  and facet/interface requirements per integral and for the whole form.
- Carries the canonical finite-element complex-inner convention explicitly: the right/test operand
  is conjugated.
- Exposes the digest-bound scalar-H1 compatibility oracle when present.
- Reports structured TensorIR/QFunction kernel generation as explicitly deferred to FC6. FC4 is
  TensorIR preprocessing/reference interpretation and FC5 is QFunction/operator factorization;
  FC0-FC1 do not claim a generated local kernel.
- Confirms assembly level is not part of form identity and carries Resolvent's truthful derivative
  artifact and operator-claim state through inspection.
- Adds tests for scalar transient diffusion, tensor-axis inventory, wrong-stage/schema rejection,
  and tampered-artifact rejection with stable diagnostics.

## Validation gate

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Cross-repository contract

Sinbad must pin this exact Malleus revision and the exact Resolvent revision above. The FC0-FC1
runtime slice may execute only through the retained V1 scalar-H1 oracle after confirming the V2
artifact and Malleus audit agree on digests. Structured kernel generation begins at FC6.

## Next work

FC6 consumes the indexed TensorIR and QFunction/operator-factorization artifacts established by
FC4-FC5 and emits actual structured local kernels. No FC2-FC3 mesh, element, quadrature, or
assembly ownership moves into Malleus.
