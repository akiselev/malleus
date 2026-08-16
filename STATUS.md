# malleus — STATUS

## Current (M1)

Faithful CPU JIT: scalar opcode IR (`ConstraintOp` / `CompiledConstraints`) plus
Cranelift codegen (`JITCompiler` / `JITFunction` / `CompiledNewtonStep`).

- Builds standalone (`cargo test`), 28 tests (12 unit + 16 integration).
- Compiles with `--no-default-features` (JIT off).
- `jit` feature is default-on and pulls Cranelift 0.116.

Public surface (crate root): `CompiledConstraints`, `ConstraintOp`,
`JacobianEntry`, `HessianEntry`, `Reg`, `ValidationError`, `OpcodeEmitter`,
`JITCompiler`, `JITError`, `JITFunction`, `CompiledNewtonStep`, `JitMode`,
`JITConfig`, `jit_available`.

Consumers:

- [solverang](https://github.com/akiselev/solverang) — `jit` feature re-exports
  this crate under `solverang::jit`.
- [sinbad](https://github.com/akiselev/sinbad) — workspace git dependency.

## Unsafe

`#![deny(unsafe_op_in_unsafe_fn)]`. Unsafe is only the JIT call boundary and the
code-pointer transmutes that mint function pointers (`cranelift.rs`,
`compiled_newton.rs`). Each site has a `// SAFETY:` comment.

## Deferred to M2+

- Reverse-mode automatic differentiation
- E-graph rewriting / equality saturation
- Revolve checkpointing
- Non-CPU backends (GPU / WASM / FPGA)
- numeric-contracts `LinearOperator` generation
