# malleus

Differentiable, retargetable graph compiler: a scalar opcode IR plus Cranelift
codegen for residuals, Jacobians, and compiled Newton steps.

`malleus` is the JIT that used to live in [solverang](https://github.com/akiselev/solverang).
Solverang and [Sinbad](https://github.com/akiselev/sinbad) now depend on this
crate. Later milestones add reverse-mode AD, e-graph rewriting, and non-CPU
backends; this release is a faithful extraction of the existing CPU JIT.

## Usage

```toml
malleus = { git = "https://github.com/akiselev/malleus" }
```

The `jit` feature is on by default. Disable it when you only need the crate to
exist as a dependency graph placeholder:

```toml
malleus = { git = "https://github.com/akiselev/malleus", default-features = false }
```

## Surface

Build an opcode stream with `OpcodeEmitter`, then compile it with `JITCompiler`:

- `JITFunction` — `evaluate_residuals` / `evaluate_jacobian` (and fused / dense variants)
- `CompiledNewtonStep` — a whole Newton step compiled to native code

`jit_available()` reports whether the current platform is supported (x86_64 and
aarch64 on Linux, macOS, and Windows).

## License

MIT OR Apache-2.0
