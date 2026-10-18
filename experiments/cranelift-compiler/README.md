# Malleus Cranelift compiler experiment

This is a working local JIT backend for Malleus's existing `Executable`, including
kernels produced by its JVP/VJP differentiation. It lowers validated Malleus IR to
Cranelift machine code; it does not interpret another expression language. The
core crate and its dependencies are unchanged. This isolated workspace contains
the native dependency and unsafe ABI boundary.

## Reproduce

From the repository root, using Rust 1.93.0 on a 64-bit native host:

```sh
cargo test --manifest-path experiments/cranelift-compiler/Cargo.toml --locked
cargo run --manifest-path experiments/cranelift-compiler/Cargo.toml --locked --quiet > results.json
cargo clippy --manifest-path experiments/cranelift-compiler/Cargo.toml --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --manifest-path experiments/cranelift-compiler/Cargo.toml --locked --no-deps
```

The benchmark configuration intentionally compiles the core interpreter at Rust
`opt-level=3`, the prototype host at `opt-level=1`, and native code with Cranelift
`opt_level=speed`. Do not compare against an unoptimized interpreter. Direct
Cranelift codegen/frontend/module/jit/native dependencies are pinned to 0.128.3;
`Cargo.lock` records every transitive version, including compatible 0.128.4 crates.

## Implemented contract

- f32/f64 local arithmetic: constants, indices, loads, ordered locals, negation,
  absolute value, square root, floor/ceil, add/subtract/multiply/divide.
- Lazy selects and short-circuit predicates, all six comparisons, not/and/or.
- Sequential nested loops, affine addressing with signed coefficients, arbitrary
  validated dense layouts, buffer regions, Write/ReadWrite stores, additive and
  multiplicative reductions. Canonical reduction order overrides loop permutation;
  schedule-defined reduction order follows the declared permutation.
- f32 operations use native f32 instructions, with conversion from/to the core's
  f64 buffer ABI. No implicit FMA or reassociation. NaN payload preservation is not
  promised (the comparison considers any two NaNs equivalent).
- `JitKernel::run` checks binding identities, duplicates, lengths and completeness.
  `prepare` performs these checks once and borrows buffers plus the code owner for
  repeated calls. Neither API implicitly clears reduction outputs.
- Machine-code pointers stay private. The owning module is retained during calls
  and freed on drop, including compilation failures. Validated affine bounds,
  checked bindings and conservative IR/address budgets protect native accesses.
- `CampaignRunner` implements the existing `LocalExecutableRunner`; its in-memory
  cache uses the kernel digest and complete schedule. It feeds the actual Malleus
  differential campaign, rather than a replacement campaign.

Explicit typed refusals cover transcendental functions, pow/min/max/atan2,
min/max reductions, tiled/vectorized/parallel schedules, non-64-bit hosts, and
kernels exceeding prototype budgets. IteratorKind::Parallel denotes independent
iterations and is admitted but currently executed sequentially. Composition and
facet wrappers are not lowered as units. There is no disk cache, object-file/AOT
backend, SIMD lowering, automatic interpreter fallback, or production cache policy.

## Measured result

`results.json` records the full run on 2026-10-18, x86_64 AMD EPYC 9V74 under KVM.
Numbers are medians of seven batches; compilation batches contain three kernels
and exclude module teardown. Execution uses identical repeat counts and existing
bindings for both backends. Compilation includes lowering and JIT finalization
but excludes Malleus structural validation and differentiation. "Checked" includes
binding validation and pointer-table allocation on every call; "prepared" pays
those costs once. Benchmarks compare outputs before timing. Repeated reductions
intentionally retain output state in both backends, as their execution contract
requires; no per-call reset or buffer allocation is timed. This is a single-host
microbenchmark with no CPU affinity or confidence intervals.

| Kernel | Interpreter | Checked JIT | Prepared JIT | Checked speedup | Compile / checked break-even |
|---|---:|---:|---:|---:|---:|
| One guard polynomial | 232 ns | 48 ns | 2.2 ns | 4.8x | 223 us / 1,213 calls |
| 64 guard evaluations | 10.25 us | 95 ns | 54 ns | 108x | 276 us / 28 calls |
| 4,096 guard evaluations | 603 us | 3.38 us | 3.33 us | 178x | 218 us / 1 call |
| Quadratic JVP, 1,024 inputs | 247 us | 3.37 us | 3.33 us | 73x | 259 us / 2 calls |
| Quadratic VJP, 1,024 inputs | 221 us | 3.39 us | 3.35 us | 65x | 259 us / 2 calls |
| Dense matvec 64x64 | 396 us | 8.32 us | 8.11 us | 48x | 256 us / 1 call |

All 13 benchmark outputs match the interpreter bitwise (NaNs equivalent). These
speedups measure Malleus interpretation versus native execution, not versus the
Cadabra specialized Rust evaluator, and not a complete Boolean speedup. Sub-3 ns
prepared scalar timings are indirect-call microbenchmarks, not application latency.

Validation: 13 differential tests cover f32/f64 and empty domains, generated
JVP/VJP, the actual seven-check derivative campaign (including centered differences,
parameter-only directions and the adjoint identity), layout/offset/reversed maps,
canonical versus schedule-defined reductions, scalar domains, signed zeros,
NaNs/infinities, every admitted arithmetic/comparison operation, multiply reductions,
ReadWrite effects, repeated prepared calls, malformed bindings and unsupported
schedules. An adversarial rounding test checks that forbidden FMA is not introduced.
The unchanged core suite has 39 tests. Strict Clippy and rustdoc checks also pass
for both workspaces.

## Consequence for Cadabra guards and dragging

The compiler hypothesis is supported: local derivative kernels can be compiled
once and repeatedly evaluated much faster than interpreting the same IR. Compile
and cache the *kernel shape*, leaving geometry and drag parameters in buffers;
recompiling on every pointer movement would erase much of the small-kernel gain.
Prepared bindings suit a drag session when its buffers remain fixed.

This does not certify a topology-stable range. These instructions implement ordinary
finite-precision arithmetic; derivative sampling or a Newton prediction alone is
not a bound. A valid pipeline still needs:

1. Resolvent-owned exact/interval coefficient derivation, with a roundoff enclosure.
2. A complete event-guard set for the supported Boolean domain, including swept
   broad-phase exclusions so previously absent candidate pairs cannot appear silently.
3. Whole-interval sign/Jacobian bounds for those guards, plus the current session,
   identity, publication and supported-domain conditions. Strict nonzero guards
   preserve their branches only while all these obligations hold.
4. A geometry/session-versioned validity token; stop at its boundary and use the
   robust path when any proof is inconclusive. Native derivative evaluation can
   predict where to refine, while certified enclosures justify skipping work.

Cadabra's existing `zz-boolean-sensitivity-experiment` already specializes exact
quadratic guards into native Rust interval evaluators with exact Bernstein fallback.
It is the stronger starting point for range certification. This JIT demonstrates
the intended general execution owner, not a replacement for those proofs. Next,
lower an explicit interval/error-bound guard kernel to Malleus and cross-check its
enclosures against Resolvent exact arithmetic before allowing it to issue tokens.
Keep topology and token ownership in Cadabra; Malleus should only execute local
numeric kernels.

## Bounded arithmetic integration (2026-10-18)

[INTERVALS.md](INTERVALS.md) describes the new explicit endpoint/error-bound pass,
its roundoff argument and limits, and the CADabra exact-arithmetic integration.
The prototype now has 17 passing tests. Reusable calls can safely update Read
inputs and inspect outputs between invocations. The original plain-arithmetic
benchmark above is retained as historical data for the initial compiler commit;
the bounded-kernel comparison and native Rust fast-path repair have separate
results in CADabra's experiment.
