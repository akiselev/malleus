# malleus-cranelift

Optional Cranelift JIT backend for Malleus's validated `Executable`, including kernels
produced by JVP/VJP differentiation and FMA contraction. It is a workspace member
(`crates/malleus-cranelift`, package `malleus-cranelift`) and never a dependency of the
`malleus` core, which stays dependency-light and free of unsafe code. Promoted from
`experiments/cranelift-compiler` in W11 M2 (no facade remains).

## Reproduce

From the repository root, using Rust 1.93+ on a 64-bit native host:

```sh
cargo test -p malleus-cranelift --locked
cargo run -p malleus-cranelift --release --locked --quiet > crates/malleus-cranelift/results.json
cargo clippy -p malleus-cranelift --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc -p malleus-cranelift --locked --no-deps
```

Direct Cranelift codegen/frontend/module/jit/native dependencies are pinned to `=0.128.3`;
the workspace `Cargo.lock` records every transitive version.

## Implemented contract

- Everything the reference interpreter executes, bitwise (NaNs equivalent), at every
  numeric level (`malleus::policy`): f32/f64 constants, indices, loads, ordered locals,
  neg/abs/sqrt/floor/ceil and add/sub/mul/div as native instructions; explicit `Fma` as one
  Cranelift `fma` (single rounding); exp/ln/sin/cos/tan/pow/atan2/min/max and min/max
  reductions as calls to `malleus_unary`/`malleus_binary`, registered JIT symbols that run
  `malleus::policy::{unary, binary}` with the kernel's math library (host `std` at L1/L2,
  pinned `libm` at L0). f32 values are widened exactly for the call and narrowed exactly
  after it, so f32 rounding happens inside the reference routine. No implicit contraction or
  reassociation.
- Lazy selects and short-circuit predicates, all comparisons, sequential nested loops,
  affine addressing, validated dense layouts, regions, Write/ReadWrite stores and all four
  reductions. Canonical reduction order overrides loop permutation.
- One generated function per kernel with the ABI
  `extern "C" fn(table: *const *mut f64, strides: *const usize, points: usize)`: a native
  loop over points, each rebasing operand `i` by `p * strides[i]` elements. This is exactly
  the `malleus::prepared` batch contract (`points` sequential calls), so shared and
  overlapping strides are executed, not refused.
- Entry points: `JitKernel::{compile, prepare, prepare_owned, run}` and
  `PreparedCall::{update_input, values, run}` (unchanged names; checked `BufferBinding`s);
  positional `JitKernel::{call, run_batch}` over `malleus::prepared::{ReadBinding,
  WriteBinding}` with a reusable `JitState`, refusing exactly what the core
  `PreparedKernel::check_bindings` refuses (same `CallError`), allocation-free once warm
  (`tests/alloc.rs`). `PreparedCall::run` re-derives its raw pointers on every call (the
  cached-pointer-then-reborrow pattern is UB under Stacked/Tree Borrows; unit-tested so that
  it runs under miri).
- `JitCache`: thread-safe in-memory cache of `Arc<JitKernel>` keyed by `CacheKey { kernel
  digest, schedule, level, BACKEND_VERSION, cpu_features(), BATCH_ABI }`; no disk cache.
  `CampaignRunner` (identity `BACKEND_VERSION`) uses it for the local differential campaign.

Explicit typed refusals remain for tiled/vectorized/parallel schedules, non-64-bit hosts and
kernels exceeding the IR/address budgets. Compositions and facet wrappers are not lowered as
units. There is no SIMD lowering across points (the batch loop is scalar per point) and no
automatic interpreter fallback.

## Historical measurement (pre-promotion experiment)

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
The core suite now has 45 tests (see the root `STATUS.md`). Strict Clippy and rustdoc checks also pass
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
The prototype now has 18 passing tests, including a seeded random interpreter-versus-JIT
differential over generated interval programs. Reusable calls can safely update Read
inputs and inspect outputs between invocations. The original plain-arithmetic
benchmark above is retained as historical data for the initial compiler commit;
the bounded-kernel comparison and native Rust fast-path repair have separate
results in CADabra's experiment.

The interval pass is now in the core `malleus::interval` API and is shared by the Rust source backend. See the root `RUST-BACKEND.md`; the core emitter adds no native compiler dependencies.
