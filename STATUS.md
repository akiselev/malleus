# Malleus status

Updated: 2026-10-18
Branch: `master`
Milestone: W11 M4 interval upgrades (successor-bound padding, centered and affine
enclosures) and FPCore/symbolic export; W11 M2 numeric policy levels, explicit `Fma`, portable transcendentals; W11 M1
prepared kernels and batched execution (`EXECUTION.md`); strict Rust
source/unrolled and core interval lowering; W8/W7/SV0-B2 contracts retained

## Numeric policy levels (W11 M2, module `malleus::policy`)

- `NumericPolicy::level: policy::NumericLevel { StrictPortable (L0), StrictHost (L1, default),
  Contracted (L2) }`, `serde(default, skip_serializing_if = is_strict_host)`: existing kernels
  keep bytes and digests (`tests/digest_pins.rs` pins from base `7bb4ccd`; `tests/policy.rs`
  pins the policy JSON). L3 fixed-tree reductions: not implemented (open).
- L0 evaluates exp/ln/sin/cos/tan/pow/atan2/min/max (and min/max reductions) with the pure-Rust
  `libm` crate pinned `=0.2.16`, `default-features = false` (MIT/Apache; no deps; same code
  path on every host; `policy::PORTABLE_MATH`). L1 keeps host `std`. sqrt/abs/floor/ceil/fma
  are IEEE-exact and shared. Measured L0 vs L1 (glibc, this host): max 1 ulp for every op in
  f64 and f32 on 4,000 samples each incl. specials; asserted bound 2 ulp.
- Explicit `ScalarExpr::Fma { lhs, rhs, addend }` (new variant, existing encodings unchanged),
  single rounding (`mul_add`) in interpreter, prepared point/columnar, Rust emitter; AD rule
  `da*b + a*db + dc` (strict, shaped like the unfused derivative); admitted at every level.
- `policy::contract_fma(&kernel) -> Contraction { kernel, sites }`: fixed sites
  `Add(Mul(a,b),c)`/`Add(c,Mul(a,b))` -> `Fma(a,b,c)`, `Sub(Mul(a,b),c)` -> `Fma(a,b,-c)`,
  `Sub(c,Mul(a,b))` -> `Fma(-a,b,c)`; not through locals or reducing stores; L1 input only,
  output level `Contracted`. Refusals `ContractionError { Level, ImprecisePolicy, Invalid }`.
- Vague `FmaPolicy::Allowed`/`Reassociation::Allowed`: no backend acts on them (strict
  execution); accepted at L1 for identity, refused at L0/L2 by
  `ValidationError::ImpreciseNumericPolicy` (new variant) and by `contract_fma`.
- New root-level names: modules `policy` and `rust` (the latter was private; it now also
  exposes `rust::{lower_rust_batched, BatchedRustOptions}`).
- `lower_rust_batched(e, name, BatchedRustOptions { width (1..=64, default 8), avx2_dispatch })`:
  standalone `pub fn name(points, b0, s0, b1, s1, ..)` with the run_batch contract
  (overflow-checked extents before effects; `points = 0` no-op); disjoint writable windows run
  full lane blocks over `[T; W]` arrays, the rest point by point; optional AVX2
  `#[target_feature]` copy behind `is_x86_feature_detected!` (one documented `unsafe` call).
  Proven by `tests/rust_source.rs::compiled_batched_rust_matches_sequential_interpreter_calls`
  (rustc; L0/L1, f32/f64, Fma, lane-array locals, reductions, JVP/VJP, matvec; disjoint,
  shared and overlapping strides; refusal).

## Local backends

- Core `malleus::prepared::PreparedKernel` (W11 M1; module-scoped, no root re-exports): an `Executable` compiled once into register bytecode
  (lazy point program + eager columnar program), affine addresses folded and proven in
  range at preparation, positional `ReadBinding`/`WriteBinding` calls with zero heap
  allocation on a sized `PreparedState`, Interpreter-compatible checked `run`, and
  `run_batch` (per-binding batch stride, contract = n sequential calls; columnar in
  `BATCH_BLOCK = 32` lanes when written windows are disjoint, sequential otherwise).
  `PreparedModule`, `PreparedComposition` (+ `CompositionState`, interpreter copy
  semantics, no per-call allocation) and the campaign `PreparedRunner`. Bitwise equal to
  the interpreter on every fixture family and 400 random kernels. No new dependency, no IR
  or digest change (pinned in `tests/digest_pins.rs`). See `EXECUTION.md`.

- Core `lower_rust` emits standalone safe Rust over checked operand-order slices,
  f32/f64 operation precision, affine layouts/regions, predicates, ordered effects
  and all reference reductions. `lower_rust_unrolled` preserves traversal while
  expanding at most 64 iterations under a source-size budget. Core dependencies
  are unchanged. See `RUST-BACKEND.md`.
- Core `interval::lower_intervals`: strict f64 Read/Write neg/add/sub/mul, correlated
  squares, explicit `Fma` (real `a·b+c`, W11 M4) and index-only branches; FP-environment
  probe, nonfinite widening, typed refusals. W11 M4: padding is the Rump–Zimmermann–Boldo–
  Melquiond successor bound `RN(RN(φ|v|)+η)`, φ = 2^-53(1+2^-52), η = 2^-1074 (proof in
  `crates/malleus-cranelift/INTERVALS.md`; each enclosure is inside the previous
  `|v|·2^-52 + MIN_POSITIVE` one). New `interval::{lower_centered_intervals,
  lower_affine_intervals}` (same endpoint layout, intersected with naive). Taylor models open.
- Validation: core 45 tests, including rustc-compiled looped/unrolled differential
  kernels, seeded randomized interval enclosure soundness against exact rationals
  (with an inward-shrink negative control) and a random interpreter-vs-Rust-source
  interval differential; isolated Cranelift 18 tests (adds a random interpreter-vs-JIT
  interval differential). `PreparedCall::run` re-derives its raw pointers each call.
  Lowered interval kernels' input contract (ordered non-NaN endpoints, caller checks
  `ieee_environment_supported()`) is in the `interval` module docs. Core test support
  lives in `tests/support/` (no dependency on `crates/`). Clippy/formatting pass.
- `crates/malleus-cranelift` (package `malleus-cranelift`, W11 M2; workspace member, not a
  core dependency; replaces `experiments/cranelift-compiler`, no facade): bitwise-equal JIT for
  every IR form at every level (callouts `malleus_unary`/`malleus_binary` -> `policy::{unary,
  binary}`; explicit `Fma` -> Cranelift `fma`; min/max reductions), one generated function per
  kernel with a native strided point loop (the M1 batch contract), positional
  `call`/`run_batch` + `JitState` (allocation-free, `CallError` refusals decided by core
  `PreparedKernel::check_bindings`), thread-safe `JitCache` keyed by digest + schedule + level +
  `BACKEND_VERSION` + `cpu_features()` + `BATCH_ABI`. cadabra3 names unchanged
  (`JitKernel::{compile, prepare, prepare_owned, run}`, `PreparedCall::{update_input, values,
  run}`, `fixtures::{buffers, polynomial_batch}`). Cranelift pins `=0.128.3`; root
  `Cargo.lock` now holds the workspace (core versions unchanged, additions only).
- CADabra consumes both backends for exact-checked geometric guards, discovered
  degree-mask specialization and full Boolean profiling. Geometry/event/topology
  authority remains in CADabra; exact coefficient algebra remains Resolvent-owned.

## Current role

Malleus owns backend-independent, finite-precision local kernel IR, structural
validation, scheduling contracts, reference execution, and the local AD products
over them. It does not own a scientific language, mesh topology, finite-element
spaces, global assembly, solver policy, coupled state, or simulation history. It
has no port, connector, or system vocabulary: a bound model input is an opaque
external operand, and composition binds buffers between verbatim kernels.

## Implemented

- `primal_output_reads_input(&StructuredKernel, OperandId) -> Result<bool, ValidationError>`:
  validates the full kernel and readable input, follows ordered local and operand dependencies
  to final writable outputs, ignores dead locals/overwritten stores, retains reduction history,
  and conservatively includes select predicates and both branches. No algebraic cancellation.
  Static affine addresses cannot read data. All writable buffers, including read-write buffers,
  are outputs; carried dependence is conservative across repeated iteration addresses. A false
  result proves no value dependency, not freedom from errors in eagerly executed dead expressions.
  Scientia consumes this owner proof for property argument admission; no frontend IR walker needed.

- One `malleus` crate. Runtime dependencies: `serde`, `serde_json`, `blake3`
  (the last two only for deterministic artifact digests; pinned in `Cargo.lock`).
- `StructuredKernel` and `StructuredModule` with fixed iteration domains,
  affine indexing maps, explicit buffer regions and dense layouts, ordered SSA-like locals, scalar
  expressions, predicates, stores, and reductions.
- Explicit numeric policies and backend-neutral JVP/VJP/Jacobian request types;
  schedule-independent structured forward and reverse AD emits new validated IR.
  `DerivativeProduct::primal_operands` (additive, `serde(default)`) pairs every readable primal
  operand with its operand in the derivative kernel so consumers bind primal values through an
  explicit table instead of assuming ids are preserved.
- Deterministic validation of operand names/maps, ranks, layout permutations,
  affine bounds, write effects/aliasing, axes, access modes, local definition
  order, derivative requests, and module kernel names.
- `KernelSchedule`, `Executable`, and `ExecutableModule::reference` as the
  backend boundary; deterministic sequential `Interpreter::run` with row/column-major layouts,
  canonical reductions, and declared f32/f64 operation precision.
- **Digests** (`digest.rs`): `Digest { algorithm, hex }` (blake3 over canonical serde JSON of a
  schema-bearing payload; same wire shape as the federation's `Digest`). `kernel_digest`
  (`malleus-kernel-digest/1`, pinned by a golden test) and `module_digest`
  (`malleus-module-digest/1`).
- **Kernel compositions** (`compose.rs`, ARCHITECTURE.md §6 `BoundChain::Composed`):
  `KernelComposition { name, stages: Vec<StructuredKernel>, shared_buffers: Vec<SharedBuffer> }`
  holds stage kernels verbatim; a `SharedBuffer { members: Vec<StageOperand> }` aliases one
  buffer across stages (one `Write` member or several same-op `Reduce` members, at least one
  `Read` member, writers strictly before readers, equal shape/layout/region; readers may repeat
  in one stage, writers may not). `validate_composition` → `ValidatedComposition` →
  `ExecutableComposition::reference`; `Interpreter::run_composition` binds unshared operands by
  `StageOperand` and, optionally, whole shared buffers by index (unbound shared buffers are
  zero-initialized scratch). `composition_digest` (`malleus-kernel-composition/1`) covers the
  stage digests and the sharing table only. `differentiate_composition` builds the JVP/VJP as a
  composition of per-stage `differentiate` products: reproduced primal producer stages, then
  derivative stages (forward order for JVP, reverse for VJP), with the producer tangent shared
  into each consumer direction (JVP) or each consumer cotangent reduced into the producer seed
  (VJP). Fan-out to several consumers accumulates. Structurally disconnected request operands
  are typed refusals (`IndependentUnreachable`/`DependentUnreachable`), as are shared operands
  in a request.
- **Facet-pair kernels** (`facet.rs`, ARCHITECTURE.md §4/§8, SV2-B5):
  `FacetPairKernel { kernel, roles: Vec<FacetOperandRole> }` with
  `FacetOperandRole::Cell { side: Minus|Plus, partner: Option<OperandId> } | Facet { parity:
  Even|Odd }`. The only convention Malleus fixes is `FACET_NORMAL_CONVENTION = "minus_to_plus"`:
  odd facet data (the normal, oriented fluxes) negates under side relabelling. A side-owned
  outward flux datum (the port unknown in the dual trace space) is a `Cell` operand of its
  owner; the balance row between the two owners is an `Even` facet output.
  `validate_facet_pair` checks role count, presence of both sides, symmetric partners with equal
  shape/layout/region/access, and no odd non-additive reduction outputs.
  `check_facet_swap_symmetry` proves swap covariance by executing the kernel with the sides
  exchanged and reports per-output deviations (a wrong parity declaration is detected, an
  unpaired cell operand is refused). `differentiate_facet_pair` carries roles onto derivative
  operands (partners survive only when both sides are requested). `facet_pair_digest`
  (`malleus-facet-pair-kernel/1`) covers the inner kernel digest, roles, and convention.
- A module-complete `run_local_differential_campaign` API (SV0-B2) comparing primal, JVP, VJP,
  and parameter-only JVP execution between the interpreter and a distinct version-identified
  `LocalExecutableRunner`, with centered differences, the adjoint identity, explicit tolerances,
  and retained numeric-policy mutation fixtures (`check_numeric_policy_mutation`).
- Complete Serde representations for modules, kernels, compositions, facet-pair kernels, and
  derivative products. Consumers re-run structural validation after decoding.

## Validation

Checks rerun successfully on 2026-10-19 (W11 M4 clone, `CARGO_PROFILE_TEST_OPT_LEVEL=2`):

- `cargo test -p malleus --locked`: 68 tests passed (1 unit; composition 9; digest_pins 1;
  facet_pair 6; interval_differential 1; interval_enclosures 5; interval_padding 2;
  interval_soundness 3; output_dependency 6; policy 6; prepared 5; prepared_alloc 1;
  rust_source 4; structured_kernel 11; sv0_campaign 7).
- `cargo test -p malleus-cranelift --locked`: 28 passed (3 unit; alloc 1; differential 13;
  interval 4; interval_random 2 (adds centered/affine JIT-vs-interpreter: 71 of 120 lowered
  programs compiled, the rest exceed the JIT budget); native 5).
- clippy `-D warnings`, rustdoc `-D warnings`, `cargo fmt -p <crate> -- --check` for both crates.
- Miri (`scripts/miri.sh`, nightly miri 2026-07-08, `MIRIFLAGS=-Zmiri-deterministic-floats`;
  run once, Stacked Borrows): core `--lib` 1/1, `structured_kernel` 11/11, `prepared`
  binding/module/composition tests 3/3, `policy` fma/serialization/validation 3/3;
  `malleus-cranelift --lib` 3/3 (PreparedCall pointer re-derivation regression ported from the
  cadabra3 toy, BoundBuffers refusals, JitState tables). No UB reported. Without the flag miri
  perturbs `fma` results by design (one bit test fails), so the flag is part of the command.
  Excluded for runtime (>280 s each under miri): `prepared_alloc`, `interval_soundness`. JIT
  integration tests are `#![cfg(not(miri))]`.
- `cargo clippy -p malleus --all-targets -- -D warnings`
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps`
- `cargo fmt --package malleus -- --check`; `git diff --check`
Proof tests for W7/W8 packages: `tests/{composition,facet_pair,output_dependency}.rs`.

## Prepared execution measurements (W11 M1)

`cargo run --release --example prepared_bench`, 2026-10-18, Intel i7-8850H (12 threads)
**under heavy contention (load average 64-76)**: absolute numbers are inflated several-fold
and noisy; ratios are indicative only. Median of seven batches, ~20 ms each, time per call;
"batched" = one `run_batch` over 64 strided independent points (10^5 for the property
kernel), per point. All four paths are compared bitwise before timing.

| Kernel | Interpreter | Prepared one-shot | Prepared reusable | Batched (per point) |
|---|---:|---:|---:|---:|
| One guard polynomial | 1.14 us | 1.04 us | 222.9 ns | 55.6 ns |
| 64 guard evaluations | 36.13 us | 19.50 us | 10.10 us | 1.70 us |
| 4,096 guard evaluations | 2.32 ms | 1.25 ms | 1.11 ms | 184.01 us |
| Quadratic JVP, 1,024 inputs | 1.51 ms | 176.60 us | 376.86 us | 37.84 us |
| Quadratic VJP, 1,024 inputs | 515.68 us | 149.08 us | 264.47 us | 99.11 us |
| Dense matvec 64x64 | 1.65 ms | 404.26 us | 1.06 ms | 216.30 us |
| Property kernel, 10^5 point calls | 1.50 us | 1.86 us | 614.0 ns | 74.3 ns |

The historical Cranelift table in `crates/malleus-cranelift/README.md` was measured on a
different, idle host (EPYC 9V74); do not compare across the two tables. Rerun on an idle
host before quoting speedups.

## W11 M2 backend measurements

`cargo run -p malleus-cranelift --release --locked --quiet > crates/malleus-cranelift/results-m2.json`,
2026-10-19, Intel i7-8850H, **contended host (load average 22-41)**: ratios indicative only.
ns per call, median of 7 batches; "/pt" = one call over 64 independent strided points, per
point. Every path is checked bitwise against the interpreter (batches against 64 sequential
interpreter calls) before timing. Batched Rust: `lower_rust_batched` width 8 via `rustc -O`.

| Kernel | Interpreter | M1 prepared | M1 batch/pt | JIT checked | JIT prepared | JIT positional | JIT batch/pt | Batched Rust/pt | JIT compile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guard-polynomial-1 | 919.5 | 265.4 | 31.4 | 270.2 | 42.0 | 126.8 | 16.7 | 2.3 | 1629 us |
| guard-polynomial-64 | 38228.3 | 10124.8 | 1496.0 | 837.4 | 621.9 | 653.4 | 629.0 | 66.0 | 1337 us |
| guard-polynomial-4096 | 2394836.0 | 576093.6 | 119849.6 | 38611.1 | 39118.2 | 35654.9 | 42631.5 | 13651.3 | 1285 us |
| quadratic-Jvp-1024 | 908558.5 | 246533.6 | 57154.2 | 10817.5 | 11248.5 | 11412.0 | 11550.4 | 2731.6 | 1525 us |
| quadratic-Vjp-1024 | 919162.8 | 225710.5 | 57699.3 | 12639.1 | 13417.0 | 13288.2 | 12258.7 | 17583.0 | 1539 us |
| matvec-64x64 | 1594166.4 | 316180.1 | 82548.6 | 28286.0 | 29649.8 | 29733.6 | 34679.4 | 5212.1 | 1514 us |

Positional JIT `call` costs ~85 ns over borrowed `PreparedCall::run` (1-point kernel: binding checks + pointer table); batching amortizes it.

## Contract shapes landed (for GX-CONTRACTS C12)

- `malleus::Digest { algorithm: "blake3", hex }`; schemas `malleus-kernel-digest/1`,
  `malleus-module-digest/1`, `malleus-kernel-composition/1`, `malleus-facet-pair-kernel/1`.
- `KernelComposition`, `SharedBuffer`, `StageOperand`, `CompositionTarget`,
  `CompositionDerivativeRequest`, `CompositionDerivativeProduct { primal_stages,
  derivative_stages: [{ primal_stage, stage, primal_operands }], independent_operands,
  dependent_operands }`, `CompositionError`, `CompositionExecutionError`,
  `CompositionDifferentiationError`.
- `FacetPairKernel`, `FacetOperandRole`, `FacetSide`, `SwapParity`, `FACET_NORMAL_CONVENTION`,
  `FacetPairDerivativeProduct`, `FacetSwapReport`, `FacetPairError`, `FacetSwapError`.
- `DerivativeProduct::primal_operands` (additive).
- W11 M1 (additive, no serialized form, module `malleus::prepared` only; new root-level
  names: only the module name `prepared`): `PreparedKernel { new, run, call, run_batch,
  read_operands, write_operands, required_len, state }`, `PreparedState`, `ReadBinding` /
  `WriteBinding { operand, values, stride }`, `BATCH_BLOCK = 32`, `CallError { ReadCount,
  WriteCount, ReadOperand, WriteOperand, ShortBuffer, ExtentOverflow }`, `PrepareError {
  AddressOutOfRange, ProgramTooLarge }`, `PreparedModule`, `PreparedComposition`,
  `CompositionState`, `PreparedRunner` (identity `malleus-prepared/bytecode-v1`).

## Deviations from `sinbad/ARCHITECTURE.md` and why

- §6 says the cross block is the product of two local point kernels evaluated by Finitum at
  quadrature points. Malleus makes that product a typed, digested object (`differentiate_composition`
  with the independent set restricted to the producer's inputs yields exactly the consumer input
  tangent composed with the producer output tangent) instead of a pair of closures, so the
  chain rule has an identity, a wire form, and an interpreter proof. Finitum may still evaluate
  the two kernels itself; the composition is the contract, not a mandate to change its loop.
- Binding is buffer-level between whole stage invocations, not per-quadrature-point value
  passing. This is what makes tensor-valued outputs, reduced outputs, and fan-out well-defined
  without touching kernel bodies; it also means a composition never crosses a mesh (that is
  `BoundChain::Transferred`, Finitum/Krasis).
- Trace classes (`Value`/`Normal`/`Tangential`/`FacetL2`, §3.4) are not recorded on facet-pair
  operands: they are function-space vocabulary owned by Scientia. Malleus records only side
  ownership, partners, and swap parity, which is all the kernel's covariance needs.

## Consumer surface changes required (recorded for the Scientia/Finitum lanes)

- Scientia: lower `output` declarations to point-kernel bundles; `SysBlock::Composed` builds a
  `KernelComposition` whose `composition_digest` joins the system artifact chain.
- Finitum: composed `bind_kernels`/`execute` binding `StageOperand`s and recording
  `composition_digest`; facet-pair traversal for minus/plus traces (SV2-B2).

## Current limits and next work

- Prepared execution (W11 M1): compositions have no batch entry point; `PreparedComposition`
  copies stage operand windows in and out like the interpreter (aliasing them is a future
  optimization that must keep the copy semantics observable-equal). The point program
  recomputes each affine address per access (no strength reduction yet). Measurements above
  were taken on a contended host.

- Request operands of a composition derivative must be exposed (unshared); tangents of an
  intermediate shared output are not exposed (a caller can still observe its primal value by
  binding the shared buffer).
- `run_local_differential_campaign` does not yet accept compositions or facet-pair kernels;
  the interpreter proofs above are direct tests, not campaign cases.
- No facet-pair composition wrapper: composing a producer output into a facet-pair kernel's
  side operand works at the plain-kernel level and loses no roles, but no helper builds it.
- Fusing a composition into one kernel (its own artifact identity) is not implemented.
- Vector/tensor gap (W7 package 3, not implemented): one kernel cannot hold outputs with
  different free-axis sets without an index-equality `Select` guard; Scientia uses one module
  per output.
- Interval upgrades (W11 M4): Taylor models are open; centered/affine admit only the naive
  op set; the centered form differentiates once per read operand (expanded JVPs can hit the
  typed lowering budget, <2% of random 4-input programs).
- Materialized Jacobians and differentiation through read-write operands remain refusals.
- Bounds and simple injective-write maps are proved conservatively; general affine
  injectivity remains future work before production parallel schedules.
