# Malleus status

Updated: 2026-10-19
Branch: `master`
Milestone: W11 M3 nonsmooth policy, Jacobian mode, exact post-AD simplification
(`malleus::{ad, simplify}`); W11 M4 interval upgrades (successor-bound padding, centered and affine
enclosures) and FPCore/symbolic export; W11 M2 numeric policy levels, explicit `Fma`, portable transcendentals; W11 M1
prepared kernels and batched execution (`EXECUTION.md`); strict Rust
source/unrolled and core interval lowering; W8/W7/SV0-B2 contracts retained

## W11 M3 derivatives (modules `malleus::ad`, `malleus::simplify`)

- **Behaviour fix (results were wrong before):** JVP tangents were looked up by *primal*
  operand id on expressions already remapped to derivative-kernel ids, so a kernel declaring a
  writable operand before a readable one got a wrong or silently zero tangent. Now keyed by the
  remapped id. Kernels whose readable operands all precede writable ones are bitwise
  unchanged (all fixtures, digest pins). `derivative_options::jvp_reads_tangents_of_operands_
  declared_after_outputs` fails on the old code (0 instead of 3).
- `ad::{DerivativeOptions { nonsmooth: NonsmoothPolicy { Conventional (default) | Refuse },
  simplification: Simplification { None (default) | Exact } }, differentiate_with,
  differentiate_jacobian, kink_sites, KinkSite { statement, kind: KinkKind { Abs, Floor, Ceil,
  Min, Max, MinReduction, MaxReduction } }, JacobianProduct { product, columns:
  [JacobianColumns { primal, offset, len }], seed_axis, width }, MAX_JACOBIAN_WIDTH = 2^24}`.
  `DerivativeRequest` is unchanged (a new field would break consumer struct literals); options
  travel beside it and serialize as `{}` at the default. New `DifferentiationError::
  {NonsmoothRefused(Vec<KinkSite>), JacobianTooWide { columns }}`.
- Kink sites (user decision D6): abs/min/max whose differentiated argument carries a
  structural tangent from a requested independent; floor/ceil whose argument value varies with
  one; stores into min/max reductions; only statements reaching a requested dependent; select
  is not a kink (conditions are never differentiated). `Conventional` keeps d|x|(0) = +dx,
  min/max tie -> lhs tangent, floor/ceil -> 0; `Refuse` names every site.
- `DerivativeMode::Jacobian` is admitted by `differentiate` (was `UnsupportedMode`;
  compositions still refuse it): vector forward mode over a trailing `Parallel` seed axis of
  extent `width`; the identity seed is `select(column(p) == s, 1, 0)` with an exactly formed
  column index; output `d_y` has shape `S ++ [width]` (row-major extension, region offset 0,
  same access). Bitwise equal to per-seed JVPs; runs through `PreparedKernel::run_batch`.
- `simplify::simplify_exact(&kernel) -> Simplified { kernel, rewrites }`: `x*1`, `1*x`,
  `x-(+0)`, `x+(-0)`, `(-0)+x`, `--x`, CSE (bit-keyed hash-consing into new lets; skipped with
  read-write operands), dead `Write` stores overwritten later, `-0` stores into `Reduce(Add)`
  and `1` into `Reduce(Multiply)`. Never `x+0`, `x*0`, `+0` reduce stores, reassociation.
  Opt-in only (`Simplification::Exact`): default derivative kernels and digests are unchanged.
- Proof: `tests/derivative_options.rs` (7): default options == `differentiate` on every
  fixture family; refusal sites; Jacobian == identity-seed JVPs on fixtures + 40 random
  kernels (and batched); simplified == unsimplified bitwise on fixtures + 300 random kernels
  (abs/min/max/sqrt/div/fma/select/floor, f32 and f64) with NaN/inf/+-0 inputs, for primal,
  JVP, VJP and Jacobian kernels.

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
- W11 M4 export (`EXPORT.md`; new root names: modules `symbolic`, `fpcore` only):
  `symbolic::{export_symbolic, SymbolicKernel { schema "malleus-symbolic/1", kernel,
  kernel_digest, precision, level, inputs, nodes, outputs }, SymbolicNode, ExactConstant,
  SymbolicInput, SymbolicOutput, SymbolicError, MAX_POINTS = 4096, MAX_NODES = 65536}` unrolls
  the domain into a hash-consed DAG (exact constants, `fma { single_rounding }`, lazy `select`,
  reductions as binary nodes); `fpcore::{export_fpcore, fpcore_from_symbolic, InputBox,
  FpcoreBenchmark, FpcoreError}` emits FPCore 2.0 per output element with `:pre` from boxes.
  `UnaryOp`/`BinaryOp`/`CompareOp`/`ScalarType` gained `Hash` (serde unchanged).
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
  conservative ordered dependency to final writable outputs (dead locals/overwritten stores
  ignored, both select branches included, no algebraic cancellation); a false result proves no
  value dependency. Scientia consumes it for property argument admission.

- Core runtime dependencies: `serde`, `serde_json`, `blake3`, `libm =0.2.16` (`Cargo.lock`).
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
  `StageOperand` and optionally whole shared buffers (unbound ones are zero scratch).
  `composition_digest` (`malleus-kernel-composition/1`) covers stage digests and sharing.
  `differentiate_composition` composes per-stage `differentiate` products (primal producers,
  then derivative stages; tangents shared forward, cotangents reduced backward; fan-out
  accumulates; disconnected or shared request operands are typed refusals).
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

Checks rerun on 2026-10-19 (W11 M3 clone, `CARGO_PROFILE_TEST_OPT_LEVEL=2`, `CARGO_BUILD_JOBS=4`):

- `cargo test -p malleus --locked`: 83 passed (1 unit; composition 9; derivative_options 7;
  digest_pins 1; export 8; facet_pair 6; interval_differential 1; interval_enclosures 5;
  interval_padding 2; interval_soundness 3; output_dependency 6; policy 6; prepared 5;
  prepared_alloc 1; rust_source 4; structured_kernel 11; sv0_campaign 7). `prepared_alloc`
  failed once under host contention (its global allocation counter is process-wide) and
  passed on 3 reruns; pre-existing, not touched by M3.
- `cargo test -p malleus-cranelift --locked`: 28 passed (3 unit; alloc 1; differential 13;
  interval 4; interval_random 2; native 5).
- clippy `-D warnings` (both crates, all targets), rustdoc `-D warnings`,
  `cargo fmt -p malleus -- --check`.
- Miri (`scripts/miri.sh`, `MIRIFLAGS=-Zmiri-deterministic-floats`, Stacked Borrows; last full
  run W11 M2/M4 2026-10-19, no UB): core `--lib`, `structured_kernel`, `prepared`/`policy`/
  `export` subsets, `malleus-cranelift --lib`. Excluded for runtime: `prepared_alloc`,
  `interval_soundness`; JIT integration tests are `#![cfg(not(miri))]`.

## Prepared execution measurements (W11 M1)

`cargo run --release --example prepared_bench`, 2026-10-18, i7-8850H under heavy contention
(load 64-76; ratios only). Per call, interpreter / prepared reusable / batched per point:
one guard polynomial 1.14 us / 222.9 ns / 55.6 ns; quadratic JVP 1,024 inputs 1.51 ms /
376.86 us / 37.84 us; matvec 64x64 1.65 ms / 1.06 ms / 216.30 us; property kernel 1.50 us /
614.0 ns / 74.3 ns. All paths compared bitwise before timing (full table in git history).

## W11 M2 backend measurements

`cargo run -p malleus-cranelift --release --locked --quiet > crates/malleus-cranelift/results-m2.json`
(2026-10-19, i7-8850H, contended, load 22-41: ratios indicative; full table in that file).
ns per point, interpreter / M1 batch / JIT batch / batched Rust: guard-polynomial-64
38228 / 1496 / 629 / 66; quadratic-Jvp-1024 908559 / 57154 / 11550 / 2732; matvec-64x64
1594166 / 82549 / 34679 / 5212. JIT compile 1.3-1.6 ms. Positional JIT `call` ~85 ns over
borrowed `PreparedCall::run`.

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

- §6 cross blocks are a typed, digested `differentiate_composition` product instead of a pair of
  closures; binding is buffer-level between whole stage invocations (tensor outputs, reductions
  and fan-out stay well-defined; a composition never crosses a mesh). Trace classes are Scientia
  vocabulary; facet-pair operands record only sides, partners and swap parity.

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
- Differentiation through read-write operands remains a refusal; compositions refuse
  Jacobian mode and take no `DerivativeOptions` yet.
- Bounds and simple injective-write maps are proved conservatively; general affine
  injectivity remains future work before production parallel schedules.
