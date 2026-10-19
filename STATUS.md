# Malleus status

Updated: 2026-10-18
Branch: `master`
Milestone: W11 M1 prepared kernels and batched execution (`EXECUTION.md`); strict Rust
source/unrolled and core interval lowering; W8/W7/SV0-B2 contracts retained

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
- Explicit outward `interval::lower_intervals` moved from the compiler experiment
  into the core. Strict f64 Read/Write neg/add/sub/mul, correlated squares and
  index-only branches; FP-environment probe, conservative nonfinite widening and
  typed capability/budget refusals. No alternate numeric IR or compatibility facade.
- Validation: core 45 tests, including rustc-compiled looped/unrolled differential
  kernels, seeded randomized interval enclosure soundness against exact rationals
  (with an inward-shrink negative control) and a random interpreter-vs-Rust-source
  interval differential; isolated Cranelift 18 tests (adds a random interpreter-vs-JIT
  interval differential). `PreparedCall::run` re-derives its raw pointers each call.
  Lowered interval kernels' input contract (ordered non-NaN endpoints, caller checks
  `ieee_environment_supported()`) is in the `interval` module docs. Core test support
  lives in `tests/support/` (no dependency on `experiments/`). Clippy/formatting pass.
- Isolated `experiments/cranelift-compiler` still lowers existing `Executable` and
  generated JVP/VJP IR; checked/borrowed calls, input updates and explicit refusals.
  Direct Cranelift codegen/frontend/module/jit/native pins remain `=0.128.3`, full
  transitive versions in its lockfile; native dependencies stay outside the core.
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

Core checks rerun successfully on 2026-10-18 (W11 M1 clone):

- `cargo test -p malleus --locked` (opt-level 2): 53 tests passed (1 unit; composition 9;
  digest_pins 1; facet_pair 6; interval_differential 1; interval_soundness 3;
  output_dependency 6; prepared 5; prepared_alloc 1; rust_source 2; structured_kernel 11;
  sv0_campaign 7).
- `cargo clippy -p malleus --all-targets -- -D warnings`
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps`
- `cargo fmt --package malleus -- --check`; `git diff --check`
- Focused `output_dependency` 6/6: unused/dead/overwritten data, local/operand transitivity,
  all outputs, no cancellation, predicate and branch conservatism, affine/reduction behavior,
  invalid locals and invalid/nonreadable queried operands. Queried read-write operands
  conservatively return true: partial stores and empty iteration domains can retain initial data.
  This corrects the initial whole-buffer overwrite assumption; both cases have regressions.

Proof tests for the W7 packages: `two_kernel_composition_evaluates_equal_to_the_hand_inlined_kernel`,
`jvp_through_the_composition_matches_the_inlined_jvp`,
`cross_block_jvp_is_the_chain_rule_of_producer_and_consumer_tangents`,
`vjp_through_the_composition_satisfies_the_adjoint_identity` (1e-12),
`fan_out_to_two_consumers_accumulates_the_producer_cotangent`,
`component_kernel_digests_are_unchanged_by_composition_and_differentiation`,
`swap_covariance_is_proven_by_execution_and_a_wrong_parity_is_detected`,
`jvp_and_vjp_of_the_facet_pair_satisfy_the_adjoint_identity_and_keep_roles`,
`kernel_and_module_digests_are_deterministic_and_pinned`.

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

The Cranelift table in `experiments/cranelift-compiler/README.md` was measured on a
different, idle host (EPYC 9V74); do not compare across the two tables. Rerun on an idle
host before quoting speedups.

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

- Scientia: lower `output` declarations to point-kernel bundles over the model's QFunction
  inputs (today only form integrals and property definitions lower); a `SysBlock::Composed`
  needs the producer output bundle and the consumer slot operand to build a `KernelComposition`,
  whose `composition_digest` belongs in the system artifact chain.
- Finitum: `bind_kernels`/`execute` gain a composed variant that binds `StageOperand`s from
  the two instances' point evaluations (values by `derivative_stages[..].primal_operands`,
  directions/seeds by the request tables) and records `composition_digest` in its receipt;
  facet-pair kernels need the interior/interface facet traversal to gather minus/plus traces and
  the minus-to-plus normal (SV2-B2 pull-forward).

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
- The reference interpreter copies stage buffers in and out of a composition; a backend may
  alias them.
- Vector/tensor gap observed while building the facet fixture (W7 package 3, not implemented):
  one kernel cannot hold outputs with different free-axis sets (a scalar jump beside a
  reduced normal flux) without an index-equality `Select` guard, because a scalar `Write` under
  a reduction domain is an aliasing write. Scientia sidesteps it with one module per output;
  Stokes/elasticity composition revealed no further gap, so nothing else was added.
- Materialized Jacobians and differentiation through read-write operands remain refusals.
- Bounds and simple injective-write maps are proved conservatively; general affine
  injectivity remains future work before production parallel schedules.
