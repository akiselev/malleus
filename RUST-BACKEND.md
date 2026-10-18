# Rust source and bounded local kernels

Malleus now owns two source transformations in its dependency-neutral core:

- `lower_rust(&Executable, symbol)` emits standalone safe Rust for the existing
  validated finite-precision IR, using one slice argument per operand in operand
  order. Read operands receive `&[f64]`; writable operands receive `&mut [f64]`.
  Every required buffer length is checked before any effect. No runtime Malleus
  dependency, compiler subprocess, unsafe code, pointer ABI or dynamic library is
  needed to run the emitted function.
- `interval::lower_intervals(&ValidatedKernel)` lifts the real-arithmetic meaning
  of a supported strict f64 kernel into ordinary endpoint/error-bound Malleus IR.
  That IR can go to Rust source, the interpreter or the isolated Cranelift backend.
  The implementation moved out of the compiler experiment; consumers now import
  the core API directly. No compatibility facade remains.

`lower_rust_unrolled` optionally expands an entire local iteration domain, capped
at 64 iterations and two million emitted body bytes. It preserves the reference
canonical/schedule-defined traversal and every arithmetic operation. This lets a
normal Rust optimizer propagate fixed indices and remove unused branches of small
local batches without an algebraic reassociation pass.

## Execution contract

The generated function returns `Result<(), &'static str>` and retains the core's
f64 storage ABI. A kernel declaring f32 performs every operation in f32 and widens
only when storing. The emitter supports all reference unary/binary operations,
lazy selects, short-circuit predicates, ordered locals, signed affine maps, dense
layouts, regions, Write/ReadWrite and the four reference reductions. Reductions
retain caller-supplied initial output values. Constants use explicit IEEE bits.

Normal Rust floating-point semantics are required. Do not introduce fast-math,
reassociation or contraction overrides. The emitter never writes `mul_add`.
Platform transcendental behavior follows that host's Rust implementation, as does
the interpreter. NaN payload agreement is not promised. The numeric policy may
permit more optimization, but this backend still emits strict operations.

Tiles, explicit vector/parallel schedules and workloads beyond documented source,
index, buffer, recursion and iteration budgets are typed refusals. Independent
iterations execute sequentially; rustc may apply only its semantics-preserving
optimizations. Function symbols must be valid non-keyword ASCII identifiers.
Index arithmetic uses bounded i128 intermediates and checked slice access. Rust's
borrows enforce the same no-mutable-aliasing contract as ordinary buffer bindings.

The core only emits source. The consumer owns build/cache/target selection. A
build script can validate a kernel, emit Rust to OUT_DIR and include it normally;
the usual Cargo/rustc pipeline then compiles it with the rest of the application.
Changing bindings does not require recompilation unless a consumer deliberately
specializes the kernel and verifies the corresponding binding preconditions.

## Bounds and specialization

The interval pass retains the earlier outward padding argument: each finite
arithmetic endpoint v is expanded using `RN(abs(v)*EPSILON + MIN_POSITIVE)`;
nonfinite intermediates widen to the whole line. It requires round-to-nearest,
gradual underflow and no contraction/reassociation. The environment probe is in
the core API. These bounds are deliberately wider than nextafter interval arithmetic
near underflow, and overlap remains inconclusive. Neg/add/sub/mul, correlated
squares and index-only predicates are supported; data-dependent branching and other
operations still refuse. Callers supply valid endpoint enclosures of exact inputs.

CADabra's experiment discovers per-guard polynomial degrees with Resolvent exact
arithmetic, generates the corresponding Malleus index branches, lowers their
bounds and emits unrolled Rust. Immutable runtime bindings verify that the exact
degree mask matches the generated specialization. Candidates include constant,
affine, expanded quadratic, Horner quadratic and discovered masks. Exact whole-
interval and admission checks precede empirical timing selection. Selection is
only among tested candidates on the measured host, not proof of global optimality.
CADabra owns those geometric expressions, parameter boxes and event policies.
Resolvent owns their exact reference mathematics. Malleus owns execution/lowering.

## Validation

The core test suite has 41 tests. The source differential test emits and compiles
42 kernels with rustc -O (looped and unrolled variants), executes them, and compares
all buffers against the interpreter. It includes primal/JVP/VJP, f32/f64, reductions,
lazy branches, IEEE inputs, an adversarial FMA case, dense matvec, reversed indexing
and nonzero regions. A short input is rejected before output mutation. Another
test rejects symbol injection, keywords and unsupported scheduling. The isolated
Cranelift suite retains 17 tests using the promoted interval API.

The CADabra Rust/source integration adds exact extrema, generated parameter JVPs,
actual Boolean observations, budgeted event searches and parameter-box subdivision.
Its `RUST-LOWERING.md` records the generated/handwritten/JIT comparison and full
Boolean stage profile. Full application topology certification remains a consumer
obligation; a local arithmetic enclosure alone does not establish it.
