# Prepared and batched execution

`Interpreter` stays the reference oracle. `PreparedKernel` is a second execution path that is
bitwise identical to it (NaNs equivalent) and is built for call-heavy consumers such as one
call per quadrature point. Neither path changes IR, `kernel_digest` or serialized bytes.

## Preparation

`PreparedKernel::new(&Executable)` (and `PreparedModule::new(&ExecutableModule)`):

- assigns every operand a binding slot: `Read` operands are *read slots*, `Write`,
  `ReadWrite` and `Reduce` operands are *writable slots*, each in operand-id order
  (`read_operands()`, `write_operands()`);
- folds indexing map, dense layout and buffer region into one affine address
  `base + sum(stride[axis] * coordinate[axis])` and proves, over the whole iteration domain,
  that it stays inside `[region.offset, region.offset + region.length)`. Validated kernels
  always pass; a failure is the typed `PrepareError::AddressOutOfRange`;
- compiles the ordered statements into register bytecode twice: a *point program* whose
  selects and `and`/`or` predicates jump (the interpreter's lazy evaluation, literally), and
  a *columnar program* that evaluates both branches and blends per lane.

Every arithmetic instruction calls the interpreter's own scalar operation functions with the
kernel's scalar type, so `f32` kernels round each operation (and each constant, index and
load) to `f32` and widen on store, exactly as the interpreter does. No fused or reassociated
operation is ever emitted. Iteration uses the canonical axis order unless the numeric policy
is `ReductionOrder::ScheduleDefined`, in which case the schedule's loop order is used.

**Why execution cannot fail.** With addresses proven in range, expression evaluation is pure
and total: a load has no effect and cannot fail, and IEEE operations do not trap. All errors are
therefore binding errors, checked before the first effect, and evaluating an untaken branch
(columnar blending) cannot change any result.

## Calls

| Entry point | Binding form | Allocation |
|---|---|---|
| `run(&mut [BufferBinding])` | the interpreter's: any order, ids checked | allocates tables and scratch |
| `call(&mut PreparedState, &[ReadBinding], &mut [WriteBinding])` | positional | none once the state is sized |
| `run_batch(&mut PreparedState, points, &[ReadBinding], &mut [WriteBinding])` | positional + strides | none once the state is sized |

`run` accepts and refuses exactly the bindings `Interpreter::run` does, with the same
`ExecutionError` values. `call` is `run_batch` with one point.

Positional bindings: `reads[i]` must name `read_operands()[i]` and `writes[i]` must name
`write_operands()[i]`. Refusals (`CallError`): `ReadCount`, `WriteCount`, `ReadOperand`,
`WriteOperand` (wrong identity at a position; duplicates and omissions are caught here),
`ShortBuffer` and `ExtentOverflow`. Read buffers are `&[f64]` and writable buffers
`&mut [f64]`, so a read can never alias a write.

`PreparedState` owns the register file, predicate flags and coordinates. `kernel.state()`
returns one sized for every path of that kernel; a single state may also serve many kernels
and grows only when a call needs more scratch. `tests/prepared_alloc.rs` proves zero
allocations per call with a counting global allocator.

## Batches

Each binding carries a `stride`: the element offset between consecutive points (0 = one
buffer shared by every point). A binding must hold `(points - 1) * stride + offset + length`
elements; `points = 0` checks identities only and executes nothing.

**Contract:** `run_batch(points)` is exactly `points` sequential calls in point order, where
call `p` sees every buffer shifted by `p * stride`. This includes shared (stride 0) and
overlapping writable windows: later points observe, and reductions accumulate onto, earlier
points' effects in point order. No aliasing pattern is refused, because every pattern runs
under this definition.

**Execution:** if `points > 1` and every writable binding's stride is at least its operand's
`region.length` (written windows of distinct points are disjoint), points run columnar in
blocks of `BATCH_BLOCK = 32` lanes: each instruction is dispatched once per block and
iteration point and applied to all lanes; loads and stores touch only live lanes. Lanes are
then independent (reads are immutable during the call, written windows are disjoint), so
columnar order equals point order. Otherwise points run one after another through the point
program. Errors are binding errors reported before any effect.

The batch ABI (`values`, `stride` per slot, positional read/writable slots) is what a native
batched backend should accept so that it can replace `run_batch` without reshaping callers.

## Compositions

`PreparedComposition::new(&ExecutableComposition)` prepares every stage. `run(&mut
CompositionState, &mut [CompositionBinding])` has `Interpreter::run_composition`'s binding
contract and errors, and its copy-in/copy-out stage semantics (each stage runs on private
copies of its operand windows; writable copies are written back in operand order; unbound
shared buffers are zeroed scratch). The copies, scratch and binding tables live in the
reusable `CompositionState`, so calls do not allocate. Compositions have no batch entry point
yet.

## Evidence

- `tests/prepared.rs`: the Rust-source/Cranelift fixture families (primal, JVP, VJP, f32/f64,
  all four reductions, lazy selects over IEEE specials, signed zeros, dense matvec, reversed
  indexing with nonzero regions, empty domain) plus 400 random kernels over every unary,
  binary, comparison and predicate form, all access modes, column-major layouts, rank 0-2 and
  schedule-defined loop orders, with NaN/infinity/signed-zero/subnormal inputs. One-shot,
  presized and shared-state calls; batches of 0, 1, 2, 31, 32, 33 and 70 points with disjoint,
  shared and overlapping writable strides against a sequential-interpreter oracle; every
  binding refusal with no effect; compositions (primal, JVP, VJP, all binding errors).
- `tests/sv0_campaign.rs`: the seven-check local campaign through `PreparedRunner` and
  through a three-point columnar batch runner, with zero interpreter-vs-executable error.
- `tests/digest_pins.rs`: kernel and module digests pinned from the base commit.
- `examples/prepared_bench.rs`: `cargo run --release --example prepared_bench`.
