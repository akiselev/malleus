# Verification of IR-to-IR passes and schedule primitives (W11 M3)

The reference interpreter is the oracle. Every pass below is checked two ways: bitwise
differential execution (NaNs equivalent) on fixtures and random kernels, and, for body
rewrites, SMT translation validation per kernel.

## SMT translation validation (`malleus::smt`)

- `equivalence_query(source, semantics, target)` emits SMT-LIB 2 (`set-logic ALL`,
  FloatingPoint theory) for two kernels with identical operands, maps, domain and scalar type.
  One iteration point is encoded: each loaded operand and each index is a free variable, each
  writable operand a free initial value updated in statement order (`Write` replaces,
  `Reduce` folds with `fp.add`/`fp.mul RNE` or the uninterpreted `min`/`max`). Basic
  operations, `sqrt`, `abs`, `neg`, `fma`, `floor` (RTN) and `ceil` (RTP) are IEEE-exact in
  `Float64`/`Float32`; library operations are uninterpreted functions per op, type and math
  library. Read-write operands are refused.
- Equivalence notion: final states compared with SMT-LIB `=`, which distinguishes `+0`/`-0`
  and identifies all NaNs (the theory has one NaN). `unsat` of the disjunction of differences
  therefore means bit-equal or both NaN for every input — the differential tests' notion.
  `fp.eq` is not used (it identifies signed zeros).
- `check(z3, query, timeout)` runs the solver as an external process with soft (`-t`) and
  hard (`-T`) limits. `unsat` -> `Validated`; `sat` -> `Counterexample { model }`; anything
  else (`unknown`, `timeout`) -> `NotValidated { reason }`, never success. A missing solver is
  `SmtError::Solver`. Tests use `/usr/bin/z3` (Z3 4.16.0) and fail with a clear message if it
  is absent.
- Validated passes: (a) `simplify::simplify_exact` on post-AD kernels (JVP/VJP/Jacobian of the
  fixture families) and on random kernels with IEEE-special-prone operations; (b) the L2
  `policy::contract_fma` pass against the *declared fused semantics* of its input
  (`Semantics::DeclaredContraction`: at each fixed site the product and sum round once), not
  against the unfused kernel — which the same query proves different (counterexample).
  (c) Prepared/batched lowering is not IR-to-IR and is covered by differential tests only.
- Negative controls: `x + 0 -> x`, `x * 0 -> 0`, reassociation, and dropping a `+0` store into
  an additive reduction all yield counterexamples. Counterexample searches use `Float32`
  fixtures (bit-blasting `Float64` multiplication can exceed the timeout; such a query is
  reported `NotValidated`, and no test asserts success on one).
- Not done: validating the exported symbolic DAG against the kernel body.

## Schedule primitives (`malleus::schedule`)

Legality condition for an axis `a`: for every writable operand, iterations with different
`a` values never address the same element (`prove_write_injective`, exact integer decision on
`Σ g_b δ_b = 0` with a witness pair on failure; `NotProvable` past `INJECTIVITY_BUDGET`).
`VectorizedKernel::new` additionally refuses kernels that read `a` as a value, whose offsets
decrease along `a`, whose writable operands do not carry `a` alone in their most-major
dimension, or that use schedule-defined reduction order. `run` executes the lanes through
`PreparedKernel::run_batch`; `run_parallel` uses `std::thread::scope` on private buffer copies
and copies back each lane range's own elements (no reduction is shared across threads).
`tests/schedule.rs` compares both against `Interpreter::run` bitwise for 1, 2, 3 and 8
threads on fixtures, JVP/VJP kernels and 40 random kernels.

## Differential evidence for the derivative passes

`tests/derivative_options.rs`: default options reproduce `differentiate` exactly; Jacobian
columns equal identity-seed JVPs bitwise (and through `run_batch`); simplified kernels equal
unsimplified ones bitwise on fixtures and 300 random kernels with NaN/inf/±0 inputs.
