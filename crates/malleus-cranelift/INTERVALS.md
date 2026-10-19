# Explicit interval lowering

`malleus::interval::lower_intervals` (promoted to the core) transforms a validated strict f64 Malleus kernel into
another validated Malleus kernel. It introduces endpoint operands and ordinary
arithmetic, comparisons, selects and locals. Both the reference interpreter and
Cranelift run that exact IR. No new numeric IR, core dependency, external interval
runtime, libm call, or special Cranelift instruction is introduced.

For source operand i, endpoint operands are 2i and 2i+1 with identical indexing,
layout and region. Callers provide ordered, non-NaN bounds enclosing exact inputs.
The current subset is Read/Write, finite constants, exactly representable iteration
indices (extents <= 1,000,000), ordered locals, negation, addition, subtraction,
multiplication and explicit `Fma`. Structural squares retain correlation across zero. Unsupported
precision, effects, expressions or size/depth budgets refuse. This pass encloses
the source's **real-arithmetic expression**, rather than asserting its ordinary
floating-point output is exact.

## Roundoff argument (successor bound, W11 M4)

Execution requires IEEE binary64 round-to-nearest/ties-even, gradual underflow,
and forbidden FMA/reassociation. `ieee_environment_supported` tests the host's
rounding behavior and subnormal input/output operations with runtime operands.
Consumers must keep that environment unchanged during prepared execution.

For a rounded, finite arithmetic endpoint v = RN(x) of one IEEE add/sub/mul, emit
(two separate RN operations, never fused):

```
e  = RN(RN(abs(v) * φ) + η)        φ = 2^-53 (1 + 2^-52) = u(1 + 2u),  η = 2^-1074
lo = RN(v - e)
hi = RN(v + e)
```

This is the bound of Rump, Zimmermann, Boldo and Melquiond, "Computing predecessor
and successor in rounding to nearest", BIT 49 (2009), doi 10.1007/s10543-009-0218-z.
Malleus uses only its outward half, `lo <= pred(v)` and `hi >= succ(v)`, proved
here directly for binary64 under the hypotheses above (RN ties-even, gradual
underflow, each written operation rounded once, no overflow in `φ·|v|`, which is
impossible):

- Exact x lies in the rounding cell of v, between the midpoints `(pred(v)+v)/2` and
  `(v+succ(v))/2`, so `[pred(v), succ(v)]` encloses x (`succ(MAX) = +inf`).
- v subnormal or zero: `RN(φ|v|) >= 0`, so `e >= RN(η) = η`; `v ± η` is
  representable and equals `succ/pred(v)`, so monotonicity of RN gives the claim.
- v normal, `|v| = M·2^E`, `2^52 <= M < 2^53`, gap above `2^E`, gap below `<= 2^E`.
  `φ|v| >= 2^(E-1) + 2^(E-53)`. If `E-53 >= -1074` that is a double, so
  `RN(φ|v|) >= 2^(E-1) + 2^(E-53)` and `e >= RN(φ|v|) > 2^(E-1)`. Otherwise
  `RN(φ|v|) >= 2^(E-1)` (a double) lies where the spacing is η, so
  `e = RN(φ|v|) + η > 2^(E-1)`. Either way `e` exceeds half of both gaps, so
  `v + e` lies strictly above the upper midpoint and `v - e` strictly below the
  lower one: `RN(v + e) >= succ(v)`, `RN(v - e) <= pred(v)`, with no tie.
- `v + e` overflowing gives `+inf`, still `>= succ(v)`.

The extra `2^-105` in φ is necessary: with φ = 2^-53 a power of two v gives
`e = 2^(E-1)` and `v + e` ties back to v. Since `φ < 2^-52` and `η < MIN_POSITIVE`,
monotonicity gives `e <=` the previous radius `RN(|v|·2^-52 + MIN_POSITIVE)`, so each
new enclosure is contained in the previous one. `tests/interval_padding.rs` checks
`lo <= pred(v)`, `hi >= succ(v)` and containment in the previous bound on every
power of two from 2^-1074 to 2^1023 with neighbours, both signs, plus 20,000
biased random doubles (36,471 of 36,790 are exactly `[pred(v), succ(v)]`; 26,724 are
strictly tighter than before), and exact dyadic containment of `x+y`, `x−y`, `x·y`
over all pairs of a 52-value near-underflow / near-overflow / exact-tie set
(`1 + 2^-53`, `MAX + 2^970`, `η·0.5`, ...).

If the original rounded endpoint is nonfinite (including 0*infinity), both sides
widen to [-infinity,+infinity]. Nonfinite values are generated from finite MAX*2
expressions because core Malleus rightly rejects nonfinite literal constants.
Ordered comparisons on abs(v) reject NaNs as well as infinities before choosing a
finite bound.

Addition/subtraction use monotone endpoint combinations. Multiplication takes the
minimum/maximum of four outward-enclosed products. Squares use the nonnegative
zero lower bound when the interval crosses zero and the appropriate squared
endpoint otherwise. Negation is exact. Explicit `Fma` encloses the real `a·b + c`
as an outward product followed by an outward sum. Composition therefore encloses the
real expression provided its input intervals enclose the data. Interval dependency
can widen a result; overlap with zero is inconclusive, never an optimistic sign.
This is an engineering proof argument and differential validation, not a
machine-checked proof of the compiler.

## Centered form and affine arithmetic (W11 M4)

Both reuse the endpoint operand layout and are ordinary validated IR (interpreter,
Rust source, Cranelift). Both intersect their result with the naive enclosure, so
"never wider than naive" is a theorem; the tests assert it.

- `lower_centered_intervals`: per read operand element a finite centre `c ∈ X`
  (rounded midpoint if inside, else a finite endpoint, else 0); `f(c)` lifted on
  point intervals; per read operand j the JVP with j alone independent, direction
  `[1, 1]`, primal loads bound to X, encloses `∂_j f(X)`; result
  `f(c) + Σ_j ∂_j f(X)·(X_j − c_j)` with outward `X_j − c_j`. Sound by the mean
  value theorem on `[c, x] ⊆ X` (the admitted ops are polynomial in the data;
  index-only branches are fixed per point).
- `lower_affine_intervals` (de Figueiredo–Stolfi, AF1-style): each value is
  `x0 + Σ_i x_i ε_i ± e` with one noise symbol per read operand (one element per
  iteration point) and one accumulated radius e. Input `[lo, hi]` becomes
  `c + r ε` with `r >= max(hi − c, c − lo)` rounded up. Every rounded centre and
  coefficient adds its successor-bound error to e; products add
  `|x0|e_y + |y0|e_x + R_x R_y` (`R = Σ|x_i| + e`, all sums/products rounded up by
  the outward bound); structural squares shift the centre by `h >= R²/2` and add h.
  The range `x0 ± (Σ|x_i| + e)` is rounded outwards. Any nonfinite or NaN
  coefficient or radius widens to the naive enclosure.
- Measured (`tests/interval_enclosures.rs`, 600 random programs each): mean
  width/naive width, centered 0.980 wide boxes / 0.949 narrow boxes; affine
  0.969 / 0.947. On `x(1−x)` over `[0.4, 0.6]` (exact `[0.24, 0.25]`): naive
  `[0.16, 0.36]`, centered `[0.23, 0.27]`, affine `[0.24, 0.26]`.
- Taylor models: not implemented (open). A sound first version needs a
  polynomial-coefficient representation with per-term outward rounding and a
  remainder interval per output; it was deferred to land centered + affine fully.

## Integration and validation

CADabra's optional `malleus-jit` first-Boolean experiment binds exact Resolvent
polynomial coefficients into endpoint buffers, lowers the primal and actual
Malleus-generated JVP, and checks whole-interval extrema using independent exact
rational endpoint/critical-point calculations. That integration, geometric event
search, topology policy and window tokens live in CADabra. Malleus owns only local
bounded execution. No topology-stability or complete-Boolean claim is made here.

The prototype has 17 tests: the prior 13 plus four for interval primal/JVP lowering,
interpreter/native agreement, overflow/indeterminate widening, capability refusals,
and reusable input updates. CADabra adds 18,800 whole-interval value/derivative
checks over seeded mixed polynomials and five scales, additional extreme-input
checks, sweep admission comparisons and event-search regressions. See CADabra's
`crates/zz-boolean-sensitivity-experiment/INTEGRATION.md` for results and reproduction.

`prepare_owned` retains descriptor ownership while borrowing all buffers and the
JIT code owner. `update_input` bounds-checks writes and admits only Read operands;
`values` exposes immutable inspection between calls. Code pointers remain private,
buffer lengths remain fixed, and caller writes never overlap an active invocation.
These APIs let a consumer prepare once and change parameters repeatedly without
allocation, including offline adversarial testing and parameter sweeps.
