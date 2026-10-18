# Explicit interval lowering

`interval::lower_intervals` transforms a validated strict f64 Malleus kernel into
another validated Malleus kernel. It introduces endpoint operands and ordinary
arithmetic, comparisons, selects and locals. Both the reference interpreter and
Cranelift run that exact IR. No new numeric IR, core dependency, external interval
runtime, libm call, or special Cranelift instruction is introduced.

For source operand i, endpoint operands are 2i and 2i+1 with identical indexing,
layout and region. Callers provide ordered, non-NaN bounds enclosing exact inputs.
The current subset is Read/Write, finite constants, exactly representable iteration
indices (extents <= 1,000,000), ordered locals, negation, addition, subtraction and
multiplication. Structural squares retain correlation across zero. Unsupported
precision, effects, expressions or size/depth budgets refuse. This pass encloses
the source's **real-arithmetic expression**, rather than asserting its ordinary
floating-point output is exact.

## Roundoff argument

Execution requires IEEE binary64 round-to-nearest/ties-even, gradual underflow,
and forbidden FMA/reassociation. `ieee_environment_supported` tests the host's
rounding behavior and subnormal input/output operations with runtime operands.
Consumers must keep that environment unchanged during prepared execution.

For a rounded, finite arithmetic endpoint v, emit:

```
r = RN(abs(v) * 2^-52 + MIN_POSITIVE)
lo = RN(v - r)
hi = RN(v + r)
```

For normal v whose scaled value remains normal, multiplication by 2^-52 is an
exact exponent shift: the result is at least one ulp of v. Adding a positive
normal floor and rounding cannot reduce it below that representable value.
If the scale underflows, MIN_POSITIVE dominates the subnormal rounding error.
Thus r exceeds the rounding error of the original add/subtract/multiply; rounding
v±r still reaches beyond the adjacent rounding cell. Overflow of the padded
endpoint safely gives an unbounded side. If the original rounded endpoint is
nonfinite (including 0*infinity), both sides widen to [-infinity,+infinity].
Nonfinite values are generated from finite MAX*2 expressions because core Malleus
rightly rejects nonfinite literal constants. Ordered comparisons on abs(v) reject
NaNs as well as infinities before choosing a finite bound.

Addition/subtraction use monotone endpoint combinations. Multiplication takes the
minimum/maximum of four outward-enclosed products. Squares use the nonnegative
zero lower bound when the interval crosses zero and the appropriate squared
endpoint otherwise. Negation is exact. Composition therefore encloses the real
expression provided its input intervals enclose the data. Interval dependency
can widen a result; overlap with zero is inconclusive, never an optimistic sign.

The normal floor deliberately sacrifices tightness near underflow. It is wider
than Resolvent's nextafter arithmetic, and a consumer should use its existing exact
fallback when this widening blocks a decision. This is an engineering proof
argument and differential validation, not a machine-checked proof of the compiler.

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
