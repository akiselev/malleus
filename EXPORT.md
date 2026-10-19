# Symbolic and FPCore export (W11 M4)

Both exports unroll a kernel's whole iteration domain in the reference interpreter's
traversal order into one hash-consed DAG (`malleus::symbolic::export_symbolic`), within
`MAX_POINTS = 4096` iteration points and `MAX_NODES = 65536` distinct nodes; beyond that
they refuse with `SymbolicError::{PointBudget, NodeBudget}`. Neither changes any IR,
serialized kernel or digest. Malleus depends on no consumer (Resolvent, FPBench).

## `malleus::symbolic` (schema `malleus-symbolic/1`, for Resolvent via Scientia)

```text
SymbolicKernel { schema: "malleus-symbolic/1", kernel: <name>, kernel_digest: Digest,
                 precision: "f64" | "f32", level: "strict_portable" | "strict_host" | "contracted",
                 inputs: [SymbolicInput { operand, element, name }],
                 nodes: [SymbolicNode],                       // topological: args < own index
                 outputs: [SymbolicOutput { operand, element, node, name }] }
SymbolicNode (serde tag "node"):
  input    { input }                    value of inputs[input]: buffer element before the call
  constant { bits, mantissa, exponent } exact mantissa·2^exponent (mantissa odd or 0); bits = binary64
  unary    { op, arg }                  op ∈ neg abs sqrt exp ln sin cos tan floor ceil
  binary   { op, lhs, rhs }             op ∈ add sub mul div pow min max atan2 (atan2(lhs, rhs) = lhs.atan2(rhs))
  fma      { lhs, rhs, addend, single_rounding: true }   real lhs·rhs + addend, rounded once
  bool { value } | compare { op, lhs, rhs } | not { arg } | and { lhs, rhs } | or { lhs, rhs }
           predicates; compare op ∈ eq not_eq less less_equal greater greater_equal (IEEE)
  select   { condition, if_true, if_false }  lazy piecewise value
```

- `element` is the index into the bound buffer (region offset included). Inputs of `reduce` and
  `read_write` operands are the initial buffer values; reads after a write see the written node.
- Iteration coordinates are constants after unrolling. Reductions are `binary` add/mul/min/max
  nodes of (previous, stored) in traversal order. `outputs` lists every element stored at least
  once, ascending by operand then element.
- Real semantics: every node is the real operation (partial where undefined, `select` evaluates
  only its chosen branch). Floating semantics: each `unary`/`binary`/`fma` node is one rounding to
  `precision` (binary32 kernels also round inputs and constants to binary32, recorded in `bits`)
  under the `level`'s math library (`malleus::policy`).
- `SymbolicKernel::check` validates schema, argument order/kind, inputs and outputs; consumers
  run it after decoding. Wire form pinned in `tests/export.rs::symbolic_schema_is_pinned`, e.g.
  `{"node":"constant","bits":4604930618986332160,"mantissa":3,"exponent":-2}` is 0.75.

## `malleus::fpcore` (FPCore 2.0)

`export_fpcore(&Executable, &[InputBox]) -> Vec<FpcoreBenchmark { operand, element, source }>`
(and `fpcore_from_symbolic`) emits one benchmark per stored output element:

```
(FPCore (c0_0 t_0 c1_0 c2_0)
  :name "polynomial-1 out[0]"
  :precision binary64
  :pre (and (<= 0x1p-1 c0_0) (<= -1 t_0 1))
  :malleus-kernel-digest "blake3:<hex>"
  :malleus-level "StrictHost"
  (+ c0_0 (* t_0 (+ c1_0 (* c2_0 t_0)))))
```

- Arguments are `<operand>_<element>` (sanitized to `[A-Za-z0-9_]`, `_o<operand>` added on name
  clashes); nodes used more than once become `let*` bindings `_t<node>`.
- `InputBox { operand, element: Option<usize>, lo, hi }` (element-specific boxes win; infinite
  bounds leave a side open; NaN or `lo > hi` refuse with `FpcoreError::InvalidBox`).
- Operators: `- fabs sqrt exp log sin cos tan floor ceil + - * / pow fmin fmax atan2 fma`,
  comparisons, `and or not`, `if`. Constants are exact (integers or `0x<m>p<e>`, `(- 0)`).
- FPCore operators are correctly rounded: exact for the IEEE-exact operations; transcendentals
  describe the correctly rounded function, not the level's libm (`:malleus-level` records it).
- Validation: goldens and a parser-free structural check (balanced lists, every head an FPCore
  operator, every symbol bound) over all fixtures and 100 random kernels. No FPBench, Racket or
  Herbie is installed on the build host (checked 2026-10-19), so no external tool is run.
