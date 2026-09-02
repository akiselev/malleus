# Malleus

Malleus is the structured compiler for finite-precision local numerical kernels.
It owns the program shape between a scientific frontend and concrete CPU/GPU
backends:

```text
StructuredModule -> validation -> ExecutableModule -> backend
                                      |
                                      +-> deterministic Interpreter
```

The IR describes fixed local iteration domains, affine operand indexing, dense
layouts, scalar expressions, reductions, numeric policy, and derivative
requests. Scheduling is separate from the mathematical kernel. The reference
executable uses canonical reduction order and the interpreter provides a small,
deterministic f32/f64 correctness oracle.

All schedule-independent IR and derivative-product types have a Serde wire form. FC11 archives
round-trip complete `StructuredModule` values and always pass decoded modules back through
`validate_module` before execution; schedules and executable containers remain rebuilt data.

Two structured objects sit beside single kernels without changing them. A
`KernelComposition` binds one kernel's output buffer to another kernel's input
operand at execution time — the kernel-level form of a bound model input — and
its JVP/VJP are compositions of the per-kernel derivative products. A
`FacetPairKernel` gives a point kernel evaluated on an interior or interface
facet a side role (minus cell, plus cell, or facet-native with a swap parity)
for every operand; swap covariance is proven by execution, not declared.

Malleus deliberately does not own equations, meshes, basis traversal, global
assembly, nonlinear solvers, time integration, simulation state, ports, or
system vocabulary. It has no dependencies on the rest of the Sinbad ecosystem.

## Public boundary

- `StructuredKernel` and `StructuredModule` are the frontend-owned inputs.
- their complete typed contents serialize for identity-sensitive artifact archives;
- `validate` and `validate_module` establish structural, bounds, layout, and
  effect invariants.
- `differentiate` constructs schedule-independent structured JVP and VJP
  products with explicit primal/derivative operand pairs.
- `Executable` and `ExecutableModule` pair validated kernels with schedules.
- `ExecutableModule::reference` constructs canonical reference executables.
- `Interpreter::run` executes one kernel against explicit buffer bindings.
- `run_local_differential_campaign` checks every kernel in a module with caller-supplied local
  buffers: primal and generated JVP/VJP/parameter-JVP products are compared between the reference
  interpreter and a distinct, version-identified `LocalExecutableRunner`; the reference
  interpreter is refused as its own candidate. Centered differences and the JVP/VJP adjoint
  identity provide independent local differential checks.
- `check_numeric_policy_mutation` retains explicit f64-to-f32, f32-to-f64, and reduction-order
  mutations and reports detection only when the mutated local result leaves the declared
  tolerance.
- `DerivativeRequest` selects independent and dependent operands; JVP and VJP
  are implemented as IR-to-IR passes, while materialized Jacobians are an
  explicit unsupported mode. `DerivativeProduct::primal_operands` maps every
  readable primal operand to its operand in the derivative kernel.
- `kernel_digest`, `module_digest`, `composition_digest`, and `facet_pair_digest`
  are deterministic blake3 identities over schema-bearing canonical encodings;
  a composition or facet-pair digest embeds its kernels by digest, so wrapping
  never changes what a kernel is.
- `KernelComposition` / `validate_composition` / `ExecutableComposition` /
  `Interpreter::run_composition` / `differentiate_composition` compose verbatim
  kernels over `SharedBuffer` groups (writers before readers; fan-out and
  reduce fan-in allowed) and derive them without fusing anything.
- `FacetPairKernel` / `validate_facet_pair` / `differentiate_facet_pair` /
  `check_facet_swap_symmetry` carry minus/plus/facet roles with partners and
  swap parity; the only fixed convention is that odd facet data is oriented
  from the minus cell to the plus cell.

Campaign cases must cover every module kernel exactly once and bind one finite buffer per operand,
disjoint state/parameter directions, seeds for every writable dependent, a positive
centered-difference step, and explicit componentwise absolute-or-relative tolerances. Backend
refusal is distinct from a completed comparison that fails. The campaign interface is deliberately
in-process and pointwise; it is not a mesh, assembly, solver, history, scientific-source, or
external-process protocol.

## License

MIT OR Apache-2.0
