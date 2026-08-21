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

Malleus deliberately does not own equations, meshes, basis traversal, global
assembly, nonlinear solvers, time integration, or simulation state. It has no
dependencies on the rest of the Sinbad ecosystem.

## Public boundary

- `StructuredKernel` and `StructuredModule` are the frontend-owned inputs.
- `validate` and `validate_module` establish structural, bounds, layout, and
  effect invariants.
- `differentiate` constructs schedule-independent structured JVP and VJP
  products with explicit primal/derivative operand pairs.
- `Executable` and `ExecutableModule` pair validated kernels with schedules.
- `ExecutableModule::reference` constructs canonical reference executables.
- `Interpreter::run` executes one kernel against explicit buffer bindings.
- `DerivativeRequest` selects independent and dependent operands; JVP and VJP
  are implemented as IR-to-IR passes, while materialized Jacobians are an
  explicit unsupported mode.

## License

MIT OR Apache-2.0
