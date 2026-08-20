# Malleus

Malleus is the structured compiler for finite-precision local numerical kernels.
It owns the program shape between a scientific frontend and concrete CPU/GPU
backends:

```text
StructuredModule -> validation -> ExecutableModule -> backend
                                      |
                                      +-> deterministic Interpreter
```

The IR describes fixed local iteration domains, affine operand indexing, scalar
expressions, reductions, numeric policy, and derivative requests. Scheduling is
separate from the mathematical kernel. The reference executable uses canonical
loop order and the interpreter provides a small, deterministic correctness
oracle.

Malleus deliberately does not own equations, meshes, basis traversal, global
assembly, nonlinear solvers, time integration, or simulation state. It has no
dependencies on the rest of the Sinbad ecosystem.

## Public boundary

- `StructuredKernel` and `StructuredModule` are the frontend-owned inputs.
- `validate` and `validate_module` establish structural invariants.
- `Executable` and `ExecutableModule` pair validated kernels with schedules.
- `ExecutableModule::reference` constructs canonical reference executables.
- `Interpreter::run` executes one kernel against explicit buffer bindings.
- `DerivativeRequest` names JVP, VJP, and Jacobian variants without prescribing
  an automatic-differentiation implementation.

## License

MIT OR Apache-2.0
