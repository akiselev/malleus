#!/bin/sh
# Miri (Stacked Borrows, the default) over the pointer-, binding- and interpreter-level tests.
# JIT-executing integration tests of malleus-cranelift are `#![cfg(not(miri))]`; its unit
# tests cover the PreparedCall pointer table (the cached-pointer regression) and JitState.
# Slow: run once per change to unsafe/binding code, with CARGO_BUILD_JOBS=4.
set -eu
# Miri deliberately perturbs float results (even fma) unless told not to; the tests assert bits.
export MIRIFLAGS="${MIRIFLAGS:--Zmiri-deterministic-floats}"
run() { echo "== cargo +nightly miri test $*"; cargo +nightly miri test --offline "$@"; }
run -p malleus --lib
run -p malleus --test structured_kernel
run -p malleus --test prepared -- malformed_bindings every_kernel_of_a_module prepared_compositions
run -p malleus --test policy -- explicit_fma default_level validation_refuses
run -p malleus --test export -- symbolic_schema fpcore_goldens oversized
run -p malleus-cranelift --lib
