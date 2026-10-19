//! Kernel identities pinned before the prepared/batched execution work landed.

mod support;

use malleus::prepared::*;
use malleus::*;
use support::fixtures;

fn pinned_kernels() -> Vec<StructuredKernel> {
    let mut kernels = vec![
        fixtures::polynomial(8),
        fixtures::quadratic(1024),
        fixtures::matvec(64),
        fixtures::polynomial_batch(8),
    ];
    let mut f32_kernel = fixtures::polynomial(8);
    f32_kernel.numeric_policy.scalar_type = ScalarType::F32;
    kernels.push(f32_kernel);
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        kernels.push(
            differentiate(
                &fixtures::quadratic(1024),
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(0)],
                    dependent_operands: vec![OperandId::new(2)],
                },
            )
            .unwrap()
            .kernel,
        );
    }
    kernels
}

/// Digests computed on base commit `7bb4ccd`, before `PreparedKernel` and `run_batch` existed.
/// Prepared and batched execution are call-level attributes and must not move any identity.
const PINS: [&str; 7] = [
    "66b14e19eaebab306dd3ee6dd2a8ffec3f7d947d1344a99c5777d4280253a8a3",
    "b3a43d343fd3f051a464c88d7b6eba60c990f87333cddf1e6f8c72285ddd0ce5",
    "e5793669015b43118b4929181ef533d80cfdb5b0e03ae283795c22cdab26b97f",
    "b674e54018c407e86c59a53734029e1c8bc9764fc5fae4960eeb4d59a1aa7776",
    "b97923fb44e69363c8be0abe92efe2719ba218de68321744f63d63130df2baa6",
    "b0bbfa8648cbed9c4baf282399bea48bad4f2c1fc82f46dfb9dac9fa017cf0c8",
    "50e10d501b543bc6cdeab9d8e8d4f44acdafdbc36554ad7c47668a2cf4b577c6",
];

#[test]
fn kernel_identities_are_unchanged_by_prepared_and_batched_execution() {
    let kernels = pinned_kernels();
    for (kernel, pin) in kernels.iter().zip(PINS) {
        assert_eq!(
            kernel_digest(kernel).hex,
            pin,
            "{} identity moved",
            kernel.name
        );
        // Preparing (and running) a kernel never mutates the kernel it was built from.
        let executable = Executable::reference(validate(kernel.clone()).unwrap());
        PreparedKernel::new(&executable).unwrap();
        assert_eq!(kernel_digest(executable.kernel().as_kernel()).hex, pin);
    }
    let module = StructuredModule {
        name: "pins".into(),
        kernels,
    };
    assert_eq!(
        module_digest(&module).hex,
        "1fcaf60249f6d64f6004eac21da7c4db6bc8c23e8990c19cc224949f1b78c4a7"
    );
}
