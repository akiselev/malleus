use malleus::*;
use malleus_cranelift_experiment::{JitKernel, fixtures, interval::*};

fn execute(e: &Executable, buffers: &mut [Vec<f64>], native: bool) {
    let mut b: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
        .collect();
    if native {
        JitKernel::compile(e).unwrap().run(&mut b).unwrap();
    } else {
        Interpreter::run(e, &mut b).unwrap();
    }
}

#[test]
fn primal_and_actual_jvp_lift_to_existing_ir_and_compile() {
    assert!(ieee_environment_supported());
    let primal = fixtures::polynomial_batch(8);
    let jvp = differentiate(
        &primal,
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![OperandId::new(3)],
            dependent_operands: vec![OperandId::new(4)],
        },
    )
    .unwrap();
    for k in [primal, jvp.kernel] {
        let program = lower_intervals(&validate(k).unwrap()).unwrap();
        let mut a = fixtures::buffers(program.executable.kernel().as_kernel());
        // Start with point coefficients and t=0.25; all endpoint pairs agree.
        for pair in a.chunks_exact_mut(2) {
            pair[1] = pair[0].clone();
        }
        let mut b = a.clone();
        execute(&program.executable, &mut a, false);
        execute(&program.executable, &mut b, true);
        assert_eq!(a, b);
    }
}

#[test]
fn overflow_and_indeterminate_products_widen_instead_of_proving_a_sign() {
    let k = fixtures::polynomial_batch(1);
    let p = lower_intervals(&validate(k).unwrap()).unwrap();
    for coeff in [f64::MAX, f64::INFINITY] {
        let mut b = vec![
            vec![0.],
            vec![0.],
            vec![coeff],
            vec![coeff],
            vec![0.],
            vec![0.],
            vec![2.],
            vec![2.],
            vec![0.],
            vec![0.],
        ];
        execute(&p.executable, &mut b, true);
        assert_eq!(b[8], [f64::NEG_INFINITY]);
        assert_eq!(b[9], [f64::INFINITY]);
    }
}

#[test]
fn unsupported_precision_effects_and_arithmetic_refuse() {
    let mut k = fixtures::polynomial_batch(1);
    k.numeric_policy.scalar_type = ScalarType::F32;
    assert!(lower_intervals(&validate(k).unwrap()).is_err());
    let k = fixtures::quadratic(3);
    assert!(lower_intervals(&validate(k).unwrap()).is_err());
    let mut k = fixtures::polynomial_batch(1);
    k.body.statements = vec![Statement::Store {
        operand: OperandId::new(4),
        value: ScalarExpr::unary(UnaryOp::Sin, ScalarExpr::Load(OperandId::new(3))),
    }];
    assert!(lower_intervals(&validate(k).unwrap()).is_err());
}

#[test]
fn prepared_input_updates_are_checked_and_visible_to_native_code() {
    let e = Executable::reference(validate(fixtures::polynomial(1)).unwrap());
    let j = JitKernel::compile(&e).unwrap();
    let mut buffers = [vec![1.], vec![0.], vec![1.], vec![0.], vec![0.]];
    let bindings = buffers
        .iter_mut()
        .enumerate()
        .rev()
        .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
        .collect();
    let mut call = j.prepare_owned(bindings).unwrap();
    call.run();
    assert_eq!(call.values(OperandId::new(4)).unwrap(), [1.]);
    call.update_input(OperandId::new(0), 0, &[3.]).unwrap();
    call.run();
    assert_eq!(call.values(OperandId::new(4)).unwrap(), [3.]);
    assert!(call.update_input(OperandId::new(4), 0, &[7.]).is_err());
    assert!(
        call.update_input(OperandId::new(0), usize::MAX, &[7.])
            .is_err()
    );
    assert!(call.update_input(OperandId::new(0), 1, &[7.]).is_err());
    assert!(call.update_input(OperandId::new(99), 0, &[7.]).is_err());
}
