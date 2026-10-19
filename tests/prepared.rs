//! Prepared and batched execution against the reference interpreter, bitwise (NaNs equivalent).

mod support;

use malleus::*;
use support::{Rng, fixtures};

fn same(a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
        })
}

fn interpret(e: &Executable, mut buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut bindings: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    Interpreter::run(e, &mut bindings).unwrap();
    drop(bindings);
    buffers
}

/// Split operand-ordered buffers into the prepared call's read and writable bindings.
fn bind<'a>(
    k: &StructuredKernel,
    buffers: &'a mut [Vec<f64>],
    strides: &[usize],
) -> (Vec<ReadBinding<'a>>, Vec<WriteBinding<'a>>) {
    let (mut reads, mut writes) = (vec![], vec![]);
    for (i, values) in buffers.iter_mut().enumerate() {
        let id = OperandId::new(i);
        if k.operands[i].access == AccessMode::Read {
            reads.push(ReadBinding::strided(id, values, strides[i]));
        } else {
            writes.push(WriteBinding::strided(id, values, strides[i]));
        }
    }
    (reads, writes)
}

fn special(r: &mut Rng) -> f64 {
    match r.below(12) {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => 0.0,
        4 => -0.0,
        5 => f64::from_bits(1 + r.below(1000)),
        6..=8 => r.finite(),
        _ => (r.below(2001) as f64 - 1000.0) / 64.0,
    }
}

fn constant(r: &mut Rng) -> f64 {
    match r.below(6) {
        0 => 0.0,
        1 => -0.0,
        2 => 1e-310,
        3 => r.finite(),
        _ => (r.below(41) as f64 - 20.0) / 4.0,
    }
}

struct Gen<'a> {
    r: &'a mut Rng,
    rank: usize,
    readable: Vec<usize>,
    locals: usize,
}

impl Gen<'_> {
    fn expr(&mut self, depth: u32) -> ScalarExpr {
        if depth == 0 || self.r.below(5) == 0 {
            return match self.r.below(4) {
                1 => ScalarExpr::Constant(constant(self.r)),
                2 if self.rank > 0 => {
                    ScalarExpr::Index(AxisId::new(self.r.below(self.rank as u64) as usize))
                }
                3 if self.locals > 0 => {
                    ScalarExpr::Local(LocalId::new(self.r.below(self.locals as u64) as usize))
                }
                _ => {
                    let i = self.r.below(self.readable.len() as u64) as usize;
                    ScalarExpr::Load(OperandId::new(self.readable[i]))
                }
            };
        }
        match self.r.below(5) {
            0 => {
                use UnaryOp::*;
                let ops = [Neg, Abs, Sqrt, Exp, Ln, Sin, Cos, Tan, Floor, Ceil];
                let op = ops[self.r.below(ops.len() as u64) as usize];
                ScalarExpr::unary(op, self.expr(depth - 1))
            }
            1 | 2 => {
                use BinaryOp::*;
                let ops = [Add, Sub, Mul, Div, Pow, Min, Max, Atan2];
                let op = ops[self.r.below(ops.len() as u64) as usize];
                ScalarExpr::binary(op, self.expr(depth - 1), self.expr(depth - 1))
            }
            _ => ScalarExpr::Select {
                condition: Box::new(self.predicate(depth - 1)),
                if_true: Box::new(self.expr(depth - 1)),
                if_false: Box::new(self.expr(depth - 1)),
            },
        }
    }

    fn predicate(&mut self, depth: u32) -> Predicate {
        let d = depth.saturating_sub(1);
        match if depth == 0 { 4 } else { self.r.below(6) } {
            0 => Predicate::Constant(self.r.below(2) == 0),
            1 => Predicate::Not(Box::new(self.predicate(d))),
            2 => Predicate::And(Box::new(self.predicate(d)), Box::new(self.predicate(d))),
            3 => Predicate::Or(Box::new(self.predicate(d)), Box::new(self.predicate(d))),
            _ => {
                use CompareOp::*;
                let ops = [Eq, NotEq, Less, LessEqual, Greater, GreaterEqual];
                Predicate::Compare {
                    op: ops[self.r.below(6) as usize],
                    lhs: Box::new(self.expr(d)),
                    rhs: Box::new(self.expr(d)),
                }
            }
        }
    }
}

/// A random kernel over every expression form, all access modes, affine reversed/constant
/// maps, column-major layout, nonzero regions, f32/f64 and schedule-defined loop orders.
fn random_executable(r: &mut Rng) -> Executable {
    use AccessMode::*;
    let rank = r.below(3) as usize;
    let reduce = [
        ReductionOp::Add,
        ReductionOp::Multiply,
        ReductionOp::Min,
        ReductionOp::Max,
    ][r.below(4) as usize];
    let (i, j) = (AxisId::new(0), AxisId::new(1));
    let (extents, operands, maps, iterators) = match rank {
        0 => (
            vec![],
            vec![
                KernelOperand::scalar("a", Read),
                KernelOperand::tensor("b", vec![3], Read),
                KernelOperand::scalar("c", Read).with_region(BufferRegion::new(2, 1)),
                KernelOperand::scalar("rw", ReadWrite),
                KernelOperand::scalar("w", Write),
                KernelOperand::scalar("red", Reduce(reduce)),
            ],
            vec![
                vec![],
                vec![IndexExpr::constant(2)],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            vec![],
        ),
        1 => {
            let n = 1 + r.below(5) as usize;
            let reversed = IndexExpr {
                constant: n as isize - 1,
                terms: vec![IndexTerm {
                    axis: i,
                    coefficient: -1,
                }],
            };
            (
                vec![n],
                vec![
                    KernelOperand::tensor("a", vec![n], Read),
                    KernelOperand::tensor("b", vec![n], Read).with_region(BufferRegion::new(1, n)),
                    KernelOperand::scalar("c", Read),
                    KernelOperand::tensor("rw", vec![n], ReadWrite),
                    KernelOperand::tensor("w", vec![n], Write)
                        .with_region(BufferRegion::new(2, n + 1)),
                    KernelOperand::scalar("red", Reduce(reduce)),
                ],
                vec![
                    vec![IndexExpr::axis(i)],
                    vec![reversed],
                    vec![],
                    vec![IndexExpr::axis(i)],
                    vec![IndexExpr::axis(i)],
                    vec![],
                ],
                vec![IteratorKind::Parallel],
            )
        }
        _ => {
            let (m, n) = (1 + r.below(3) as usize, 1 + r.below(3) as usize);
            (
                vec![m, n],
                vec![
                    KernelOperand::tensor("a", vec![m, n], Read)
                        .with_layout(DenseLayout::column_major(2)),
                    KernelOperand::tensor("b", vec![n], Read),
                    KernelOperand::scalar("c", Read),
                    KernelOperand::tensor("rw", vec![m, n], ReadWrite),
                    KernelOperand::tensor("w", vec![m, n], Write)
                        .with_region(BufferRegion::new(3, m * n)),
                    KernelOperand::tensor("red", vec![m], Reduce(reduce)),
                ],
                vec![
                    vec![IndexExpr::axis(i), IndexExpr::axis(j)],
                    vec![IndexExpr::axis(j)],
                    vec![],
                    vec![IndexExpr::axis(i), IndexExpr::axis(j)],
                    vec![IndexExpr::axis(i), IndexExpr::axis(j)],
                    vec![IndexExpr::axis(i)],
                ],
                vec![IteratorKind::Parallel, IteratorKind::Reduction],
            )
        }
    };
    let mut g = Gen {
        r,
        rank,
        readable: vec![0, 1, 2, 3],
        locals: 0,
    };
    let mut statements = vec![];
    let let_statement = |g: &mut Gen| {
        let depth = 1 + g.r.below(4) as u32;
        let value = g.expr(depth);
        g.locals += 1;
        Statement::Let {
            local: LocalId::new(g.locals - 1),
            value,
        }
    };
    for _ in 0..g.r.below(3) {
        statements.push(let_statement(&mut g));
    }
    for operand in [3, 4, 5] {
        if operand == 4 && g.r.below(2) == 0 {
            statements.push(let_statement(&mut g));
        }
        let depth = 1 + g.r.below(4) as u32;
        let value = g.expr(depth);
        statements.push(Statement::Store {
            operand: OperandId::new(operand),
            value,
        });
    }
    if g.r.below(2) == 0 {
        // A second read-write store observes the first one.
        let value = g.expr(2);
        statements.push(Statement::Store {
            operand: OperandId::new(3),
            value,
        });
    }
    let mut policy = NumericPolicy::default();
    if g.r.below(2) == 0 {
        policy.scalar_type = ScalarType::F32;
    }
    let schedule_defined = rank == 2 && g.r.below(2) == 0;
    if schedule_defined {
        policy.reduction_order = ReductionOrder::ScheduleDefined;
    }
    let kernel = StructuredKernel {
        name: "random".into(),
        iteration_domain: IterationDomain::new(extents),
        iterators,
        operands,
        indexing_maps: maps
            .into_iter()
            .enumerate()
            .map(|(o, m)| IndexingMap::new(OperandId::new(o), m))
            .collect(),
        body: KernelRegion { statements },
        numeric_policy: policy,
    };
    let validated = validate(kernel).unwrap();
    if schedule_defined {
        let schedule = KernelSchedule {
            loop_order: vec![AxisId::new(1), AxisId::new(0)],
            tiles: vec![],
            vectorization: None,
            parallel: vec![],
        };
        Executable::new(validated, schedule).unwrap()
    } else {
        Executable::reference(validated)
    }
}

fn random_buffers(r: &mut Rng, k: &StructuredKernel) -> Vec<Vec<f64>> {
    k.operands
        .iter()
        .map(|o| {
            (0..o.region.offset + o.region.length)
                .map(|_| special(r))
                .collect()
        })
        .collect()
}

/// The fixture families of the Rust-source and Cranelift differential tests.
fn fixture_corpus() -> Vec<(Executable, Vec<Vec<f64>>)> {
    let mut cases = vec![];
    let mut push = |k: StructuredKernel, b: Vec<Vec<f64>>| {
        cases.push((Executable::reference(validate(k).unwrap()), b))
    };
    for ty in [ScalarType::F32, ScalarType::F64] {
        let mut k = fixtures::polynomial_batch(8);
        k.numeric_policy.scalar_type = ty;
        push(k.clone(), fixtures::buffers(&k));
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let p = differentiate(
                &k,
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(3)],
                    dependent_operands: vec![OperandId::new(4)],
                },
            )
            .unwrap();
            let b = fixtures::buffers(&p.kernel);
            push(p.kernel, b);
        }
        for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
            let mut q = fixtures::quadratic(64);
            q.numeric_policy.scalar_type = ty;
            let p = differentiate(
                &q,
                &DerivativeRequest {
                    mode,
                    independent_operands: vec![OperandId::new(0)],
                    dependent_operands: vec![OperandId::new(2)],
                },
            )
            .unwrap();
            let b = fixtures::buffers(&p.kernel);
            push(p.kernel, b);
        }
        for op in [
            ReductionOp::Add,
            ReductionOp::Multiply,
            ReductionOp::Min,
            ReductionOp::Max,
        ] {
            let mut k = fixtures::quadratic(8);
            k.numeric_policy.scalar_type = ty;
            k.operands[2].access = AccessMode::Reduce(op);
            push(k.clone(), fixtures::buffers(&k));
        }
        let mut k = fixtures::polynomial(8);
        k.numeric_policy.scalar_type = ty;
        let x = ScalarExpr::Load(OperandId::new(0));
        k.body.statements = vec![Statement::Store {
            operand: OperandId::new(4),
            value: ScalarExpr::Select {
                condition: Box::new(Predicate::Compare {
                    op: CompareOp::Less,
                    lhs: Box::new(x.clone()),
                    rhs: Box::new(ScalarExpr::Constant(0.)),
                }),
                if_true: Box::new(ScalarExpr::unary(UnaryOp::Neg, x.clone())),
                if_false: Box::new(ScalarExpr::unary(UnaryOp::Sqrt, x)),
            },
        }];
        let mut b = fixtures::buffers(&k);
        b[0] = vec![
            -0.,
            0.,
            -1.,
            4.,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            0.1,
        ];
        push(k, b);
        let mut k = fixtures::matvec(8);
        k.numeric_policy.scalar_type = ty;
        push(k.clone(), fixtures::buffers(&k));
        let mut k = fixtures::polynomial(1);
        k.numeric_policy.scalar_type = ty;
        let eps = 2f64.powi(if ty == ScalarType::F32 { -13 } else { -27 });
        push(
            k,
            vec![
                vec![1. + eps],
                vec![-1.],
                vec![1. - eps],
                vec![0.],
                vec![0.],
            ],
        );
    }
    let mut k = fixtures::polynomial(8);
    k.indexing_maps[0].results = vec![IndexExpr {
        constant: 7,
        terms: vec![IndexTerm {
            axis: AxisId::new(0),
            coefficient: -1,
        }],
    }];
    k.operands[4].region = BufferRegion::new(3, 8);
    push(k.clone(), fixtures::buffers(&k));
    // Empty iteration domain: nothing executes, outputs keep their contents.
    let mut k = fixtures::polynomial(1);
    k.iteration_domain = IterationDomain::new(vec![0]);
    let mut b = fixtures::buffers(&k);
    b[4] = vec![-0.0];
    push(k, b);
    cases
}

fn all_cases() -> Vec<(Executable, Vec<Vec<f64>>)> {
    let mut cases = fixture_corpus();
    let mut r = Rng(0x5eed_cafe_f00d_0001);
    for _ in 0..400 {
        let e = random_executable(&mut r);
        let b = random_buffers(&mut r, e.kernel().as_kernel());
        cases.push((e, b));
    }
    cases
}

#[test]
fn prepared_one_shot_and_reusable_calls_match_the_interpreter_bitwise() {
    let mut shared_state = PreparedState::new();
    for (index, (e, buffers)) in all_cases().into_iter().enumerate() {
        let k = e.kernel().as_kernel();
        let expected = interpret(&e, buffers.clone());
        let prepared = PreparedKernel::new(&e).unwrap();

        let mut one_shot = buffers.clone();
        let mut bindings: Vec<_> = one_shot
            .iter_mut()
            .enumerate()
            .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
            .collect();
        prepared.run(&mut bindings).unwrap();
        drop(bindings);
        assert!(same(&expected, &one_shot), "one-shot case {index}: {k:?}");

        // A presized state and one state shared across differently sized kernels.
        for state in [&mut prepared.state(), &mut shared_state] {
            let mut reusable = buffers.clone();
            let zeros = vec![0; k.operands.len()];
            let (reads, mut writes) = bind(k, &mut reusable, &zeros);
            prepared.call(state, &reads, &mut writes).unwrap();
            drop((reads, writes));
            assert!(same(&expected, &reusable), "call case {index}: {k:?}");
        }
    }
}

/// Oracle for any stride pattern: `points` sequential interpreter calls on windows shifted by
/// `p * stride` of one buffer per operand.
fn sequential_oracle(
    e: &Executable,
    mut buffers: Vec<Vec<f64>>,
    strides: &[usize],
    points: usize,
) -> Vec<Vec<f64>> {
    for p in 0..points {
        let mut bindings: Vec<_> = buffers
            .iter_mut()
            .enumerate()
            .map(|(i, v)| BufferBinding::new(OperandId::new(i), &mut v[p * strides[i]..]))
            .collect();
        Interpreter::run(e, &mut bindings).unwrap();
    }
    buffers
}

fn batch_buffers(
    r: &mut Rng,
    k: &StructuredKernel,
    strides: &[usize],
    points: usize,
) -> Vec<Vec<f64>> {
    k.operands
        .iter()
        .zip(strides)
        .map(|(o, stride)| {
            let required = o.region.offset + o.region.length;
            let len = points.saturating_sub(1) * stride + required + 2;
            (0..len).map(|_| special(r)).collect()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Every operand strided (reads possibly padded, some reads uniform): columnar.
    Disjoint,
    /// Every writable operand shared by all points: sequential accumulation.
    SharedOutputs,
    /// Writable windows overlap between consecutive points: sequential.
    Overlapping,
}

fn strides_for(r: &mut Rng, k: &StructuredKernel, layout: Layout) -> Vec<usize> {
    k.operands
        .iter()
        .map(|o| {
            let required = o.region.offset + o.region.length;
            if o.access == AccessMode::Read {
                if r.below(3) == 0 {
                    0
                } else {
                    o.region.length + r.below(3) as usize
                }
            } else {
                match layout {
                    Layout::Disjoint => o.region.length + r.below(2) as usize * required,
                    Layout::SharedOutputs => 0,
                    Layout::Overlapping => o.region.length.saturating_sub(1).max(1),
                }
            }
        })
        .collect()
}

#[test]
fn batches_match_sequential_interpreter_calls_for_every_stride_pattern() {
    let mut r = Rng(0xba7c_4ed0_0000_0002);
    let mut state = PreparedState::new();
    let cases = all_cases();
    for (index, (e, _)) in cases.iter().enumerate() {
        let k = e.kernel().as_kernel();
        let prepared = PreparedKernel::new(e).unwrap();
        let point_counts: &[usize] = if index % 7 == 0 {
            &[0, 1, 2, 31, 32, 33, 70]
        } else {
            &[3, BATCH_BLOCK + 5]
        };
        for &points in point_counts {
            for layout in [Layout::Disjoint, Layout::SharedOutputs, Layout::Overlapping] {
                let strides = strides_for(&mut r, k, layout);
                let buffers = batch_buffers(&mut r, k, &strides, points);
                let expected = sequential_oracle(e, buffers.clone(), &strides, points);
                let mut actual = buffers;
                let (reads, mut writes) = bind(k, &mut actual, &strides);
                prepared
                    .run_batch(&mut state, points, &reads, &mut writes)
                    .unwrap();
                drop((reads, writes));
                assert!(
                    same(&expected, &actual),
                    "case {index}, {points} points, {layout:?}, strides {strides:?}: {k:?}"
                );
            }
        }
    }
}

#[test]
fn malformed_bindings_are_refused_before_any_effect() {
    let e = Executable::reference(validate(fixtures::polynomial(4)).unwrap());
    let k = e.kernel().as_kernel();
    let prepared = PreparedKernel::new(&e).unwrap();
    assert_eq!(prepared.read_operands(), &[0, 1, 2, 3].map(OperandId::new));
    assert_eq!(prepared.write_operands(), &[OperandId::new(4)]);
    assert_eq!(prepared.required_len(OperandId::new(0)), Some(4));
    assert_eq!(prepared.required_len(OperandId::new(5)), None);
    let mut state = prepared.state();
    let pristine = vec![
        vec![1.0; 12],
        vec![2.0; 3],
        vec![3.0; 3],
        vec![4.0; 3],
        vec![-0.0; 12],
    ];
    let strides = [4, 1, 1, 0, 4];
    let attempt = |points: usize,
                   edit: &dyn Fn(&mut Vec<ReadBinding>, &mut Vec<WriteBinding>),
                   state: &mut PreparedState| {
        let mut buffers = pristine.clone();
        let (mut reads, mut writes) = bind(k, &mut buffers, &strides);
        edit(&mut reads, &mut writes);
        let result = prepared.run_batch(state, points, &reads, &mut writes);
        drop((reads, writes));
        assert_eq!(
            result.is_err(),
            same(&buffers, &pristine),
            "a refused call had an effect, or an accepted call had none"
        );
        result
    };
    assert_eq!(
        attempt(
            3,
            &|reads, _| {
                reads.pop();
            },
            &mut state
        ),
        Err(CallError::ReadCount {
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(
        attempt(3, &|_, writes| writes.clear(), &mut state),
        Err(CallError::WriteCount {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        attempt(3, &|reads, _| reads.swap(1, 2), &mut state),
        Err(CallError::ReadOperand {
            position: 1,
            expected: OperandId::new(1),
            actual: OperandId::new(2)
        })
    );
    assert_eq!(
        attempt(
            3,
            &|_, writes| writes[0].operand = OperandId::new(3),
            &mut state
        ),
        Err(CallError::WriteOperand {
            position: 0,
            expected: OperandId::new(4),
            actual: OperandId::new(3)
        })
    );
    // Three points need (3 - 1) * 4 + 4 = 12 elements; four points need 16.
    assert_eq!(attempt(3, &|_, _| (), &mut state), Ok(()));
    assert_eq!(
        attempt(4, &|_, _| (), &mut state),
        Err(CallError::ShortBuffer {
            operand: OperandId::new(0),
            required: 16,
            actual: 12
        })
    );
    assert_eq!(
        attempt(3, &|_, writes| writes[0].stride = usize::MAX, &mut state),
        Err(CallError::ExtentOverflow {
            operand: OperandId::new(4)
        })
    );
    // The checked one-shot path refuses exactly what the interpreter refuses.
    let malformed: Vec<Vec<(usize, usize)>> = vec![
        vec![(0, 4), (1, 1), (2, 1), (3, 1)],
        vec![(0, 4), (1, 1), (2, 1), (3, 1), (4, 4), (4, 4)],
        vec![(0, 4), (1, 1), (2, 1), (3, 1), (9, 4)],
        vec![(0, 3), (1, 1), (2, 1), (3, 1), (4, 4)],
        vec![(4, 4), (3, 1), (2, 1), (1, 0), (0, 4)],
    ];
    for shape in malformed {
        let mut a: Vec<Vec<f64>> = shape.iter().map(|(_, n)| vec![0.5; *n]).collect();
        let mut b = a.clone();
        let mut ia: Vec<_> = a
            .iter_mut()
            .zip(&shape)
            .map(|(v, (id, _))| BufferBinding::new(OperandId::new(*id), v))
            .collect();
        let mut ib: Vec<_> = b
            .iter_mut()
            .zip(&shape)
            .map(|(v, (id, _))| BufferBinding::new(OperandId::new(*id), v))
            .collect();
        let reference = Interpreter::run(&e, &mut ia).unwrap_err();
        assert_eq!(prepared.run(&mut ib).unwrap_err(), reference, "{shape:?}");
    }
}

#[test]
fn every_kernel_of_a_module_is_prepared_by_name() {
    let module = validate_module(StructuredModule {
        name: "pair".into(),
        kernels: vec![fixtures::polynomial(4), fixtures::matvec(3)],
    })
    .unwrap();
    let prepared = PreparedModule::new(&ExecutableModule::reference(module)).unwrap();
    assert_eq!(prepared.name(), "pair");
    assert_eq!(prepared.kernels().len(), 2);
    let matvec = prepared.kernel("matvec-3").unwrap();
    assert_eq!(matvec.name(), "matvec-3");
    assert_eq!(matvec.scalar_type(), ScalarType::F64);
    assert!(prepared.kernel("missing").is_none());
}

fn composition() -> KernelComposition {
    KernelComposition {
        name: "polynomial-into-quadratic".into(),
        stages: vec![fixtures::polynomial(4), fixtures::quadratic(4)],
        shared_buffers: vec![SharedBuffer::bind(
            StageOperand::new(0, OperandId::new(4)),
            StageOperand::new(1, OperandId::new(0)),
        )],
    }
}

type CompositionBuffers = Vec<(CompositionTarget, Vec<f64>)>;
type Edit = Box<dyn Fn(&mut CompositionBuffers)>;

/// Bindings for every exposed operand and, when `bind_shared`, every shared buffer.
fn composition_buffers(
    r: &mut Rng,
    e: &ExecutableComposition,
    bind_shared: bool,
) -> CompositionBuffers {
    let mut buffers = vec![];
    for (s, stage) in e.stages().iter().enumerate() {
        for (o, operand) in stage.kernel().as_kernel().operands.iter().enumerate() {
            let target = StageOperand::new(s, OperandId::new(o));
            if e.shared_buffer_of(target).is_none() {
                let len = operand.region.offset + operand.region.length;
                buffers.push((
                    CompositionTarget::Operand(target),
                    (0..len).map(|_| special(r)).collect(),
                ));
            }
        }
    }
    if bind_shared {
        let mut buffer = 0;
        while let Some(len) = e.shared_length(buffer) {
            buffers.push((
                CompositionTarget::Shared(buffer),
                (0..len).map(|_| special(r)).collect(),
            ));
            buffer += 1;
        }
    }
    buffers
}

fn composition_bindings(
    buffers: &mut [(CompositionTarget, Vec<f64>)],
) -> Vec<CompositionBinding<'_>> {
    buffers
        .iter_mut()
        .map(|(target, values)| CompositionBinding {
            target: *target,
            values,
        })
        .collect()
}

#[test]
fn prepared_compositions_match_the_interpreter_including_derivatives_and_errors() {
    let primal = composition();
    let mut executables = vec![primal.clone()];
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        let product = differentiate_composition(
            &primal,
            &CompositionDerivativeRequest {
                mode,
                independent_operands: vec![StageOperand::new(0, OperandId::new(0))],
                dependent_operands: vec![StageOperand::new(1, OperandId::new(2))],
            },
        )
        .unwrap();
        executables.push(product.composition);
    }
    let mut single = primal.clone();
    for stage in &mut single.stages {
        stage.numeric_policy.scalar_type = ScalarType::F32;
    }
    executables.push(single);
    let mut r = Rng(0xc0_4905_1710_0003);
    for composition in executables {
        let e = ExecutableComposition::reference(validate_composition(composition).unwrap());
        let prepared = PreparedComposition::new(&e).unwrap();
        assert_eq!(prepared.name(), e.name());
        assert_eq!(prepared.stages().len(), e.stages().len());
        let mut state = prepared.state();
        for round in 0..20 {
            let buffers = composition_buffers(&mut r, &e, round % 2 == 0);
            let mut expected = buffers.clone();
            Interpreter::run_composition(&e, &mut composition_bindings(&mut expected)).unwrap();
            let mut actual = buffers;
            prepared
                .run(&mut state, &mut composition_bindings(&mut actual))
                .unwrap();
            let values = |b: &[(CompositionTarget, Vec<f64>)]| {
                b.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>()
            };
            assert!(same(&values(&expected), &values(&actual)), "{}", e.name());
        }
    }
    // Every binding error, in the interpreter's order and with its values.
    let e = ExecutableComposition::reference(validate_composition(primal).unwrap());
    let prepared = PreparedComposition::new(&e).unwrap();
    let mut state = prepared.state();
    let base = composition_buffers(&mut r, &e, false);
    let edits: Vec<Edit> = vec![
        Box::new(|b| {
            b.remove(0);
        }),
        Box::new(|b| {
            let first = b[0].clone();
            b.push(first)
        }),
        Box::new(|b| b[1].1.clear()),
        Box::new(|b| {
            b.push((
                CompositionTarget::Operand(StageOperand::new(1, OperandId::new(0))),
                vec![0.0; 4],
            ))
        }),
        Box::new(|b| {
            b.push((
                CompositionTarget::Operand(StageOperand::new(7, OperandId::new(0))),
                vec![0.0; 4],
            ))
        }),
        Box::new(|b| b.push((CompositionTarget::Shared(3), vec![0.0; 4]))),
        Box::new(|b| b.push((CompositionTarget::Shared(0), vec![0.0; 1]))),
        Box::new(|b| {
            b.push((CompositionTarget::Shared(0), vec![0.0; 4]));
            b.push((CompositionTarget::Shared(0), vec![0.0; 4]));
        }),
    ];
    for (index, edit) in edits.iter().enumerate() {
        let mut expected = base.clone();
        edit(&mut expected);
        let mut actual = expected.clone();
        let reference =
            Interpreter::run_composition(&e, &mut composition_bindings(&mut expected)).unwrap_err();
        let error = prepared
            .run(&mut state, &mut composition_bindings(&mut actual))
            .unwrap_err();
        assert_eq!(error, reference, "edit {index}");
    }
}
