//! Prepared calls, batches and compositions perform no heap allocation once their state is
//! sized. One test per binary: the counting allocator is process-global.

mod support;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use malleus::*;
use support::fixtures;

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates every operation to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    f();
    ALLOCATIONS.load(Ordering::SeqCst) - before
}

#[test]
fn sized_prepared_calls_never_allocate() {
    let mut select = fixtures::polynomial(64);
    let x = ScalarExpr::Load(OperandId::new(0));
    select.body.statements = vec![
        Statement::Let {
            local: LocalId::new(0),
            value: ScalarExpr::unary(UnaryOp::Exp, x.clone()),
        },
        Statement::Store {
            operand: OperandId::new(4),
            value: ScalarExpr::Select {
                condition: Box::new(Predicate::Or(
                    Box::new(Predicate::Compare {
                        op: CompareOp::Less,
                        lhs: Box::new(x.clone()),
                        rhs: Box::new(ScalarExpr::Constant(0.)),
                    }),
                    Box::new(Predicate::Constant(false)),
                )),
                if_true: Box::new(ScalarExpr::Local(LocalId::new(0))),
                if_false: Box::new(ScalarExpr::unary(UnaryOp::Sqrt, x)),
            },
        },
    ];
    for kernel in [select, fixtures::matvec(16), fixtures::quadratic(32)] {
        let e = Executable::reference(validate(kernel).unwrap());
        let k = e.kernel().as_kernel();
        let prepared = PreparedKernel::new(&e).unwrap();
        let mut state = prepared.state();
        const POINTS: usize = 2 * BATCH_BLOCK + 3;
        let mut buffers: Vec<Vec<f64>> = fixtures::buffers(k)
            .into_iter()
            .map(|b| b.repeat(POINTS))
            .collect();
        let (mut reads, mut writes) = (vec![], vec![]);
        for (index, values) in buffers.iter_mut().enumerate() {
            let stride = values.len() / POINTS;
            let operand = OperandId::new(index);
            if k.operands[index].access == AccessMode::Read {
                reads.push(ReadBinding::strided(operand, values, stride));
            } else {
                writes.push(WriteBinding::strided(operand, values, stride));
            }
        }
        let count = allocations(|| {
            for _ in 0..10 {
                prepared.call(&mut state, &reads, &mut writes).unwrap();
                prepared
                    .run_batch(&mut state, POINTS, &reads, &mut writes)
                    .unwrap();
            }
            // Shared outputs take the sequential path.
            for write in writes.iter_mut() {
                write.stride = 0;
            }
            prepared
                .run_batch(&mut state, POINTS, &reads, &mut writes)
                .unwrap();
        });
        assert_eq!(count, 0, "{}", k.name);
    }

    let composition = KernelComposition {
        name: "polynomial-into-quadratic".into(),
        stages: vec![fixtures::polynomial(4), fixtures::quadratic(4)],
        shared_buffers: vec![SharedBuffer::bind(
            StageOperand::new(0, OperandId::new(4)),
            StageOperand::new(1, OperandId::new(0)),
        )],
    };
    let e = ExecutableComposition::reference(validate_composition(composition).unwrap());
    let prepared = PreparedComposition::new(&e).unwrap();
    let mut state = prepared.state();
    let (mut t, mut c0, mut c1, mut c2, mut p, mut sum) = (
        vec![0.5; 4],
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![0.25],
        vec![0.0],
    );
    let operand = |s, o| CompositionTarget::Operand(StageOperand::new(s, OperandId::new(o)));
    let mut bindings = [
        CompositionBinding {
            target: operand(0, 0),
            values: &mut t,
        },
        CompositionBinding {
            target: operand(0, 1),
            values: &mut c0,
        },
        CompositionBinding {
            target: operand(0, 2),
            values: &mut c1,
        },
        CompositionBinding {
            target: operand(0, 3),
            values: &mut c2,
        },
        CompositionBinding {
            target: operand(1, 1),
            values: &mut p,
        },
        CompositionBinding {
            target: operand(1, 2),
            values: &mut sum,
        },
    ];
    let count = allocations(|| {
        for _ in 0..10 {
            prepared.run(&mut state, &mut bindings).unwrap();
        }
    });
    assert_eq!(count, 0, "composition");
}
