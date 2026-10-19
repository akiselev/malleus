//! Warm native calls allocate nothing: prepared `run`, positional `call` and `run_batch`.
// Executes JIT-generated machine code, which miri cannot interpret.
#![cfg(not(miri))]
use malleus::prepared::*;
use malleus::*;
use malleus_cranelift::{JitKernel, fixtures};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
// SAFETY: forwards to the system allocator; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn warm_native_calls_do_not_allocate() {
    let k = fixtures::polynomial(4);
    let e = Executable::reference(validate(k.clone()).unwrap());
    let jit = JitKernel::compile(&e).unwrap();
    let mut buffers = fixtures::buffers(&k);
    let mut bindings: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, v)| BufferBinding::new(OperandId::new(i), v))
        .collect();
    let mut call = jit.prepare(&mut bindings).unwrap();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..100 {
        call.run();
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), before, "prepared run");
    drop(call);
    drop(bindings);

    let points = 16;
    let t = vec![0.5; 4 * points];
    let (c0, c1, c2) = ([1.0], [2.0], [3.0]);
    let mut out = vec![0.0; 4 * points];
    let reads = [
        ReadBinding::strided(OperandId::new(0), &t, 4),
        ReadBinding::new(OperandId::new(1), &c0),
        ReadBinding::new(OperandId::new(2), &c1),
        ReadBinding::new(OperandId::new(3), &c2),
    ];
    let mut writes = [WriteBinding::strided(OperandId::new(4), &mut out, 4)];
    let mut state = jit.state();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..100 {
        jit.call(&mut state, &reads, &mut writes).unwrap();
        jit.run_batch(&mut state, points, &reads, &mut writes)
            .unwrap();
    }
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        before,
        "positional calls"
    );
    assert!(out.iter().all(|&x| x == 1.0 + 0.5 * (2.0 + 3.0 * 0.5)));
}
