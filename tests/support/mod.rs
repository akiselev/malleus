//! Shared test support: fixtures, exact arithmetic, random interval programs and a rustc runner.
#![allow(dead_code)]

pub mod dyadic;
pub mod fixtures;

use dyadic::Dyadic;
use malleus::interval::{IntervalProgram, lower_intervals};
use malleus::*;
use std::{fs, process::Command};

/// xorshift64: small, deterministic and independent of any external crate.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Finite doubles biased towards the hard cases: subnormals, near-MAX, +-0, tiny and huge
    /// exponents, and short dyadics.
    pub fn finite(&mut self) -> f64 {
        loop {
            let sign = if self.below(2) == 0 { 1.0 } else { -1.0 };
            let x = match self.below(7) {
                0 => f64::from_bits(self.next()),
                1 => (self.below(2001) as f64 - 1000.0) / 8.0,
                2 => f64::from_bits(self.next() & 0x800F_FFFF_FFFF_FFFF),
                3 => (self.next() as f64 / u64::MAX as f64 - 0.5) * 4.0,
                4 => {
                    let e = (1023 + self.below(120) as i64 - 60) as u64;
                    sign * f64::from_bits(e << 52 | (self.next() & 0xF_FFFF_FFFF_FFFF))
                }
                5 => f64::from_bits(f64::MAX.to_bits() - self.below(4)) * sign,
                _ => [
                    0.0,
                    -0.0,
                    1.0,
                    -1.0,
                    0.1,
                    f64::MAX,
                    f64::MIN_POSITIVE,
                    1e154,
                    1e-160,
                    f64::from_bits(1),
                ][self.below(10) as usize],
            };
            if x.is_finite() {
                return x;
            }
        }
    }
    /// An ordered, non-NaN endpoint pair; degenerate (point) intervals are common. When
    /// `allow_infinite`, an endpoint is occasionally infinite.
    pub fn interval(&mut self, allow_infinite: bool) -> (f64, f64) {
        let a = self.finite();
        let b = if self.below(3) == 0 { a } else { self.finite() };
        let (mut lo, mut hi) = if a <= b { (a, b) } else { (b, a) };
        if allow_infinite && self.below(12) == 0 {
            lo = f64::NEG_INFINITY;
        }
        if allow_infinite && self.below(12) == 0 {
            hi = f64::INFINITY;
        }
        (lo, hi)
    }
}

/// Source-level expression over the supported interval fragment.
#[derive(Clone, Debug)]
pub enum Ex {
    Load(usize),
    Const(f64),
    Local(usize),
    Neg(Box<Ex>),
    Add(Box<Ex>, Box<Ex>),
    Sub(Box<Ex>, Box<Ex>),
    Mul(Box<Ex>, Box<Ex>),
    Square(Box<Ex>),
    /// `if axis0 < threshold { a } else { b }`: an index-only branch.
    IndexSelect(usize, Box<Ex>, Box<Ex>),
}

impl Ex {
    pub fn scalar(&self) -> ScalarExpr {
        let b = |e: &Ex| e.scalar();
        match self {
            Ex::Load(i) => ScalarExpr::Load(OperandId::new(*i)),
            Ex::Const(x) => ScalarExpr::Constant(*x),
            Ex::Local(i) => ScalarExpr::Local(LocalId::new(*i)),
            Ex::Neg(a) => ScalarExpr::unary(UnaryOp::Neg, b(a)),
            Ex::Add(x, y) => ScalarExpr::binary(BinaryOp::Add, b(x), b(y)),
            Ex::Sub(x, y) => ScalarExpr::binary(BinaryOp::Sub, b(x), b(y)),
            Ex::Mul(x, y) => ScalarExpr::binary(BinaryOp::Mul, b(x), b(y)),
            Ex::Square(x) => ScalarExpr::binary(BinaryOp::Mul, b(x), b(x)),
            Ex::IndexSelect(t, x, y) => ScalarExpr::Select {
                condition: Box::new(Predicate::Compare {
                    op: CompareOp::Less,
                    lhs: Box::new(ScalarExpr::Index(AxisId::new(0))),
                    rhs: Box::new(ScalarExpr::Constant(*t as f64)),
                }),
                if_true: Box::new(b(x)),
                if_false: Box::new(b(y)),
            },
        }
    }

    /// Exact real value at iteration `i` with exact `loads` and already-evaluated `locals`.
    pub fn exact(&self, loads: &[Dyadic], locals: &[Dyadic], i: usize) -> Dyadic {
        let e = |x: &Ex| x.exact(loads, locals, i);
        match self {
            Ex::Load(k) => loads[*k].clone(),
            Ex::Const(x) => Dyadic::from_f64(*x),
            Ex::Local(k) => locals[*k].clone(),
            Ex::Neg(a) => e(a).neg(),
            Ex::Add(a, b) => e(a).add(&e(b)),
            Ex::Sub(a, b) => e(a).sub(&e(b)),
            Ex::Mul(a, b) => e(a).mul(&e(b)),
            Ex::Square(a) => {
                let v = e(a);
                v.mul(&v)
            }
            Ex::IndexSelect(t, a, b) => {
                if i < *t {
                    e(a)
                } else {
                    e(b)
                }
            }
        }
    }

    fn generate(r: &mut Rng, depth: u32, locals: usize) -> Ex {
        let leaf = depth == 0 || r.below(8) == 0;
        let k = if leaf { r.below(3) } else { 3 + r.below(7) };
        let sub = |r: &mut Rng| Box::new(Ex::generate(r, depth.saturating_sub(1), locals));
        match k {
            0 => Ex::Load(r.below(4) as usize),
            1 => Ex::Const(r.finite()),
            2 if locals > 0 => Ex::Local(r.below(locals as u64) as usize),
            2 => Ex::Load(r.below(4) as usize),
            3 => Ex::Neg(sub(r)),
            4 => Ex::Square(sub(r)),
            5 => Ex::Add(sub(r), sub(r)),
            6 => Ex::Sub(sub(r), sub(r)),
            7 | 8 => Ex::Mul(sub(r), sub(r)),
            _ => Ex::IndexSelect(r.below(3) as usize, sub(r), sub(r)),
        }
    }
}

/// A random kernel: `n` iterations over four Read interval operands `x0..x3` and one Write
/// operand, with a few shared locals and one store.
#[derive(Clone, Debug)]
pub struct Prog {
    pub n: usize,
    pub lets: Vec<Ex>,
    pub store: Ex,
}

impl Prog {
    pub fn generate(r: &mut Rng, max_depth: u32) -> Prog {
        let n = 1 + r.below(3) as usize;
        let mut lets = vec![];
        for _ in 0..r.below(3) {
            let d = 1 + r.below(max_depth.max(1) as u64) as u32;
            lets.push(Ex::generate(r, d, lets.len()));
        }
        let d = 1 + r.below(max_depth.max(1) as u64) as u32;
        let store = Ex::generate(r, d, lets.len());
        Prog { n, lets, store }
    }

    pub fn kernel(&self) -> StructuredKernel {
        let axis = AxisId::new(0);
        let mut operands: Vec<_> = (0..4)
            .map(|i| KernelOperand::tensor(format!("x{i}"), vec![self.n], AccessMode::Read))
            .collect();
        operands.push(KernelOperand::tensor(
            "out",
            vec![self.n],
            AccessMode::Write,
        ));
        let mut statements: Vec<_> = self
            .lets
            .iter()
            .enumerate()
            .map(|(i, e)| Statement::Let {
                local: LocalId::new(i),
                value: e.scalar(),
            })
            .collect();
        statements.push(Statement::Store {
            operand: OperandId::new(4),
            value: self.store.scalar(),
        });
        StructuredKernel {
            name: "random-interval".into(),
            iteration_domain: IterationDomain::new(vec![self.n]),
            iterators: vec![IteratorKind::Parallel],
            operands,
            indexing_maps: (0..5)
                .map(|i| IndexingMap::new(OperandId::new(i), vec![IndexExpr::axis(axis)]))
                .collect(),
            body: KernelRegion { statements },
            numeric_policy: NumericPolicy::default(),
        }
    }

    pub fn lower(&self) -> IntervalProgram {
        lower_intervals(&validate(self.kernel()).unwrap()).unwrap()
    }

    /// Exact real value of the store at iteration `i` for exact operand values.
    pub fn exact(&self, loads: &[Dyadic], i: usize) -> Dyadic {
        let mut locals = vec![];
        for e in &self.lets {
            let v = e.exact(loads, &locals, i);
            locals.push(v);
        }
        self.store.exact(loads, &locals, i)
    }
}

/// Per-element input enclosures for the four source operands.
pub type Inputs = Vec<[(f64, f64); 4]>;

pub fn random_inputs(r: &mut Rng, n: usize, allow_infinite: bool) -> Inputs {
    (0..n)
        .map(|_| std::array::from_fn(|_| r.interval(allow_infinite)))
        .collect()
}

/// Endpoint buffers in lowered operand order (`lo`/`hi` per source operand) with NaN-filled
/// output endpoints.
pub fn endpoint_buffers(n: usize, inputs: &Inputs) -> Vec<Vec<f64>> {
    let mut b = vec![];
    for k in 0..4 {
        b.push(inputs.iter().map(|e| e[k].0).collect());
        b.push(inputs.iter().map(|e| e[k].1).collect());
    }
    b.push(vec![f64::NAN; n]);
    b.push(vec![f64::NAN; n]);
    b
}

/// Run an executable on the reference interpreter and return all buffers afterwards.
pub fn interpret(e: &Executable, mut buffers: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut bindings: Vec<_> = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, b)| BufferBinding::new(OperandId::new(i), b))
        .collect();
    Interpreter::run(e, &mut bindings).unwrap();
    drop(bindings);
    buffers
}

/// Bit equality, treating any two NaNs as equal.
pub fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Compile `source` (which must define `fn main`) with `rustc -O`, run it, return stdout.
pub fn rustc_run(tag: &str, source: &str) -> String {
    let dir = std::env::temp_dir().join(format!("malleus-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("kernels.rs"), source).unwrap();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let compiled = Command::new(rustc)
        .arg("--edition=2024")
        .arg("-O")
        .arg(dir.join("kernels.rs"))
        .arg("-o")
        .arg(dir.join("kernels"))
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(dir.join("kernels")).output().unwrap();
    assert!(run.status.success());
    let stdout = String::from_utf8(run.stdout).unwrap();
    fs::remove_dir_all(dir).unwrap();
    stdout
}
