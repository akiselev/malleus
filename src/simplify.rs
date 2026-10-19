//! IEEE-exact IR-to-IR simplification.
//!
//! Every rewrite here leaves every output bit unchanged for every input, including signed
//! zeros, infinities and NaNs (NaN payloads are not part of the contract: the reference
//! semantics already treat any NaN as equivalent). All values the reference interpreter
//! produces are already rounded to the kernel's scalar type (loads, constants, indices and
//! every operation result), so the rewrites hold in `f32` kernels too.
//!
//! The rewrites are:
//! - `x * 1 → x`, `1 * x → x`, `x - (+0) → x`, `x + (-0) → x`, `(-0) + x → x`, `-(-x) → x`;
//! - common-subexpression elimination of structurally identical subexpressions (constants
//!   compared by bits) into new `Let` locals. Expressions are pure and total over the
//!   validated domain, so evaluating a shared subexpression once, eagerly, is exact even when
//!   one occurrence sits in an untaken `select` branch. Skipped for kernels with a read-write
//!   operand (a later store could change what a load observes);
//! - dead stores: a store into a `Write` operand that a later store in the same body
//!   overwrites, and stores of the constant `-0` into `Reduce(Add)` operands (`acc + -0 =
//!   acc` exactly) and of `1` into `Reduce(Multiply)` operands.
//!
//! Deliberately not rewritten: `x + 0` (`-0 + 0 = +0`), `x * 0` (NaN, infinities, signs),
//! stores of `+0` into additive reductions (they turn a `-0` accumulator into `+0`),
//! reassociation and constant folding through rounding. `crate::smt` validates these
//! rewrites per kernel with an SMT solver; tests also compare executions bitwise.

use std::collections::HashMap;

use crate::{
    AccessMode, BinaryOp, CompareOp, LocalId, Predicate, ReductionOp, ScalarExpr, Statement,
    StructuredKernel, UnaryOp,
};

/// A simplified kernel and the number of rewrites applied (0 means the kernel is unchanged).
#[derive(Clone, Debug, PartialEq)]
pub struct Simplified {
    pub kernel: StructuredKernel,
    pub rewrites: usize,
}

/// Apply the IEEE-exact rewrites of the module docs. The result has the same operands, maps,
/// domain and policy; only the body changes.
pub fn simplify_exact(kernel: &StructuredKernel) -> Simplified {
    let mut rewrites = 0;
    let mut statements: Vec<Statement> = kernel
        .body
        .statements
        .iter()
        .map(|statement| match statement {
            Statement::Let { local, value } => Statement::Let {
                local: *local,
                value: peephole(value, &mut rewrites),
            },
            Statement::Store { operand, value } => Statement::Store {
                operand: *operand,
                value: peephole(value, &mut rewrites),
            },
        })
        .collect();

    // Dead stores.
    let mut keep = vec![true; statements.len()];
    for (index, statement) in statements.iter().enumerate() {
        let Statement::Store { operand, value } = statement else {
            continue;
        };
        let dead = match kernel.operands[operand.index()].access {
            AccessMode::Write => statements[index + 1..]
                .iter()
                .any(|later| matches!(later, Statement::Store { operand: o, .. } if o == operand)),
            AccessMode::Reduce(ReductionOp::Add) => {
                matches!(value, ScalarExpr::Constant(c) if c.to_bits() == (-0.0f64).to_bits())
            }
            AccessMode::Reduce(ReductionOp::Multiply) => {
                matches!(value, ScalarExpr::Constant(c) if c.to_bits() == 1.0f64.to_bits())
            }
            _ => false,
        };
        if dead {
            keep[index] = false;
            rewrites += 1;
        }
    }
    let mut index = 0;
    statements.retain(|_| {
        index += 1;
        keep[index - 1]
    });

    if !kernel
        .operands
        .iter()
        .any(|operand| operand.access == AccessMode::ReadWrite)
    {
        statements = cse(statements, &mut rewrites);
    }

    let mut simplified = kernel.clone();
    simplified.body.statements = statements;
    Simplified {
        kernel: simplified,
        rewrites,
    }
}

fn is_const(expr: &ScalarExpr, value: f64) -> bool {
    matches!(expr, ScalarExpr::Constant(c) if c.to_bits() == value.to_bits())
}

fn peephole(expr: &ScalarExpr, rewrites: &mut usize) -> ScalarExpr {
    match expr {
        ScalarExpr::Unary { op, arg } => {
            let arg = peephole(arg, rewrites);
            if *op == UnaryOp::Neg
                && let ScalarExpr::Unary {
                    op: UnaryOp::Neg,
                    arg: inner,
                } = arg
            {
                *rewrites += 1;
                return *inner;
            }
            ScalarExpr::unary(*op, arg)
        }
        ScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = peephole(lhs, rewrites);
            let rhs = peephole(rhs, rewrites);
            let keep_lhs = match op {
                BinaryOp::Mul => is_const(&rhs, 1.0),
                BinaryOp::Sub => is_const(&rhs, 0.0),
                BinaryOp::Add => is_const(&rhs, -0.0),
                _ => false,
            };
            let keep_rhs = match op {
                BinaryOp::Mul => is_const(&lhs, 1.0),
                BinaryOp::Add => is_const(&lhs, -0.0),
                _ => false,
            };
            if keep_lhs {
                *rewrites += 1;
                lhs
            } else if keep_rhs {
                *rewrites += 1;
                rhs
            } else {
                ScalarExpr::binary(*op, lhs, rhs)
            }
        }
        ScalarExpr::Fma { lhs, rhs, addend } => ScalarExpr::fma(
            peephole(lhs, rewrites),
            peephole(rhs, rewrites),
            peephole(addend, rewrites),
        ),
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(peephole_predicate(condition, rewrites)),
            if_true: Box::new(peephole(if_true, rewrites)),
            if_false: Box::new(peephole(if_false, rewrites)),
        },
        other => other.clone(),
    }
}

fn peephole_predicate(predicate: &Predicate, rewrites: &mut usize) -> Predicate {
    match predicate {
        Predicate::Constant(value) => Predicate::Constant(*value),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op: *op,
            lhs: Box::new(peephole(lhs, rewrites)),
            rhs: Box::new(peephole(rhs, rewrites)),
        },
        Predicate::Not(value) => Predicate::Not(Box::new(peephole_predicate(value, rewrites))),
        Predicate::And(lhs, rhs) => Predicate::And(
            Box::new(peephole_predicate(lhs, rewrites)),
            Box::new(peephole_predicate(rhs, rewrites)),
        ),
        Predicate::Or(lhs, rhs) => Predicate::Or(
            Box::new(peephole_predicate(lhs, rewrites)),
            Box::new(peephole_predicate(rhs, rewrites)),
        ),
    }
}

/// Hash-consed node keys; locals use the renumbered ids of the output body.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Constant(u64),
    Index(usize),
    Load(usize),
    Local(usize),
    Unary(UnaryOp, usize),
    Binary(BinaryOp, usize, usize),
    Fma(usize, usize, usize),
    Select(usize, usize, usize),
    PConstant(bool),
    Compare(CompareOp, usize, usize),
    Not(usize),
    And(usize, usize),
    Or(usize, usize),
}

impl Key {
    fn shareable(self) -> bool {
        matches!(
            self,
            Key::Unary(..) | Key::Binary(..) | Key::Fma(..) | Key::Select(..)
        )
    }
}

#[derive(Default)]
struct Table {
    ids: HashMap<Key, usize>,
    keys: Vec<Key>,
    exprs: Vec<Option<ScalarExpr>>,
    predicates: Vec<Option<Predicate>>,
    uses: Vec<usize>,
}

impl Table {
    fn intern(&mut self, key: Key) -> usize {
        if let Some(id) = self.ids.get(&key) {
            return *id;
        }
        let id = self.keys.len();
        self.ids.insert(key, id);
        self.keys.push(key);
        self.exprs.push(None);
        self.predicates.push(None);
        self.uses.push(0);
        id
    }

    fn expr(&mut self, expr: &ScalarExpr, locals: &[usize]) -> usize {
        let key = match expr {
            ScalarExpr::Constant(value) => Key::Constant(value.to_bits()),
            ScalarExpr::Index(axis) => Key::Index(axis.index()),
            ScalarExpr::Load(operand) => Key::Load(operand.index()),
            ScalarExpr::Local(local) => Key::Local(locals[local.index()]),
            ScalarExpr::Unary { op, arg } => Key::Unary(*op, self.expr(arg, locals)),
            ScalarExpr::Binary { op, lhs, rhs } => {
                Key::Binary(*op, self.expr(lhs, locals), self.expr(rhs, locals))
            }
            ScalarExpr::Fma { lhs, rhs, addend } => Key::Fma(
                self.expr(lhs, locals),
                self.expr(rhs, locals),
                self.expr(addend, locals),
            ),
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => Key::Select(
                self.predicate(condition, locals),
                self.expr(if_true, locals),
                self.expr(if_false, locals),
            ),
        };
        let id = self.intern(key);
        if self.exprs[id].is_none() {
            self.exprs[id] = Some(expr.clone());
        }
        id
    }

    fn predicate(&mut self, predicate: &Predicate, locals: &[usize]) -> usize {
        let key = match predicate {
            Predicate::Constant(value) => Key::PConstant(*value),
            Predicate::Compare { op, lhs, rhs } => {
                Key::Compare(*op, self.expr(lhs, locals), self.expr(rhs, locals))
            }
            Predicate::Not(value) => Key::Not(self.predicate(value, locals)),
            Predicate::And(lhs, rhs) => {
                Key::And(self.predicate(lhs, locals), self.predicate(rhs, locals))
            }
            Predicate::Or(lhs, rhs) => {
                Key::Or(self.predicate(lhs, locals), self.predicate(rhs, locals))
            }
        };
        let id = self.intern(key);
        if self.predicates[id].is_none() {
            self.predicates[id] = Some(predicate.clone());
        }
        id
    }

    /// Count tree occurrences: every reference from a parent, plus one per statement root.
    fn count(&mut self, id: usize) {
        self.uses[id] += 1;
        if self.uses[id] > 1 {
            return;
        }
        match self.keys[id] {
            Key::Unary(_, a) | Key::Not(a) => self.count(a),
            Key::Binary(_, a, b) | Key::Compare(_, a, b) | Key::And(a, b) | Key::Or(a, b) => {
                self.count(a);
                self.count(b);
            }
            Key::Fma(a, b, c) | Key::Select(a, b, c) => {
                self.count(a);
                self.count(b);
                self.count(c);
            }
            _ => {}
        }
    }
}

struct Rebuild<'a> {
    table: &'a Table,
    bound: HashMap<usize, LocalId>,
    out: Vec<Statement>,
    next_local: usize,
    rewrites: &'a mut usize,
}

impl Rebuild<'_> {
    fn expr(&mut self, id: usize) -> ScalarExpr {
        if let Some(local) = self.bound.get(&id) {
            *self.rewrites += 1;
            return ScalarExpr::Local(*local);
        }
        let built = self.node(id);
        if self.table.keys[id].shareable() && self.table.uses[id] > 1 {
            let local = LocalId::new(self.next_local);
            self.next_local += 1;
            self.out.push(Statement::Let {
                local,
                value: built,
            });
            self.bound.insert(id, local);
            ScalarExpr::Local(local)
        } else {
            built
        }
    }

    fn node(&mut self, id: usize) -> ScalarExpr {
        match self.table.keys[id] {
            Key::Constant(bits) => ScalarExpr::Constant(f64::from_bits(bits)),
            Key::Index(axis) => ScalarExpr::Index(crate::AxisId::new(axis)),
            Key::Load(operand) => ScalarExpr::Load(crate::OperandId::new(operand)),
            Key::Local(local) => ScalarExpr::Local(LocalId::new(local)),
            Key::Unary(op, a) => ScalarExpr::unary(op, self.expr(a)),
            Key::Binary(op, a, b) => {
                let lhs = self.expr(a);
                ScalarExpr::binary(op, lhs, self.expr(b))
            }
            Key::Fma(a, b, c) => {
                let lhs = self.expr(a);
                let rhs = self.expr(b);
                ScalarExpr::fma(lhs, rhs, self.expr(c))
            }
            Key::Select(p, a, b) => {
                let condition = self.predicate(p);
                let if_true = self.expr(a);
                ScalarExpr::Select {
                    condition: Box::new(condition),
                    if_true: Box::new(if_true),
                    if_false: Box::new(self.expr(b)),
                }
            }
            _ => unreachable!("predicate keys are rebuilt by `predicate`"),
        }
    }

    fn predicate(&mut self, id: usize) -> Predicate {
        match self.table.keys[id] {
            Key::PConstant(value) => Predicate::Constant(value),
            Key::Compare(op, a, b) => {
                let lhs = self.expr(a);
                Predicate::Compare {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(self.expr(b)),
                }
            }
            Key::Not(a) => Predicate::Not(Box::new(self.predicate(a))),
            Key::And(a, b) => {
                let lhs = self.predicate(a);
                Predicate::And(Box::new(lhs), Box::new(self.predicate(b)))
            }
            Key::Or(a, b) => {
                let lhs = self.predicate(a);
                Predicate::Or(Box::new(lhs), Box::new(self.predicate(b)))
            }
            _ => unreachable!("expression keys are rebuilt by `expr`"),
        }
    }
}

fn cse(statements: Vec<Statement>, rewrites: &mut usize) -> Vec<Statement> {
    // Pass 1: intern every statement root with locals renumbered in output order. Output
    // local ids are assigned in pass 2; here a primal local is keyed by its own Let root so
    // that structurally equal uses of the same local share a node.
    let mut table = Table::default();
    let mut roots = Vec::new();
    let mut local_roots: Vec<usize> = Vec::new();
    for statement in &statements {
        let (Statement::Let { value, .. } | Statement::Store { value, .. }) = statement;
        let root = table.expr(value, &local_roots);
        if matches!(statement, Statement::Let { .. }) {
            // Key primal locals by a unique placeholder so they never merge with each other.
            local_roots.push(usize::MAX - local_roots.len());
        }
        roots.push(root);
    }
    for root in &roots {
        table.count(*root);
    }

    // Pass 2: rebuild in order, hoisting shared nodes the first time they are needed.
    let mut rebuild = Rebuild {
        table: &table,
        bound: HashMap::new(),
        out: Vec::new(),
        next_local: 0,
        rewrites,
    };
    let mut primal_locals: HashMap<usize, LocalId> = HashMap::new();
    let mut primal_index = 0;
    for (statement, root) in statements.iter().zip(roots) {
        match statement {
            Statement::Let { .. } => {
                let value = rebuild.expr(root);
                let local = LocalId::new(rebuild.next_local);
                rebuild.next_local += 1;
                rebuild.out.push(Statement::Let { local, value });
                primal_locals.insert(usize::MAX - primal_index, local);
                primal_index += 1;
            }
            Statement::Store { operand, .. } => {
                let value = rebuild.expr(root);
                rebuild.out.push(Statement::Store {
                    operand: *operand,
                    value,
                });
            }
        }
    }
    let out = std::mem::take(&mut rebuild.out);
    out.into_iter()
        .map(|statement| match statement {
            Statement::Let { local, value } => Statement::Let {
                local,
                value: relabel(value, &primal_locals),
            },
            Statement::Store { operand, value } => Statement::Store {
                operand,
                value: relabel(value, &primal_locals),
            },
        })
        .collect()
}

/// Replace the placeholder ids of primal locals with their output ids.
fn relabel(expr: ScalarExpr, primal: &HashMap<usize, LocalId>) -> ScalarExpr {
    match expr {
        ScalarExpr::Local(local) => primal
            .get(&local.index())
            .copied()
            .map_or(ScalarExpr::Local(local), ScalarExpr::Local),
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(op, relabel(*arg, primal)),
        ScalarExpr::Binary { op, lhs, rhs } => {
            ScalarExpr::binary(op, relabel(*lhs, primal), relabel(*rhs, primal))
        }
        ScalarExpr::Fma { lhs, rhs, addend } => ScalarExpr::fma(
            relabel(*lhs, primal),
            relabel(*rhs, primal),
            relabel(*addend, primal),
        ),
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => ScalarExpr::Select {
            condition: Box::new(relabel_predicate(*condition, primal)),
            if_true: Box::new(relabel(*if_true, primal)),
            if_false: Box::new(relabel(*if_false, primal)),
        },
        other => other,
    }
}

fn relabel_predicate(predicate: Predicate, primal: &HashMap<usize, LocalId>) -> Predicate {
    match predicate {
        Predicate::Constant(value) => Predicate::Constant(value),
        Predicate::Compare { op, lhs, rhs } => Predicate::Compare {
            op,
            lhs: Box::new(relabel(*lhs, primal)),
            rhs: Box::new(relabel(*rhs, primal)),
        },
        Predicate::Not(value) => Predicate::Not(Box::new(relabel_predicate(*value, primal))),
        Predicate::And(lhs, rhs) => Predicate::And(
            Box::new(relabel_predicate(*lhs, primal)),
            Box::new(relabel_predicate(*rhs, primal)),
        ),
        Predicate::Or(lhs, rhs) => Predicate::Or(
            Box::new(relabel_predicate(*lhs, primal)),
            Box::new(relabel_predicate(*rhs, primal)),
        ),
    }
}
