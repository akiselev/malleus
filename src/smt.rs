//! SMT translation validation of IR-to-IR passes (SMT-LIB 2, FloatingPoint theory).
//!
//! [`equivalence_query`] encodes the bodies of two kernels with identical operands, maps,
//! domain and scalar type as straight-line terms over one iteration point and asserts that
//! some writable operand ends in a different state; `unsat` therefore proves the bodies
//! equivalent at every point (and hence, the surrounding loop being identical, the kernels).
//!
//! # Encoding
//! - Sort `Float64` or `Float32` from the scalar type; every arithmetic operation rounds
//!   with `RNE`, exactly as the reference interpreter does in that type. `sqrt`, `abs`,
//!   `neg`, `fma`, `floor` (`roundToIntegral RTN`) and `ceil` (`RTP`) are IEEE-specified.
//! - Each loaded operand is one free variable (an operand has one indexing map, so all loads
//!   of it at a point read one element); each `Index(axis)` is a free variable (equivalence
//!   is proved for every value, a stronger statement); constants are exact bit patterns.
//! - Library-dependent operations (`exp`, `ln`, `sin`, `cos`, `tan`, `pow`, `atan2`, `min`,
//!   `max`) are uninterpreted functions named by operation, scalar type and math library, so a
//!   proof holds for any deterministic implementation.
//! - Writable operands start from a free initial value; a store replaces (`Write`) or folds
//!   (`Reduce`) it in statement order. Read-write operands are refused (a load after a store
//!   would observe the store).
//!
//! # Equivalence notion
//! SMT-LIB FloatingPoint has a single NaN, and the query compares final states with `=`, which
//! distinguishes `+0` from `-0` and identifies all NaNs. So `unsat` means: for every input,
//! every output is bit-equal or both are NaN — exactly the bitwise-modulo-NaN-payload notion of
//! the differential tests. (`fp.eq` would identify signed zeros and is not used.)
//!
//! # Declared fused semantics
//! For the L2 contraction pass, [`Semantics::DeclaredContraction`] encodes the *source* kernel
//! under the declared `policy::contract_fma` semantics: at the fixed sites (a binary `Add`/`Sub`
//! whose left, else right, operand is literally a `Mul`, bottom-up, not through locals) the
//! product and sum round once. The pass output is validated against that declaration, not
//! against the unfused kernel (which it legitimately differs from).
//!
//! The solver is an external process ([`check`]); timeouts and `unknown` are typed
//! [`TvOutcome::NotValidated`] results, never success.

use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::policy::MathLibrary;
use crate::{
    AccessMode, BinaryOp, CompareOp, Predicate, ReductionOp, ScalarExpr, ScalarType, Statement,
    StructuredKernel, UnaryOp,
};

/// How the first kernel of a query is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantics {
    /// The reference interpreter's semantics.
    Reference,
    /// Reference semantics with the declared `contract_fma` sites fused (module docs).
    DeclaredContraction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmtError {
    /// The kernels differ outside their bodies (operands, maps, domain or scalar type).
    Mismatch(&'static str),
    /// Read-write operands are not encoded.
    ReadWriteOperand(usize),
    /// The solver could not be run or produced no verdict line.
    Solver(String),
}

impl fmt::Display for SmtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SMT translation validation failed: {self:?}")
    }
}

impl Error for SmtError {}

/// The verdict on a query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TvOutcome {
    /// `unsat`: the kernels are equivalent (module docs).
    Validated,
    /// `sat`: an input on which they differ; `model` is the solver's model text.
    Counterexample { model: String },
    /// `unknown`, `timeout` or any other verdict: nothing is claimed.
    NotValidated { reason: String },
}

/// The SMT-LIB 2 query whose `unsat` proves `source` (under `semantics`) equivalent to
/// `target`.
pub fn equivalence_query(
    source: &StructuredKernel,
    semantics: Semantics,
    target: &StructuredKernel,
) -> Result<String, SmtError> {
    if source.operands != target.operands {
        return Err(SmtError::Mismatch("operands"));
    }
    if source.indexing_maps != target.indexing_maps {
        return Err(SmtError::Mismatch("indexing maps"));
    }
    if source.iteration_domain != target.iteration_domain {
        return Err(SmtError::Mismatch("iteration domain"));
    }
    if source.numeric_policy.scalar_type != target.numeric_policy.scalar_type {
        return Err(SmtError::Mismatch("scalar type"));
    }
    if let Some(index) = source
        .operands
        .iter()
        .position(|operand| operand.access == AccessMode::ReadWrite)
    {
        return Err(SmtError::ReadWriteOperand(index));
    }
    let scalar_type = source.numeric_policy.scalar_type;
    let sort = match scalar_type {
        ScalarType::F64 => "Float64",
        ScalarType::F32 => "Float32",
    };
    let mut q = String::new();
    q.push_str("(set-logic ALL)\n");
    let mut functions = std::collections::BTreeSet::new();
    let mut body = String::new();
    let a = Encoder {
        prefix: "a",
        kernel: source,
        fused: semantics == Semantics::DeclaredContraction,
        functions: &mut functions,
        out: &mut body,
    }
    .encode();
    let b = Encoder {
        prefix: "b",
        kernel: target,
        fused: false,
        functions: &mut functions,
        out: &mut body,
    }
    .encode();
    for (index, _) in source.operands.iter().enumerate() {
        let _ = writeln!(q, "(declare-const in{index} {sort})");
        let _ = writeln!(q, "(declare-const init{index} {sort})");
    }
    for axis in 0..source.iteration_domain.rank() {
        let _ = writeln!(q, "(declare-const idx{axis} {sort})");
    }
    for (name, arity) in &functions {
        let args = vec![sort; *arity].join(" ");
        let _ = writeln!(q, "(declare-fun {name} ({args}) {sort})");
    }
    q.push_str(&body);
    let differences: Vec<String> = a
        .iter()
        .zip(&b)
        .filter_map(|(x, y)| Some(format!("(not (= {} {}))", x.as_ref()?, y.as_ref()?)))
        .collect();
    if differences.is_empty() {
        q.push_str("(assert false)\n");
    } else {
        let _ = writeln!(q, "(assert (or false {}))", differences.join(" "));
    }
    q.push_str("(check-sat)\n(get-model)\n");
    Ok(q)
}

/// Run `z3` (at `solver`) on `query` with a per-query timeout.
pub fn check(solver: &Path, query: &str, timeout: Duration) -> Result<TvOutcome, SmtError> {
    let millis = timeout.as_millis().max(1);
    let hard = timeout.as_secs() + 2;
    let mut child = Command::new(solver)
        .args([
            "-smt2",
            "-in",
            &format!("-t:{millis}"),
            &format!("-T:{hard}"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| SmtError::Solver(format!("cannot run {}: {error}", solver.display())))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(query.as_bytes())
        .map_err(|error| SmtError::Solver(error.to_string()))?;
    let output = child
        .wait_with_output()
        .map_err(|error| SmtError::Solver(error.to_string()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let verdict = lines.next().unwrap_or("").trim().to_owned();
    Ok(match verdict.as_str() {
        "unsat" => TvOutcome::Validated,
        "sat" => TvOutcome::Counterexample {
            model: lines.collect::<Vec<_>>().join("\n"),
        },
        "" => {
            return Err(SmtError::Solver(format!(
                "no verdict; stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        other => TvOutcome::NotValidated {
            reason: other.to_owned(),
        },
    })
}

/// Validate `simplify::simplify_exact` (or any body rewrite) of `before` into `after`.
pub fn validate_rewrite(
    solver: &Path,
    before: &StructuredKernel,
    after: &StructuredKernel,
    timeout: Duration,
) -> Result<TvOutcome, SmtError> {
    check(
        solver,
        &equivalence_query(before, Semantics::Reference, after)?,
        timeout,
    )
}

/// Validate the L2 FMA-insertion pass output against the declared fused semantics of its
/// input. The pass changes the level from `StrictHost` to `Contracted`; both evaluate
/// library operations with the host library, so their uninterpreted functions coincide.
pub fn validate_contraction(
    solver: &Path,
    source: &StructuredKernel,
    contracted: &StructuredKernel,
    timeout: Duration,
) -> Result<TvOutcome, SmtError> {
    check(
        solver,
        &equivalence_query(source, Semantics::DeclaredContraction, contracted)?,
        timeout,
    )
}

struct Encoder<'a> {
    prefix: &'static str,
    kernel: &'a StructuredKernel,
    fused: bool,
    functions: &'a mut std::collections::BTreeSet<(String, usize)>,
    out: &'a mut String,
}

impl Encoder<'_> {
    /// Emit definitions; return the final state term of every operand (`None` for reads).
    fn encode(mut self) -> Vec<Option<String>> {
        let kernel = self.kernel;
        let mut state: Vec<Option<String>> = kernel
            .operands
            .iter()
            .enumerate()
            .map(|(index, operand)| {
                (operand.access != AccessMode::Read).then(|| format!("init{index}"))
            })
            .collect();
        let sort = self.sort();
        let mut locals = 0;
        for (position, statement) in kernel.body.statements.iter().enumerate() {
            match statement {
                Statement::Let { value, .. } => {
                    let term = self.expr(value);
                    let _ = writeln!(
                        self.out,
                        "(define-fun {}_l{locals} () {sort} {term})",
                        self.prefix
                    );
                    locals += 1;
                }
                Statement::Store { operand, value } => {
                    let term = self.expr(value);
                    let index = operand.index();
                    let previous = state[index].clone().expect("validated store target");
                    let updated = match kernel.operands[index].access {
                        AccessMode::Reduce(ReductionOp::Add) => {
                            format!("(fp.add RNE {previous} {term})")
                        }
                        AccessMode::Reduce(ReductionOp::Multiply) => {
                            format!("(fp.mul RNE {previous} {term})")
                        }
                        AccessMode::Reduce(ReductionOp::Min) => self.call("min", &[previous, term]),
                        AccessMode::Reduce(ReductionOp::Max) => self.call("max", &[previous, term]),
                        _ => term,
                    };
                    let name = format!("{}_s{position}", self.prefix);
                    let _ = writeln!(self.out, "(define-fun {name} () {sort} {updated})");
                    state[index] = Some(name);
                }
            }
        }
        state
    }

    fn sort(&self) -> &'static str {
        match self.kernel.numeric_policy.scalar_type {
            ScalarType::F64 => "Float64",
            ScalarType::F32 => "Float32",
        }
    }

    fn call(&mut self, op: &str, args: &[String]) -> String {
        let library = match self.kernel.numeric_policy.level.math() {
            MathLibrary::Host => "host",
            MathLibrary::Portable => "portable",
        };
        let name = format!("{op}_{}_{library}", self.sort());
        self.functions.insert((name.clone(), args.len()));
        format!("({name} {})", args.join(" "))
    }

    fn constant(&self, value: f64) -> String {
        match self.kernel.numeric_policy.scalar_type {
            ScalarType::F64 => format!("((_ to_fp 11 53) #x{:016x})", value.to_bits()),
            ScalarType::F32 => format!("((_ to_fp 8 24) #x{:08x})", (value as f32).to_bits()),
        }
    }

    fn expr(&mut self, expr: &ScalarExpr) -> String {
        match expr {
            ScalarExpr::Constant(value) => self.constant(*value),
            ScalarExpr::Index(axis) => format!("idx{}", axis.index()),
            ScalarExpr::Load(operand) => format!("in{}", operand.index()),
            ScalarExpr::Local(local) => format!("{}_l{}", self.prefix, local.index()),
            ScalarExpr::Unary { op, arg } => {
                let arg = self.expr(arg);
                match op {
                    UnaryOp::Neg => format!("(fp.neg {arg})"),
                    UnaryOp::Abs => format!("(fp.abs {arg})"),
                    UnaryOp::Sqrt => format!("(fp.sqrt RNE {arg})"),
                    UnaryOp::Floor => format!("(fp.roundToIntegral RTN {arg})"),
                    UnaryOp::Ceil => format!("(fp.roundToIntegral RTP {arg})"),
                    UnaryOp::Exp => self.call("exp", &[arg]),
                    UnaryOp::Ln => self.call("ln", &[arg]),
                    UnaryOp::Sin => self.call("sin", &[arg]),
                    UnaryOp::Cos => self.call("cos", &[arg]),
                    UnaryOp::Tan => self.call("tan", &[arg]),
                }
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                if self.fused
                    && matches!(op, BinaryOp::Add | BinaryOp::Sub)
                    && let Some(fused) = self.fused_site(*op, lhs, rhs)
                {
                    return fused;
                }
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                match op {
                    BinaryOp::Add => format!("(fp.add RNE {l} {r})"),
                    BinaryOp::Sub => format!("(fp.sub RNE {l} {r})"),
                    BinaryOp::Mul => format!("(fp.mul RNE {l} {r})"),
                    BinaryOp::Div => format!("(fp.div RNE {l} {r})"),
                    BinaryOp::Pow => self.call("pow", &[l, r]),
                    BinaryOp::Min => self.call("min", &[l, r]),
                    BinaryOp::Max => self.call("max", &[l, r]),
                    BinaryOp::Atan2 => self.call("atan2", &[l, r]),
                }
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let a = self.expr(lhs);
                let b = self.expr(rhs);
                let c = self.expr(addend);
                format!("(fp.fma RNE {a} {b} {c})")
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let c = self.predicate(condition);
                let t = self.expr(if_true);
                let f = self.expr(if_false);
                format!("(ite {c} {t} {f})")
            }
        }
    }

    /// The declared single-rounding meaning of a contraction site: `RN(a·b + c)`,
    /// `RN(a·b − c)` or `RN(c − a·b)`.
    fn fused_site(&mut self, op: BinaryOp, lhs: &ScalarExpr, rhs: &ScalarExpr) -> Option<String> {
        let product = |e: &ScalarExpr| match e {
            ScalarExpr::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            } => Some(((**lhs).clone(), (**rhs).clone())),
            _ => None,
        };
        let (left_product, a, b, other) = match (product(lhs), product(rhs)) {
            (Some((a, b)), _) => (true, a, b, rhs),
            (None, Some((a, b))) => (false, a, b, lhs),
            (None, None) => return None,
        };
        let a = self.expr(&a);
        let b = self.expr(&b);
        let c = self.expr(other);
        Some(match (op, left_product) {
            (BinaryOp::Add, _) => format!("(fp.fma RNE {a} {b} {c})"),
            (_, true) => format!("(fp.fma RNE {a} {b} (fp.neg {c}))"),
            (_, false) => format!("(fp.fma RNE (fp.neg {a}) {b} {c})"),
        })
    }

    fn predicate(&mut self, predicate: &Predicate) -> String {
        match predicate {
            Predicate::Constant(value) => value.to_string(),
            Predicate::Compare { op, lhs, rhs } => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                match op {
                    CompareOp::Eq => format!("(fp.eq {l} {r})"),
                    CompareOp::NotEq => format!("(not (fp.eq {l} {r}))"),
                    CompareOp::Less => format!("(fp.lt {l} {r})"),
                    CompareOp::LessEqual => format!("(fp.leq {l} {r})"),
                    CompareOp::Greater => format!("(fp.gt {l} {r})"),
                    CompareOp::GreaterEqual => format!("(fp.geq {l} {r})"),
                }
            }
            Predicate::Not(value) => format!("(not {})", self.predicate(value)),
            Predicate::And(lhs, rhs) => {
                let l = self.predicate(lhs);
                format!("(and {l} {})", self.predicate(rhs))
            }
            Predicate::Or(lhs, rhs) => {
                let l = self.predicate(lhs);
                format!("(or {l} {})", self.predicate(rhs))
            }
        }
    }
}
