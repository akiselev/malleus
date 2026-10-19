//! FPCore 2.0 export (<https://fpbench.org/spec/fpcore-2.0.html>) for rigorous offline
//! roundoff analysis (FPTaylor, Daisy, Gappa, Herbie).
//!
//! One benchmark per stored output element of the unrolled kernel ([`crate::symbolic`]; the
//! same point and node budgets apply, with the same typed refusals). Each benchmark carries
//! `:name`, `:precision` (`binary64`/`binary32`), a `:pre` built from caller-supplied input
//! boxes, and the source `:malleus-kernel-digest`. Shared subexpressions become `let*`
//! bindings; constants are exact (integers or hexadecimal `0x<m>p<e>`).
//!
//! FPCore operations are correctly rounded. The export is exact for the IEEE-exact operations
//! (`+ - * / sqrt fabs floor ceil fma fmin fmax`, comparisons); for transcendentals it
//! describes the correctly rounded function, which is what an analysis tool models, not the
//! kernel level's libm (`:malleus-level` records which one the kernel uses).

use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt::{self, Write};

use crate::symbolic::{
    ExactConstant, SymbolicError, SymbolicKernel, SymbolicNode, export_symbolic,
};
use crate::{BinaryOp, CompareOp, Executable, OperandId, ScalarType, UnaryOp};

/// A caller-supplied enclosure of input values, emitted as a precondition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputBox {
    pub operand: OperandId,
    /// One buffer element, or every element of the operand when `None` (an element-specific
    /// box takes precedence).
    pub element: Option<usize>,
    /// Infinite bounds leave that side open.
    pub lo: f64,
    pub hi: f64,
}

/// One FPCore benchmark for one output element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FpcoreBenchmark {
    pub operand: OperandId,
    pub element: usize,
    pub source: String,
}

/// A typed refusal of the FPCore export.
#[derive(Clone, Debug, PartialEq)]
pub enum FpcoreError {
    /// The kernel cannot be unrolled (budgets, indices).
    Symbolic(SymbolicError),
    /// A box has a NaN bound or `lo > hi`.
    InvalidBox { index: usize },
}

impl fmt::Display for FpcoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FPCore export: {self:?}")
    }
}

impl Error for FpcoreError {}

/// Unroll `executable` and emit one FPCore benchmark per stored output element.
pub fn export_fpcore(
    executable: &Executable,
    boxes: &[InputBox],
) -> Result<Vec<FpcoreBenchmark>, FpcoreError> {
    let symbolic = export_symbolic(executable).map_err(FpcoreError::Symbolic)?;
    fpcore_from_symbolic(&symbolic, boxes)
}

/// Emit FPCore benchmarks from an already exported DAG.
pub fn fpcore_from_symbolic(
    symbolic: &SymbolicKernel,
    boxes: &[InputBox],
) -> Result<Vec<FpcoreBenchmark>, FpcoreError> {
    symbolic.check().map_err(FpcoreError::Symbolic)?;
    for (index, b) in boxes.iter().enumerate() {
        if b.lo.is_nan() || b.hi.is_nan() || b.lo > b.hi {
            return Err(FpcoreError::InvalidBox { index });
        }
    }
    let names = input_names(symbolic);
    let precision = match symbolic.precision {
        ScalarType::F64 => "binary64",
        ScalarType::F32 => "binary32",
    };
    Ok(symbolic
        .outputs
        .iter()
        .map(|output| {
            let mut printer = Printer {
                nodes: &symbolic.nodes,
                names: &names,
                bound: HashMap::new(),
            };
            let reachable = reachable(symbolic, output.node);
            let mut uses = HashMap::new();
            for &node in &reachable {
                for (arg, _) in symbolic.nodes[node].arguments() {
                    *uses.entry(arg).or_insert(0usize) += 1;
                }
            }
            let mut bindings = String::new();
            for &node in &reachable {
                let leaf = matches!(
                    symbolic.nodes[node],
                    SymbolicNode::Input { .. }
                        | SymbolicNode::Constant(_)
                        | SymbolicNode::Bool { .. }
                );
                if !leaf && uses.get(&node).copied().unwrap_or(0) > 1 {
                    let name = format!("_t{node}");
                    let _ = write!(bindings, "[{name} {}] ", printer.expr(node));
                    printer.bound.insert(node, name);
                }
            }
            let root = printer.expr(output.node);
            let body = if bindings.is_empty() {
                root
            } else {
                format!("(let* ({}) {root})", bindings.trim_end())
            };
            let arguments: Vec<&str> = reachable
                .iter()
                .filter_map(|&n| match symbolic.nodes[n] {
                    SymbolicNode::Input { input } => Some(input),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|i| names[i].as_str())
                .collect();
            let mut pre = vec![];
            for &arg in &arguments {
                let input = names.iter().position(|n| n == arg).expect("argument");
                let i = &symbolic.inputs[input];
                let chosen = boxes
                    .iter()
                    .find(|b| b.operand == i.operand && b.element == Some(i.element))
                    .or_else(|| {
                        boxes
                            .iter()
                            .find(|b| b.operand == i.operand && b.element.is_none())
                    });
                if let Some(b) = chosen {
                    let mut terms = vec![];
                    if b.lo.is_finite() {
                        terms.push(number(b.lo));
                    }
                    terms.push(arg.to_string());
                    if b.hi.is_finite() {
                        terms.push(number(b.hi));
                    }
                    if terms.len() > 1 {
                        pre.push(format!("(<= {})", terms.join(" ")));
                    }
                }
            }
            let mut source = format!(
                "(FPCore ({})\n  :name \"{} {}[{}]\"\n  :precision {precision}\n",
                arguments.join(" "),
                symbolic.kernel.replace(['"', '\\'], "_"),
                output.name.replace(['"', '\\'], "_"),
                output.element
            );
            match pre.len() {
                0 => {}
                1 => {
                    let _ = writeln!(source, "  :pre {}", pre[0]);
                }
                _ => {
                    let _ = writeln!(source, "  :pre (and {})", pre.join(" "));
                }
            }
            let _ = write!(
                source,
                "  :malleus-kernel-digest \"{}:{}\"\n  :malleus-level \"{:?}\"\n  {body})",
                symbolic.kernel_digest.algorithm, symbolic.kernel_digest.hex, symbolic.level
            );
            FpcoreBenchmark {
                operand: output.operand,
                element: output.element,
                source,
            }
        })
        .collect())
}

/// Nodes reachable from `root`, ascending (a topological order).
fn reachable(symbolic: &SymbolicKernel, root: usize) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if seen.insert(n) {
            stack.extend(symbolic.nodes[n].arguments().into_iter().map(|(a, _)| a));
        }
    }
    seen.into_iter().collect()
}

/// FPCore symbols `<operand name>_<element>` over `[A-Za-z0-9_]`, disambiguated by operand
/// index when two operand names sanitize alike. Bindings are `_t<node>` and never collide.
fn input_names(symbolic: &SymbolicKernel) -> Vec<String> {
    let sanitize = |name: &str| {
        let mut s: String = name
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if !s.starts_with(|c: char| c.is_ascii_alphabetic()) {
            s.insert(0, 'v');
        }
        s
    };
    let mut owners: HashMap<String, BTreeSet<OperandId>> = HashMap::new();
    for i in &symbolic.inputs {
        owners
            .entry(sanitize(&i.name))
            .or_default()
            .insert(i.operand);
    }
    symbolic
        .inputs
        .iter()
        .map(|i| {
            let base = sanitize(&i.name);
            if owners[&base].len() > 1 {
                format!("{base}_o{}_{}", i.operand.index(), i.element)
            } else {
                format!("{base}_{}", i.element)
            }
        })
        .collect()
}

/// An exact FPCore number literal.
fn number(value: f64) -> String {
    constant(&ExactConstant::new(value).expect("finite bound"))
}

fn constant(c: &ExactConstant) -> String {
    if c.mantissa == 0 {
        return if c.bits >> 63 == 1 { "(- 0)" } else { "0" }.into();
    }
    let sign = if c.mantissa < 0 { "-" } else { "" };
    let m = c.mantissa.unsigned_abs();
    if (0..=10).contains(&c.exponent) {
        format!("{sign}{}", (m as u128) << c.exponent)
    } else {
        format!("{sign}0x{m:x}p{}", c.exponent)
    }
}

struct Printer<'a> {
    nodes: &'a [SymbolicNode],
    names: &'a [String],
    bound: HashMap<usize, String>,
}

impl Printer<'_> {
    fn expr(&self, node: usize) -> String {
        let e = |n: usize| self.bound.get(&n).cloned().unwrap_or_else(|| self.expr(n));
        match self.nodes[node] {
            SymbolicNode::Input { input } => self.names[input].clone(),
            SymbolicNode::Constant(c) => constant(&c),
            SymbolicNode::Bool { value } => if value { "TRUE" } else { "FALSE" }.into(),
            SymbolicNode::Unary { op, arg } => format!("({} {})", unary(op), e(arg)),
            SymbolicNode::Binary { op, lhs, rhs } => {
                format!("({} {} {})", binary(op), e(lhs), e(rhs))
            }
            SymbolicNode::Fma {
                lhs, rhs, addend, ..
            } => format!("(fma {} {} {})", e(lhs), e(rhs), e(addend)),
            SymbolicNode::Compare { op, lhs, rhs } => {
                format!("({} {} {})", compare(op), e(lhs), e(rhs))
            }
            SymbolicNode::Not { arg } => format!("(not {})", e(arg)),
            SymbolicNode::And { lhs, rhs } => format!("(and {} {})", e(lhs), e(rhs)),
            SymbolicNode::Or { lhs, rhs } => format!("(or {} {})", e(lhs), e(rhs)),
            SymbolicNode::Select {
                condition,
                if_true,
                if_false,
            } => format!("(if {} {} {})", e(condition), e(if_true), e(if_false)),
        }
    }
}

fn unary(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "-",
        UnaryOp::Abs => "fabs",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Exp => "exp",
        UnaryOp::Ln => "log",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Floor => "floor",
        UnaryOp::Ceil => "ceil",
    }
}

fn binary(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Pow => "pow",
        BinaryOp::Min => "fmin",
        BinaryOp::Max => "fmax",
        BinaryOp::Atan2 => "atan2",
    }
}

fn compare(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "==",
        CompareOp::NotEq => "!=",
        CompareOp::Less => "<",
        CompareOp::LessEqual => "<=",
        CompareOp::Greater => ">",
        CompareOp::GreaterEqual => ">=",
    }
}
