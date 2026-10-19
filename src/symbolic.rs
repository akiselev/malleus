//! Neutral symbolic export: a dependency-free, serializable expression DAG per output element.
//!
//! [`export_symbolic`] unrolls a kernel's whole iteration domain (in the reference
//! interpreter's traversal order, within [`MAX_POINTS`] and [`MAX_NODES`]) into one shared DAG
//! of [`SymbolicNode`]s over named input symbols. Consumers (Resolvent, via Scientia) import it
//! without Malleus depending on them, and check it with [`SymbolicKernel::check`].
//!
//! # Semantics (schema [`SYMBOLIC_SCHEMA`])
//!
//! - Nodes are topologically ordered: every argument index is smaller than the node's own.
//!   Value nodes denote numbers; `bool`, `compare`, `not`, `and`, `or` denote predicates.
//! - `input` denotes the value of `inputs[input]`: the element `element` of operand `operand`'s
//!   bound buffer before the call (for `reduce` and `read_write` operands, the initial value
//!   the kernel accumulates onto). Binary32 kernels see it rounded to binary32.
//! - `constant` is exact: `mantissa · 2^exponent` (mantissa odd, or zero with exponent 0) and
//!   `bits` are the binary64 bits of the same value (the sign of zero lives only in `bits`).
//!   Iteration coordinates appear as constants after unrolling.
//! - `unary`/`binary` use the IR operation names (`UnaryOp`/`BinaryOp`, snake case;
//!   `atan2(lhs, rhs)` is `lhs.atan2(rhs)`). In real semantics they are the real functions
//!   (partial where undefined); in the kernel's floating semantics each such node is one
//!   rounding to `precision`, evaluated by the `level`'s math library.
//! - `fma` is the real `lhs·rhs + addend`; `single_rounding: true` records that the kernel
//!   rounds it once (an explicit IR `Fma`).
//! - `select` is a lazy piecewise value: only the branch chosen by `condition` is evaluated.
//!   `compare` is the IEEE comparison (ordered comparisons are false on NaN, `not_eq` true).
//! - `outputs` lists every element stored at least once, ascending by operand then element,
//!   with the node of its final value; reducing stores appear as `binary` nodes
//!   `add`/`mul`/`min`/`max` of the previous value and the stored value, in traversal order.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::interpreter::{cast, offset};
use crate::policy::NumericLevel;
use crate::{
    AccessMode, AxisId, BinaryOp, CompareOp, Digest, Executable, OperandId, Predicate, ReductionOp,
    ReductionOrder, ScalarExpr, ScalarType, Statement, StructuredKernel, UnaryOp, kernel_digest,
};

/// Schema identifier carried by every [`SymbolicKernel`].
pub const SYMBOLIC_SCHEMA: &str = "malleus-symbolic/1";
/// Largest iteration domain (points) that is unrolled.
pub const MAX_POINTS: usize = 4096;
/// Largest DAG (distinct nodes) that is built.
pub const MAX_NODES: usize = 1 << 16;

/// An exactly represented finite constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExactConstant {
    /// Binary64 bits of the constant (for binary32 kernels, of the binary32-rounded value).
    pub bits: u64,
    /// Odd (or zero) integer mantissa of the exact value `mantissa · 2^exponent`.
    pub mantissa: i64,
    pub exponent: i32,
}

impl ExactConstant {
    /// The exact decomposition of a finite double; `None` for infinities and NaN.
    pub fn new(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let field = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1 << 52) - 1);
        let (mut m, mut e) = if field == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), field - 1075)
        };
        if m == 0 {
            e = 0;
        } else {
            let shift = m.trailing_zeros();
            m >>= shift;
            e += shift as i32;
        }
        let mantissa = if bits >> 63 == 1 {
            -(m as i64)
        } else {
            m as i64
        };
        Some(Self {
            bits,
            mantissa,
            exponent: e,
        })
    }

    pub fn value(&self) -> f64 {
        f64::from_bits(self.bits)
    }
}

/// One DAG node; arguments are indices of earlier nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum SymbolicNode {
    Input {
        input: usize,
    },
    Constant(ExactConstant),
    Unary {
        op: UnaryOp,
        arg: usize,
    },
    Binary {
        op: BinaryOp,
        lhs: usize,
        rhs: usize,
    },
    Fma {
        lhs: usize,
        rhs: usize,
        addend: usize,
        single_rounding: bool,
    },
    Bool {
        value: bool,
    },
    Compare {
        op: CompareOp,
        lhs: usize,
        rhs: usize,
    },
    Not {
        arg: usize,
    },
    And {
        lhs: usize,
        rhs: usize,
    },
    Or {
        lhs: usize,
        rhs: usize,
    },
    Select {
        condition: usize,
        if_true: usize,
        if_false: usize,
    },
}

impl SymbolicNode {
    /// Whether the node denotes a predicate rather than a value.
    pub fn is_predicate(&self) -> bool {
        matches!(
            self,
            Self::Bool { .. }
                | Self::Compare { .. }
                | Self::Not { .. }
                | Self::And { .. }
                | Self::Or { .. }
        )
    }

    /// Arguments with whether each must be a predicate.
    pub fn arguments(&self) -> Vec<(usize, bool)> {
        match *self {
            Self::Input { .. } | Self::Constant(_) | Self::Bool { .. } => vec![],
            Self::Unary { arg, .. } => vec![(arg, false)],
            Self::Binary { lhs, rhs, .. } | Self::Compare { lhs, rhs, .. } => {
                vec![(lhs, false), (rhs, false)]
            }
            Self::Fma {
                lhs, rhs, addend, ..
            } => vec![(lhs, false), (rhs, false), (addend, false)],
            Self::Not { arg } => vec![(arg, true)],
            Self::And { lhs, rhs } | Self::Or { lhs, rhs } => vec![(lhs, true), (rhs, true)],
            Self::Select {
                condition,
                if_true,
                if_false,
            } => vec![(condition, true), (if_true, false), (if_false, false)],
        }
    }
}

/// A named input symbol: one element of one operand's buffer before the call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicInput {
    pub operand: OperandId,
    /// Index into the bound buffer (region offset included).
    pub element: usize,
    /// The operand's name.
    pub name: String,
}

/// The final value of one stored output element.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicOutput {
    pub operand: OperandId,
    /// Index into the bound buffer (region offset included).
    pub element: usize,
    pub node: usize,
    /// The operand's name.
    pub name: String,
}

/// The exported DAG of one kernel. See the module documentation for the semantics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicKernel {
    pub schema: String,
    pub kernel: String,
    pub kernel_digest: Digest,
    pub precision: ScalarType,
    pub level: NumericLevel,
    pub inputs: Vec<SymbolicInput>,
    pub nodes: Vec<SymbolicNode>,
    pub outputs: Vec<SymbolicOutput>,
}

/// A typed refusal of [`export_symbolic`] or a failed [`SymbolicKernel::check`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicError {
    /// The iteration domain has more than [`MAX_POINTS`] points.
    PointBudget { points: Option<usize> },
    /// The DAG would exceed [`MAX_NODES`] nodes.
    NodeBudget,
    /// An operand index left its buffer region at some iteration point.
    InvalidIndex { operand: usize },
    /// Structural check: unknown schema.
    Schema(String),
    /// Structural check: an argument does not name an earlier node of the right kind.
    InvalidArgument { node: usize },
    /// Structural check: an input node names no input.
    InvalidInput { node: usize },
    /// Structural check: an output names no value node.
    InvalidOutput { output: usize },
}

impl fmt::Display for SymbolicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "symbolic export: {self:?}")
    }
}

impl Error for SymbolicError {}

impl SymbolicKernel {
    /// Structural validation for consumers that decode a serialized export.
    pub fn check(&self) -> Result<(), SymbolicError> {
        if self.schema != SYMBOLIC_SCHEMA {
            return Err(SymbolicError::Schema(self.schema.clone()));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if let SymbolicNode::Input { input } = node
                && *input >= self.inputs.len()
            {
                return Err(SymbolicError::InvalidInput { node: i });
            }
            for (arg, predicate) in node.arguments() {
                if arg >= i || self.nodes[arg].is_predicate() != predicate {
                    return Err(SymbolicError::InvalidArgument { node: i });
                }
            }
        }
        for (i, output) in self.outputs.iter().enumerate() {
            if self
                .nodes
                .get(output.node)
                .is_none_or(SymbolicNode::is_predicate)
            {
                return Err(SymbolicError::InvalidOutput { output: i });
            }
        }
        Ok(())
    }
}

/// Unroll a kernel into its symbolic DAG (see the module documentation).
pub fn export_symbolic(executable: &Executable) -> Result<SymbolicKernel, SymbolicError> {
    let kernel = executable.kernel().as_kernel();
    let extents = &kernel.iteration_domain.extents;
    let points = extents
        .iter()
        .try_fold(1usize, |n, &e| n.checked_mul(e))
        .filter(|&n| n <= MAX_POINTS)
        .ok_or(SymbolicError::PointBudget {
            points: extents.iter().try_fold(1usize, |n, &e| n.checked_mul(e)),
        })?;
    let order: Vec<AxisId> = if kernel.numeric_policy.reduction_order == ReductionOrder::Canonical {
        (0..extents.len()).map(AxisId::new).collect()
    } else {
        executable.schedule().loop_order.clone()
    };
    let mut builder = Builder {
        kernel,
        nodes: vec![],
        dedup: HashMap::new(),
        inputs: vec![],
        input_nodes: HashMap::new(),
        state: BTreeMap::new(),
    };
    let mut coordinates = vec![0; extents.len()];
    for _ in 0..points {
        builder.point(&coordinates)?;
        // Advance the innermost axis of the traversal order first.
        for axis in order.iter().rev() {
            let a = axis.index();
            coordinates[a] += 1;
            if coordinates[a] < extents[a] {
                break;
            }
            coordinates[a] = 0;
        }
    }
    let outputs = builder
        .state
        .iter()
        .map(|(&(operand, element), &node)| SymbolicOutput {
            operand: OperandId::new(operand),
            element,
            node,
            name: kernel.operands[operand].name.clone(),
        })
        .collect();
    Ok(SymbolicKernel {
        schema: SYMBOLIC_SCHEMA.into(),
        kernel: kernel.name.clone(),
        kernel_digest: kernel_digest(kernel),
        precision: kernel.numeric_policy.scalar_type,
        level: kernel.numeric_policy.level,
        inputs: builder.inputs,
        nodes: builder.nodes,
        outputs,
    })
}

struct Builder<'k> {
    kernel: &'k StructuredKernel,
    nodes: Vec<SymbolicNode>,
    dedup: HashMap<SymbolicNode, usize>,
    inputs: Vec<SymbolicInput>,
    input_nodes: HashMap<(usize, usize), usize>,
    /// Current value of every element written so far.
    state: BTreeMap<(usize, usize), usize>,
}

impl Builder<'_> {
    fn node(&mut self, node: SymbolicNode) -> Result<usize, SymbolicError> {
        if let Some(&id) = self.dedup.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= MAX_NODES {
            return Err(SymbolicError::NodeBudget);
        }
        self.nodes.push(node);
        self.dedup.insert(node, self.nodes.len() - 1);
        Ok(self.nodes.len() - 1)
    }

    fn constant(&mut self, value: f64) -> Result<usize, SymbolicError> {
        let value = cast(value, self.kernel.numeric_policy.scalar_type);
        // Validation admits only finite constants; finite casts of indices stay finite.
        let constant = ExactConstant::new(value).expect("finite constant");
        self.node(SymbolicNode::Constant(constant))
    }

    fn element(&self, operand: OperandId, coordinates: &[usize]) -> Result<usize, SymbolicError> {
        let k = self.kernel;
        let definition = &k.operands[operand.index()];
        let map = k
            .indexing_maps
            .iter()
            .find(|map| map.operand == operand)
            .expect("validated map");
        offset(
            &map.results,
            &definition.shape,
            &definition.layout.minor_to_major,
            coordinates,
        )
        .and_then(|o| definition.region.offset.checked_add(o))
        .ok_or(SymbolicError::InvalidIndex {
            operand: operand.index(),
        })
    }

    fn read(&mut self, operand: usize, element: usize) -> Result<usize, SymbolicError> {
        if let Some(&node) = self.state.get(&(operand, element)) {
            return Ok(node);
        }
        if let Some(&node) = self.input_nodes.get(&(operand, element)) {
            return Ok(node);
        }
        self.inputs.push(SymbolicInput {
            operand: OperandId::new(operand),
            element,
            name: self.kernel.operands[operand].name.clone(),
        });
        let node = self.node(SymbolicNode::Input {
            input: self.inputs.len() - 1,
        })?;
        self.input_nodes.insert((operand, element), node);
        Ok(node)
    }

    fn point(&mut self, coordinates: &[usize]) -> Result<(), SymbolicError> {
        let mut locals = vec![];
        for statement in &self.kernel.body.statements {
            match statement {
                Statement::Let { value, .. } => {
                    let v = self.value(value, coordinates, &locals)?;
                    locals.push(v);
                }
                Statement::Store { operand, value } => {
                    let v = self.value(value, coordinates, &locals)?;
                    let element = self.element(*operand, coordinates)?;
                    let key = (operand.index(), element);
                    let v = match self.kernel.operands[operand.index()].access {
                        AccessMode::Reduce(reduction) => {
                            let op = match reduction {
                                ReductionOp::Add => BinaryOp::Add,
                                ReductionOp::Multiply => BinaryOp::Mul,
                                ReductionOp::Min => BinaryOp::Min,
                                ReductionOp::Max => BinaryOp::Max,
                            };
                            let previous = self.read(key.0, key.1)?;
                            self.node(SymbolicNode::Binary {
                                op,
                                lhs: previous,
                                rhs: v,
                            })?
                        }
                        _ => v,
                    };
                    self.state.insert(key, v);
                }
            }
        }
        Ok(())
    }

    fn value(
        &mut self,
        e: &ScalarExpr,
        coordinates: &[usize],
        locals: &[usize],
    ) -> Result<usize, SymbolicError> {
        Ok(match e {
            ScalarExpr::Constant(x) => self.constant(*x)?,
            ScalarExpr::Index(axis) => self.constant(coordinates[axis.index()] as f64)?,
            ScalarExpr::Local(id) => locals[id.index()],
            ScalarExpr::Load(operand) => {
                let element = self.element(*operand, coordinates)?;
                self.read(operand.index(), element)?
            }
            ScalarExpr::Unary { op, arg } => {
                let arg = self.value(arg, coordinates, locals)?;
                self.node(SymbolicNode::Unary { op: *op, arg })?
            }
            ScalarExpr::Binary { op, lhs, rhs } => {
                let lhs = self.value(lhs, coordinates, locals)?;
                let rhs = self.value(rhs, coordinates, locals)?;
                self.node(SymbolicNode::Binary { op: *op, lhs, rhs })?
            }
            ScalarExpr::Fma { lhs, rhs, addend } => {
                let lhs = self.value(lhs, coordinates, locals)?;
                let rhs = self.value(rhs, coordinates, locals)?;
                let addend = self.value(addend, coordinates, locals)?;
                self.node(SymbolicNode::Fma {
                    lhs,
                    rhs,
                    addend,
                    single_rounding: true,
                })?
            }
            ScalarExpr::Select {
                condition,
                if_true,
                if_false,
            } => {
                let condition = self.predicate(condition, coordinates, locals)?;
                let if_true = self.value(if_true, coordinates, locals)?;
                let if_false = self.value(if_false, coordinates, locals)?;
                self.node(SymbolicNode::Select {
                    condition,
                    if_true,
                    if_false,
                })?
            }
        })
    }

    fn predicate(
        &mut self,
        p: &Predicate,
        coordinates: &[usize],
        locals: &[usize],
    ) -> Result<usize, SymbolicError> {
        let node = match p {
            Predicate::Constant(value) => SymbolicNode::Bool { value: *value },
            Predicate::Compare { op, lhs, rhs } => SymbolicNode::Compare {
                op: *op,
                lhs: self.value(lhs, coordinates, locals)?,
                rhs: self.value(rhs, coordinates, locals)?,
            },
            Predicate::Not(arg) => SymbolicNode::Not {
                arg: self.predicate(arg, coordinates, locals)?,
            },
            Predicate::And(lhs, rhs) => SymbolicNode::And {
                lhs: self.predicate(lhs, coordinates, locals)?,
                rhs: self.predicate(rhs, coordinates, locals)?,
            },
            Predicate::Or(lhs, rhs) => SymbolicNode::Or {
                lhs: self.predicate(lhs, coordinates, locals)?,
                rhs: self.predicate(rhs, coordinates, locals)?,
            },
        };
        self.node(node)
    }
}
