//! Schedule-independent local-kernel intermediate representation.

/// An iteration-axis identifier local to one kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AxisId(usize);

impl AxisId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// An operand identifier local to one kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperandId(usize);

impl OperandId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// An SSA-like local identifier inside a kernel region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalId(usize);

impl LocalId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

/// Fixed extents for one local invocation domain.
///
/// An empty domain is a scalar point kernel and executes once.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IterationDomain {
    pub extents: Vec<usize>,
}

impl IterationDomain {
    pub fn new(extents: impl Into<Vec<usize>>) -> Self {
        Self {
            extents: extents.into(),
        }
    }

    pub fn rank(&self) -> usize {
        self.extents.len()
    }
}

/// Mathematical role of an iteration axis before target scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IteratorKind {
    Serial,
    Parallel,
    Reduction,
}

/// The reduction applied by a reducing store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOp {
    Add,
    Multiply,
    Min,
    Max,
}

/// Observable memory effect of an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Reduce(ReductionOp),
}

impl AccessMode {
    pub const fn can_read(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub const fn can_write(self) -> bool {
        !matches!(self, Self::Read)
    }
}

/// One externally bound scalar or dense row-major tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelOperand {
    pub name: String,
    pub shape: Vec<usize>,
    pub access: AccessMode,
}

impl KernelOperand {
    pub fn scalar(name: impl Into<String>, access: AccessMode) -> Self {
        Self {
            name: name.into(),
            shape: Vec::new(),
            access,
        }
    }

    pub fn tensor(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        access: AccessMode,
    ) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            access,
        }
    }
}

/// One term in an affine operand index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexTerm {
    pub axis: AxisId,
    pub coefficient: isize,
}

/// An affine index expression `constant + sum(coefficient * axis)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexExpr {
    pub constant: isize,
    pub terms: Vec<IndexTerm>,
}

impl IndexExpr {
    pub fn axis(axis: AxisId) -> Self {
        Self {
            constant: 0,
            terms: vec![IndexTerm {
                axis,
                coefficient: 1,
            }],
        }
    }

    pub fn constant(constant: isize) -> Self {
        Self {
            constant,
            terms: Vec::new(),
        }
    }

    pub fn offset(axis: AxisId, offset: isize) -> Self {
        Self {
            constant: offset,
            terms: vec![IndexTerm {
                axis,
                coefficient: 1,
            }],
        }
    }
}

/// Maps the current iteration coordinates to one operand element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexingMap {
    pub operand: OperandId,
    pub results: Vec<IndexExpr>,
}

impl IndexingMap {
    pub fn scalar(operand: OperandId) -> Self {
        Self {
            operand,
            results: Vec::new(),
        }
    }

    pub fn new(operand: OperandId, results: impl Into<Vec<IndexExpr>>) -> Self {
        Self {
            operand,
            results: results.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmaPolicy {
    Forbidden,
    Allowed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reassociation {
    Forbidden,
    Allowed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOrder {
    Canonical,
    ScheduleDefined,
}

/// Finite-precision choices that are part of executable identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NumericPolicy {
    pub scalar_type: ScalarType,
    pub fma: FmaPolicy,
    pub reassociation: Reassociation,
    pub reduction_order: ReductionOrder,
}

impl Default for NumericPolicy {
    fn default() -> Self {
        Self {
            scalar_type: ScalarType::F64,
            fma: FmaPolicy::Forbidden,
            reassociation: Reassociation::Forbidden,
            reduction_order: ReductionOrder::Canonical,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Atan2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    NotEq,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// A scalar expression evaluated at the current iteration point.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarExpr {
    Constant(f64),
    Index(AxisId),
    Load(OperandId),
    Local(LocalId),
    Unary {
        op: UnaryOp,
        arg: Box<Self>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Select {
        condition: Box<Predicate>,
        if_true: Box<Self>,
        if_false: Box<Self>,
    },
}

impl ScalarExpr {
    pub fn unary(op: UnaryOp, arg: Self) -> Self {
        Self::Unary {
            op,
            arg: Box::new(arg),
        }
    }

    pub fn binary(op: BinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    Constant(bool),
    Compare {
        op: CompareOp,
        lhs: Box<ScalarExpr>,
        rhs: Box<ScalarExpr>,
    },
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

/// A region is ordered: locals are defined once and may only be used by later statements.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Let {
        local: LocalId,
        value: ScalarExpr,
    },
    Store {
        operand: OperandId,
        value: ScalarExpr,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelRegion {
    pub statements: Vec<Statement>,
}

/// Schedule-independent local numerical program.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredKernel {
    pub name: String,
    pub iteration_domain: IterationDomain,
    pub iterators: Vec<IteratorKind>,
    pub operands: Vec<KernelOperand>,
    pub indexing_maps: Vec<IndexingMap>,
    pub body: KernelRegion,
    pub numeric_policy: NumericPolicy,
}

/// A backend-independent collection of local kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredModule {
    pub name: String,
    pub kernels: Vec<StructuredKernel>,
}

/// The derivative program a backend is asked to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeMode {
    Jvp,
    Vjp,
    Jacobian,
}

/// Backend-neutral differentiation request. Malleus backends may reject modes they do not
/// implement; the reference interpreter executes primal kernels only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivativeRequest {
    pub mode: DerivativeMode,
    pub independent_operands: Vec<OperandId>,
    pub dependent_operands: Vec<OperandId>,
}
