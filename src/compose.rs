//! Kernel compositions: execution-time binding of one kernel's output buffer to another
//! kernel's input operand, without rewriting either kernel.
//!
//! A [`KernelComposition`] is an ordered list of *stages* (ordinary [`StructuredKernel`]s, held
//! verbatim) plus [`SharedBuffer`] groups. Every member of a group aliases one buffer at
//! execution time: the stage that writes the buffer runs before the stages that read it, so a
//! producer kernel's output becomes a consumer kernel's input with no expression rewriting. The
//! composition's identity is built from the stage kernel digests and the sharing table, so the
//! stage kernels keep their standalone identity exactly.
//!
//! Derivatives of a composition are compositions of per-stage derivative kernels: the JVP shares
//! the producer's tangent output with the consumer's direction input, and the VJP shares the
//! consumer's cotangent output with the producer's seed input. Nothing is fused.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::digest::{Digest, kernel_digest};
use crate::{
    AccessMode, BufferBinding, DerivativeMode, DerivativeOperand, DerivativeRequest,
    DifferentiationError, Executable, ExecutionError, Interpreter, OperandId, ReductionOp,
    StructuredKernel, ValidatedKernel, ValidationError, differentiate, validate,
};

/// Schema of the payload behind [`composition_digest`].
pub const KERNEL_COMPOSITION_SCHEMA: &str = "malleus-kernel-composition/1";

/// One operand of one stage in a composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StageOperand {
    pub stage: usize,
    pub operand: OperandId,
}

impl StageOperand {
    pub const fn new(stage: usize, operand: OperandId) -> Self {
        Self { stage, operand }
    }
}

/// One buffer aliased by several stage operands.
///
/// Members must agree on shape, layout, and buffer region. A group has either exactly one
/// `Write` member or one or more `Reduce` members sharing one reduction, plus at least one
/// `Read` member; every writing stage precedes every reading stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedBuffer {
    pub members: Vec<StageOperand>,
}

impl SharedBuffer {
    pub fn new(members: impl Into<Vec<StageOperand>>) -> Self {
        Self {
            members: members.into(),
        }
    }

    /// The common two-member case: `producer`'s output feeds `consumer`'s input.
    pub fn bind(producer: StageOperand, consumer: StageOperand) -> Self {
        Self {
            members: vec![producer, consumer],
        }
    }
}

/// Schedule-independent composition of verbatim kernels over shared buffers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelComposition {
    pub name: String,
    pub stages: Vec<StructuredKernel>,
    pub shared_buffers: Vec<SharedBuffer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompositionError {
    EmptyName,
    NoStages,
    InvalidStage {
        stage: usize,
        error: ValidationError,
    },
    DuplicateStageName(String),
    TooFewMembers(usize),
    InvalidMember {
        buffer: usize,
        member: StageOperand,
    },
    DuplicateMember {
        buffer: usize,
        member: StageOperand,
    },
    MemberShape {
        buffer: usize,
        member: StageOperand,
    },
    UnsupportedReadWrite {
        buffer: usize,
        member: StageOperand,
    },
    NoWriter(usize),
    NoReader(usize),
    ConflictingWriters(usize),
    MultipleWritersInStage {
        buffer: usize,
        stage: usize,
    },
    WriterAfterReader {
        buffer: usize,
        writer: StageOperand,
        reader: StageOperand,
    },
}

impl fmt::Display for CompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid kernel composition: {self:?}")
    }
}

impl Error for CompositionError {}

#[derive(Clone, Debug, PartialEq)]
struct SharedBufferPlan {
    members: Vec<StageOperand>,
    writers: Vec<StageOperand>,
    readers: Vec<StageOperand>,
    /// Elements every member's binding must provide: `region.offset + region.length`.
    length: usize,
}

/// A composition whose stages and sharing table passed [`validate_composition`].
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedComposition {
    name: String,
    stages: Vec<ValidatedKernel>,
    shared: Vec<SharedBufferPlan>,
    membership: BTreeMap<StageOperand, usize>,
}

impl ValidatedComposition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn stages(&self) -> &[ValidatedKernel] {
        &self.stages
    }

    /// The shared buffer this operand aliases, if any.
    pub fn shared_buffer_of(&self, operand: StageOperand) -> Option<usize> {
        self.membership.get(&operand).copied()
    }

    /// Members of one shared buffer in declaration order.
    pub fn shared_members(&self, buffer: usize) -> Option<&[StageOperand]> {
        self.shared.get(buffer).map(|plan| plan.members.as_slice())
    }

    /// The elements a caller-supplied binding for shared buffer `buffer` must provide.
    pub fn shared_length(&self, buffer: usize) -> Option<usize> {
        self.shared.get(buffer).map(|plan| plan.length)
    }

    pub fn into_composition(self) -> KernelComposition {
        KernelComposition {
            name: self.name,
            stages: self
                .stages
                .into_iter()
                .map(ValidatedKernel::into_inner)
                .collect(),
            shared_buffers: self
                .shared
                .into_iter()
                .map(|plan| SharedBuffer {
                    members: plan.members,
                })
                .collect(),
        }
    }
}

pub fn validate_composition(
    composition: KernelComposition,
) -> Result<ValidatedComposition, CompositionError> {
    if composition.name.trim().is_empty() {
        return Err(CompositionError::EmptyName);
    }
    if composition.stages.is_empty() {
        return Err(CompositionError::NoStages);
    }
    let mut names = BTreeSet::new();
    let mut stages = Vec::with_capacity(composition.stages.len());
    for (index, kernel) in composition.stages.into_iter().enumerate() {
        if !names.insert(kernel.name.clone()) {
            return Err(CompositionError::DuplicateStageName(kernel.name));
        }
        stages.push(
            validate(kernel).map_err(|error| CompositionError::InvalidStage {
                stage: index,
                error,
            })?,
        );
    }
    let mut membership = BTreeMap::new();
    let mut shared = Vec::with_capacity(composition.shared_buffers.len());
    for (buffer, group) in composition.shared_buffers.into_iter().enumerate() {
        if group.members.len() < 2 {
            return Err(CompositionError::TooFewMembers(buffer));
        }
        let mut writers = Vec::new();
        let mut readers = Vec::new();
        let mut writer_stages = BTreeSet::new();
        let mut reduction: Option<ReductionOp> = None;
        let mut reference: Option<&crate::KernelOperand> = None;
        for member in &group.members {
            let definition = stages
                .get(member.stage)
                .and_then(|stage| stage.as_kernel().operands.get(member.operand.index()))
                .ok_or(CompositionError::InvalidMember {
                    buffer,
                    member: *member,
                })?;
            if membership.insert(*member, buffer).is_some() {
                return Err(CompositionError::DuplicateMember {
                    buffer,
                    member: *member,
                });
            }
            match reference {
                None => reference = Some(definition),
                Some(first)
                    if first.shape != definition.shape
                        || first.layout != definition.layout
                        || first.region != definition.region =>
                {
                    return Err(CompositionError::MemberShape {
                        buffer,
                        member: *member,
                    });
                }
                Some(_) => {}
            }
            match definition.access {
                AccessMode::Read => readers.push(*member),
                AccessMode::ReadWrite => {
                    return Err(CompositionError::UnsupportedReadWrite {
                        buffer,
                        member: *member,
                    });
                }
                AccessMode::Write => {
                    if !writers.is_empty() {
                        return Err(CompositionError::ConflictingWriters(buffer));
                    }
                    writers.push(*member);
                    writer_stages.insert(member.stage);
                }
                AccessMode::Reduce(op) => {
                    if writers.iter().any(|writer| {
                        matches!(
                            stages[writer.stage].as_kernel().operands[writer.operand.index()]
                                .access,
                            AccessMode::Write
                        )
                    }) || reduction.is_some_and(|previous| previous != op)
                    {
                        return Err(CompositionError::ConflictingWriters(buffer));
                    }
                    reduction = Some(op);
                    if !writer_stages.insert(member.stage) {
                        return Err(CompositionError::MultipleWritersInStage {
                            buffer,
                            stage: member.stage,
                        });
                    }
                    writers.push(*member);
                }
            }
        }
        if writers.is_empty() {
            return Err(CompositionError::NoWriter(buffer));
        }
        if readers.is_empty() {
            return Err(CompositionError::NoReader(buffer));
        }
        for writer in &writers {
            if let Some(reader) = readers.iter().find(|reader| reader.stage <= writer.stage) {
                return Err(CompositionError::WriterAfterReader {
                    buffer,
                    writer: *writer,
                    reader: *reader,
                });
            }
        }
        let region = reference.expect("a member exists").region;
        shared.push(SharedBufferPlan {
            members: group.members,
            writers,
            readers,
            length: region.offset + region.length,
        });
    }
    Ok(ValidatedComposition {
        name: composition.name,
        stages,
        shared,
        membership,
    })
}

#[derive(Serialize)]
struct CompositionPayload<'a> {
    schema: &'static str,
    name: &'a str,
    stages: Vec<Digest>,
    shared_buffers: &'a [SharedBuffer],
}

/// Identity of a composition: the ordered stage kernel digests plus the sharing table. Stage
/// kernels enter by digest only, so composing never changes what they are.
pub fn composition_digest(composition: &KernelComposition) -> Digest {
    Digest::of_payload(&CompositionPayload {
        schema: KERNEL_COMPOSITION_SCHEMA,
        name: &composition.name,
        stages: composition.stages.iter().map(kernel_digest).collect(),
        shared_buffers: &composition.shared_buffers,
    })
}

/// A validated composition paired with one reference executable per stage.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableComposition {
    name: String,
    stages: Vec<Executable>,
    shared: Vec<SharedBufferPlan>,
    membership: BTreeMap<StageOperand, usize>,
}

impl ExecutableComposition {
    /// Canonical reference schedules for every stage.
    pub fn reference(composition: ValidatedComposition) -> Self {
        Self {
            name: composition.name,
            stages: composition
                .stages
                .into_iter()
                .map(Executable::reference)
                .collect(),
            shared: composition.shared,
            membership: composition.membership,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn stages(&self) -> &[Executable] {
        &self.stages
    }

    pub fn shared_buffer_of(&self, operand: StageOperand) -> Option<usize> {
        self.membership.get(&operand).copied()
    }

    pub fn shared_length(&self, buffer: usize) -> Option<usize> {
        self.shared.get(buffer).map(|plan| plan.length)
    }
}

/// What one caller-supplied buffer binds: an unshared stage operand, or a whole shared buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompositionTarget {
    Operand(StageOperand),
    Shared(usize),
}

pub struct CompositionBinding<'a> {
    pub target: CompositionTarget,
    pub values: &'a mut [f64],
}

impl<'a> CompositionBinding<'a> {
    pub fn operand(operand: StageOperand, values: &'a mut [f64]) -> Self {
        Self {
            target: CompositionTarget::Operand(operand),
            values,
        }
    }

    pub fn shared(buffer: usize, values: &'a mut [f64]) -> Self {
        Self {
            target: CompositionTarget::Shared(buffer),
            values,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompositionExecutionError {
    MissingBinding(StageOperand),
    DuplicateBinding(CompositionTarget),
    InvalidBinding(CompositionTarget),
    SharedOperandBound(StageOperand),
    Stage { stage: usize, error: ExecutionError },
}

impl fmt::Display for CompositionExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel composition execution failed: {self:?}")
    }
}

impl Error for CompositionExecutionError {}

enum BufferSource {
    Caller(usize),
    Scratch(usize),
}

impl Interpreter {
    /// Execute every stage in order against caller-owned buffers.
    ///
    /// The caller binds every stage operand that is not a member of a shared buffer, exactly
    /// once. A shared buffer may be bound as a whole (`CompositionTarget::Shared`) to observe or
    /// seed it; an unbound shared buffer is a zero-initialized scratch buffer. Binding a shared
    /// member operand directly is refused. Reduce-access members accumulate into the shared
    /// buffer, so a caller-supplied reduce target carries the initial value as for one kernel.
    pub fn run_composition(
        executable: &ExecutableComposition,
        bindings: &mut [CompositionBinding<'_>],
    ) -> Result<(), CompositionExecutionError> {
        let mut by_operand = BTreeMap::new();
        let mut by_shared = BTreeMap::new();
        for (index, binding) in bindings.iter().enumerate() {
            match binding.target {
                CompositionTarget::Operand(operand) => {
                    if executable.membership.contains_key(&operand) {
                        return Err(CompositionExecutionError::SharedOperandBound(operand));
                    }
                    let definition = executable
                        .stages
                        .get(operand.stage)
                        .and_then(|stage| {
                            stage
                                .kernel()
                                .as_kernel()
                                .operands
                                .get(operand.operand.index())
                        })
                        .ok_or(CompositionExecutionError::InvalidBinding(binding.target))?;
                    if binding.values.len() < definition.region.offset + definition.region.length {
                        return Err(CompositionExecutionError::InvalidBinding(binding.target));
                    }
                    if by_operand.insert(operand, index).is_some() {
                        return Err(CompositionExecutionError::DuplicateBinding(binding.target));
                    }
                }
                CompositionTarget::Shared(buffer) => {
                    let plan = executable
                        .shared
                        .get(buffer)
                        .ok_or(CompositionExecutionError::InvalidBinding(binding.target))?;
                    if binding.values.len() < plan.length {
                        return Err(CompositionExecutionError::InvalidBinding(binding.target));
                    }
                    if by_shared.insert(buffer, index).is_some() {
                        return Err(CompositionExecutionError::DuplicateBinding(binding.target));
                    }
                }
            }
        }
        let mut scratch = Vec::with_capacity(executable.shared.len());
        for (buffer, plan) in executable.shared.iter().enumerate() {
            scratch.push(if by_shared.contains_key(&buffer) {
                Vec::new()
            } else {
                vec![0.0; plan.length]
            });
        }
        let mut sources = Vec::with_capacity(executable.stages.len());
        for (stage_index, stage) in executable.stages.iter().enumerate() {
            let kernel = stage.kernel().as_kernel();
            let mut stage_sources = Vec::with_capacity(kernel.operands.len());
            for operand_index in 0..kernel.operands.len() {
                let operand = StageOperand::new(stage_index, OperandId::new(operand_index));
                let source = match executable.membership.get(&operand) {
                    Some(buffer) => match by_shared.get(buffer) {
                        Some(binding) => BufferSource::Caller(*binding),
                        None => BufferSource::Scratch(*buffer),
                    },
                    None => BufferSource::Caller(
                        *by_operand
                            .get(&operand)
                            .ok_or(CompositionExecutionError::MissingBinding(operand))?,
                    ),
                };
                stage_sources.push(source);
            }
            sources.push(stage_sources);
        }
        for (stage_index, stage) in executable.stages.iter().enumerate() {
            let kernel = stage.kernel().as_kernel();
            let mut buffers = Vec::with_capacity(kernel.operands.len());
            for (operand_index, definition) in kernel.operands.iter().enumerate() {
                let length = definition.region.offset + definition.region.length;
                let values = match sources[stage_index][operand_index] {
                    BufferSource::Caller(binding) => &bindings[binding].values[..length],
                    BufferSource::Scratch(buffer) => &scratch[buffer][..length],
                };
                buffers.push(values.to_vec());
            }
            let mut stage_bindings = buffers
                .iter_mut()
                .enumerate()
                .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
                .collect::<Vec<_>>();
            Interpreter::run(stage, &mut stage_bindings).map_err(|error| {
                CompositionExecutionError::Stage {
                    stage: stage_index,
                    error,
                }
            })?;
            drop(stage_bindings);
            for (operand_index, definition) in kernel.operands.iter().enumerate() {
                if !definition.access.can_write() {
                    continue;
                }
                let length = definition.region.offset + definition.region.length;
                match sources[stage_index][operand_index] {
                    BufferSource::Caller(binding) => bindings[binding].values[..length]
                        .copy_from_slice(&buffers[operand_index][..length]),
                    BufferSource::Scratch(buffer) => {
                        scratch[buffer][..length].copy_from_slice(&buffers[operand_index][..length])
                    }
                }
            }
        }
        Ok(())
    }
}

/// Differentiation request over the exposed (unshared) operands of a composition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionDerivativeRequest {
    pub mode: DerivativeMode,
    pub independent_operands: Vec<StageOperand>,
    pub dependent_operands: Vec<StageOperand>,
}

/// A requested primal operand paired with its derivative operand in the derivative composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionDerivativeOperand {
    pub primal: StageOperand,
    pub derivative: StageOperand,
}

/// Where a primal stage reappears, verbatim, inside a derivative composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionPrimalStage {
    pub primal_stage: usize,
    pub stage: usize,
}

/// Where a stage's derivative kernel sits inside a derivative composition, with the table that
/// maps the stage's readable primal operands to that kernel's operands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionDerivativeStage {
    pub primal_stage: usize,
    pub stage: usize,
    pub primal_operands: Vec<DerivativeOperand>,
}

/// A derivative composition plus its explicit primal/derivative bindings.
///
/// For a JVP, `independent_operands` are direction inputs and `dependent_operands` are tangent
/// outputs. For a VJP, dependent entries are cotangent seeds and independent entries are
/// cotangent outputs. The primal value of every readable operand of a derivative stage is bound
/// through `derivative_stages[..].primal_operands`, except for operands aliased to a shared
/// buffer, which the composition feeds from the reproduced primal producer stages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompositionDerivativeProduct {
    pub mode: DerivativeMode,
    pub composition: KernelComposition,
    pub primal_stages: Vec<CompositionPrimalStage>,
    pub derivative_stages: Vec<CompositionDerivativeStage>,
    pub independent_operands: Vec<CompositionDerivativeOperand>,
    pub dependent_operands: Vec<CompositionDerivativeOperand>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompositionDifferentiationError {
    InvalidPrimal(CompositionError),
    InvalidDerivative(CompositionError),
    UnsupportedMode(DerivativeMode),
    EmptyDependentSet,
    DuplicateIndependent(StageOperand),
    DuplicateDependent(StageOperand),
    InvalidIndependent(StageOperand),
    InvalidDependent(StageOperand),
    IndependentNotReadable(StageOperand),
    DependentNotWritten(StageOperand),
    SharedIndependent(StageOperand),
    SharedDependent(StageOperand),
    /// No requested dependent depends on this independent through the composition.
    IndependentUnreachable(StageOperand),
    /// This dependent depends on no requested independent through the composition.
    DependentUnreachable(StageOperand),
    Stage {
        stage: usize,
        error: DifferentiationError,
    },
}

impl fmt::Display for CompositionDifferentiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel composition differentiation failed: {self:?}")
    }
}

impl Error for CompositionDifferentiationError {}

/// Differentiate a composition into a composition of per-stage derivative kernels.
///
/// Stages that neither receive a direction (JVP) nor feed a seeded dependent get no derivative
/// stage; primal producer stages are reproduced verbatim wherever a derivative stage needs the
/// primal value of a shared input. Requested operands that are structurally disconnected from
/// every operand on the other side of the request are refused rather than silently zero.
pub fn differentiate_composition(
    composition: &KernelComposition,
    request: &CompositionDerivativeRequest,
) -> Result<CompositionDerivativeProduct, CompositionDifferentiationError> {
    let validated = validate_composition(composition.clone())
        .map_err(CompositionDifferentiationError::InvalidPrimal)?;
    if !matches!(request.mode, DerivativeMode::Jvp | DerivativeMode::Vjp) {
        return Err(CompositionDifferentiationError::UnsupportedMode(
            request.mode,
        ));
    }
    if request.dependent_operands.is_empty() {
        return Err(CompositionDifferentiationError::EmptyDependentSet);
    }
    let stage_count = validated.stages.len();
    let operand = |target: StageOperand| {
        validated
            .stages
            .get(target.stage)
            .and_then(|stage| stage.as_kernel().operands.get(target.operand.index()))
    };
    let mut seen = BTreeSet::new();
    let mut requested_independent = vec![Vec::new(); stage_count];
    for target in &request.independent_operands {
        if !seen.insert(*target) {
            return Err(CompositionDifferentiationError::DuplicateIndependent(
                *target,
            ));
        }
        let definition =
            operand(*target).ok_or(CompositionDifferentiationError::InvalidIndependent(*target))?;
        if !definition.access.can_read() {
            return Err(CompositionDifferentiationError::IndependentNotReadable(
                *target,
            ));
        }
        if validated.membership.contains_key(target) {
            return Err(CompositionDifferentiationError::SharedIndependent(*target));
        }
        requested_independent[target.stage].push(target.operand);
    }
    seen.clear();
    let mut requested_dependent = vec![Vec::new(); stage_count];
    for target in &request.dependent_operands {
        if !seen.insert(*target) {
            return Err(CompositionDifferentiationError::DuplicateDependent(*target));
        }
        let definition =
            operand(*target).ok_or(CompositionDifferentiationError::InvalidDependent(*target))?;
        if !definition.access.can_write() {
            return Err(CompositionDifferentiationError::DependentNotWritten(
                *target,
            ));
        }
        if validated.membership.contains_key(target) {
            return Err(CompositionDifferentiationError::SharedDependent(*target));
        }
        requested_dependent[target.stage].push(target.operand);
    }

    // Reachability. `active[s]`: some requested independent reaches stage `s` through shared
    // inputs. `needed[s]`: some output of stage `s` reaches a requested dependent.
    let readers_of_stage = |stage: usize| {
        validated
            .shared
            .iter()
            .enumerate()
            .flat_map(move |(buffer, plan)| {
                plan.readers
                    .iter()
                    .filter(move |reader| reader.stage == stage)
                    .map(move |reader| (buffer, *reader))
            })
    };
    let writers_of_stage = |stage: usize| {
        validated
            .shared
            .iter()
            .enumerate()
            .flat_map(move |(buffer, plan)| {
                plan.writers
                    .iter()
                    .filter(move |writer| writer.stage == stage)
                    .map(move |writer| (buffer, *writer))
            })
    };
    let mut active = vec![false; stage_count];
    for stage in 0..stage_count {
        active[stage] = !requested_independent[stage].is_empty()
            || readers_of_stage(stage).any(|(buffer, _)| {
                validated.shared[buffer]
                    .writers
                    .iter()
                    .any(|writer| active[writer.stage])
            });
    }
    let mut needed = vec![false; stage_count];
    for stage in (0..stage_count).rev() {
        needed[stage] = !requested_dependent[stage].is_empty()
            || writers_of_stage(stage).any(|(buffer, _)| {
                validated.shared[buffer]
                    .readers
                    .iter()
                    .any(|reader| needed[reader.stage])
            });
    }
    let derived = (0..stage_count)
        .map(|stage| active[stage] && needed[stage])
        .collect::<Vec<_>>();
    for target in &request.independent_operands {
        if !derived[target.stage] {
            return Err(CompositionDifferentiationError::IndependentUnreachable(
                *target,
            ));
        }
    }
    for target in &request.dependent_operands {
        if !derived[target.stage] {
            return Err(CompositionDifferentiationError::DependentUnreachable(
                *target,
            ));
        }
    }
    // Primal producer stages reproduced verbatim: every stage whose shared output is read by a
    // derivative stage or by another reproduced stage.
    let mut included = vec![false; stage_count];
    for stage in (0..stage_count).rev() {
        included[stage] = writers_of_stage(stage).any(|(buffer, _)| {
            validated.shared[buffer]
                .readers
                .iter()
                .any(|reader| derived[reader.stage] || included[reader.stage])
        });
    }

    // Per-stage derivative requests.
    let mut stage_requests = Vec::with_capacity(stage_count);
    for stage in 0..stage_count {
        if !derived[stage] {
            stage_requests.push(None);
            continue;
        }
        let mut independent = requested_independent[stage].clone();
        for (buffer, reader) in readers_of_stage(stage) {
            if validated.shared[buffer]
                .writers
                .iter()
                .any(|writer| derived[writer.stage])
                && !independent.contains(&reader.operand)
            {
                independent.push(reader.operand);
            }
        }
        let mut dependent = requested_dependent[stage].clone();
        for (buffer, writer) in writers_of_stage(stage) {
            if validated.shared[buffer]
                .readers
                .iter()
                .any(|reader| derived[reader.stage])
                && !dependent.contains(&writer.operand)
            {
                dependent.push(writer.operand);
            }
        }
        independent.sort();
        dependent.sort();
        stage_requests.push(Some(DerivativeRequest {
            mode: request.mode,
            independent_operands: independent,
            dependent_operands: dependent,
        }));
    }

    // Assemble the derivative composition.
    let mut stages = Vec::new();
    let mut primal_position = vec![None; stage_count];
    let mut primal_stages = Vec::new();
    for stage in 0..stage_count {
        if included[stage] {
            primal_position[stage] = Some(stages.len());
            primal_stages.push(CompositionPrimalStage {
                primal_stage: stage,
                stage: stages.len(),
            });
            stages.push(validated.stages[stage].as_kernel().clone());
        }
    }
    let derivative_order: Vec<usize> = match request.mode {
        DerivativeMode::Jvp => (0..stage_count).collect(),
        _ => (0..stage_count).rev().collect(),
    };
    let mut derivative_position = vec![None; stage_count];
    let mut products: Vec<Option<crate::DerivativeProduct>> = vec![None; stage_count];
    let mut derivative_stages = Vec::new();
    for stage in derivative_order {
        let Some(stage_request) = &stage_requests[stage] else {
            continue;
        };
        let product = differentiate(validated.stages[stage].as_kernel(), stage_request)
            .map_err(|error| CompositionDifferentiationError::Stage { stage, error })?;
        derivative_position[stage] = Some(stages.len());
        derivative_stages.push(CompositionDerivativeStage {
            primal_stage: stage,
            stage: stages.len(),
            primal_operands: product.primal_operands.clone(),
        });
        stages.push(product.kernel.clone());
        products[stage] = Some(product);
    }
    let lookup = |pairs: &[DerivativeOperand], primal: OperandId| {
        pairs
            .iter()
            .find(|pair| pair.primal == primal)
            .map(|pair| pair.derivative)
            .expect("requested operand is part of the stage derivative product")
    };
    let mut shared_buffers = Vec::new();
    for plan in &validated.shared {
        // Primal value sharing: reproduced producers feed reproduced consumers and the primal
        // operands of derivative stages.
        let mut primal_members = Vec::new();
        for member in &plan.members {
            if let Some(position) = primal_position[member.stage] {
                primal_members.push(StageOperand::new(position, member.operand));
            }
        }
        for reader in &plan.readers {
            if let (Some(position), Some(product)) =
                (derivative_position[reader.stage], &products[reader.stage])
            {
                primal_members.push(StageOperand::new(
                    position,
                    lookup(&product.primal_operands, reader.operand),
                ));
            }
        }
        if !primal_members.is_empty() {
            shared_buffers.push(SharedBuffer::new(primal_members));
        }
        // Derivative sharing: JVP tangent of the writer to the direction of every reader; VJP
        // cotangent of every reader to the seed of the writer.
        let mut derivative_members = Vec::new();
        let mut has_writer_side = false;
        let mut has_reader_side = false;
        for writer in &plan.writers {
            let (Some(position), Some(product)) =
                (derivative_position[writer.stage], &products[writer.stage])
            else {
                continue;
            };
            let Some(pair) = product
                .dependent_operands
                .iter()
                .find(|pair| pair.primal == writer.operand)
            else {
                continue;
            };
            has_writer_side = true;
            derivative_members.push(StageOperand::new(position, pair.derivative));
        }
        for reader in &plan.readers {
            let (Some(position), Some(product)) =
                (derivative_position[reader.stage], &products[reader.stage])
            else {
                continue;
            };
            let Some(pair) = product
                .independent_operands
                .iter()
                .find(|pair| pair.primal == reader.operand)
            else {
                continue;
            };
            has_reader_side = true;
            derivative_members.push(StageOperand::new(position, pair.derivative));
        }
        if has_writer_side && has_reader_side {
            derivative_members.sort();
            shared_buffers.push(SharedBuffer::new(derivative_members));
        }
    }
    let independent_operands = request
        .independent_operands
        .iter()
        .map(|target| {
            let product = products[target.stage].as_ref().expect("derived stage");
            CompositionDerivativeOperand {
                primal: *target,
                derivative: StageOperand::new(
                    derivative_position[target.stage].expect("derived stage"),
                    lookup(&product.independent_operands, target.operand),
                ),
            }
        })
        .collect();
    let dependent_operands = request
        .dependent_operands
        .iter()
        .map(|target| {
            let product = products[target.stage].as_ref().expect("derived stage");
            CompositionDerivativeOperand {
                primal: *target,
                derivative: StageOperand::new(
                    derivative_position[target.stage].expect("derived stage"),
                    lookup(&product.dependent_operands, target.operand),
                ),
            }
        })
        .collect();
    let derivative = KernelComposition {
        name: format!(
            "{}::{}",
            composition.name,
            match request.mode {
                DerivativeMode::Jvp => "jvp",
                DerivativeMode::Vjp => "vjp",
                DerivativeMode::Jacobian => "jacobian",
            }
        ),
        stages,
        shared_buffers,
    };
    validate_composition(derivative.clone())
        .map_err(CompositionDifferentiationError::InvalidDerivative)?;
    Ok(CompositionDerivativeProduct {
        mode: request.mode,
        composition: derivative,
        primal_stages,
        derivative_stages,
        independent_operands,
        dependent_operands,
    })
}
