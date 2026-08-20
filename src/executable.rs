//! Backend-neutral scheduling and executable containers.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::{AxisId, IteratorKind, ValidatedKernel, ValidatedModule};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileDecision {
    pub axis: AxisId,
    pub size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorizationPlan {
    pub axis: AxisId,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelMapping {
    pub axis: AxisId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSchedule {
    pub loop_order: Vec<AxisId>,
    pub tiles: Vec<TileDecision>,
    pub vectorization: Option<VectorizationPlan>,
    pub parallel: Vec<ParallelMapping>,
}

impl KernelSchedule {
    pub fn canonical(kernel: &ValidatedKernel) -> Self {
        Self {
            loop_order: (0..kernel.as_kernel().iteration_domain.rank())
                .map(AxisId::new)
                .collect(),
            tiles: Vec::new(),
            vectorization: None,
            parallel: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutableError {
    InvalidLoopOrder,
    InvalidAxis(usize),
    InvalidTileSize,
    InvalidVectorWidth,
    NonParallelAxis(usize),
}

impl fmt::Display for ExecutableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid kernel schedule: {self:?}")
    }
}
impl Error for ExecutableError {}

#[derive(Clone, Debug, PartialEq)]
pub struct Executable {
    kernel: ValidatedKernel,
    schedule: KernelSchedule,
}

impl Executable {
    pub fn new(kernel: ValidatedKernel, schedule: KernelSchedule) -> Result<Self, ExecutableError> {
        let rank = kernel.as_kernel().iteration_domain.rank();
        let order: BTreeSet<_> = schedule
            .loop_order
            .iter()
            .map(|axis| axis.index())
            .collect();
        if schedule.loop_order.len() != rank || order != (0..rank).collect() {
            return Err(ExecutableError::InvalidLoopOrder);
        }
        let mut tiled = BTreeSet::new();
        for tile in &schedule.tiles {
            if tile.axis.index() >= rank {
                return Err(ExecutableError::InvalidAxis(tile.axis.index()));
            }
            if tile.size == 0 || !tiled.insert(tile.axis.index()) {
                return Err(ExecutableError::InvalidTileSize);
            }
        }
        if let Some(vector) = schedule.vectorization {
            if vector.axis.index() >= rank {
                return Err(ExecutableError::InvalidAxis(vector.axis.index()));
            }
            if vector.width == 0 {
                return Err(ExecutableError::InvalidVectorWidth);
            }
        }
        let mut parallel = BTreeSet::new();
        for mapping in &schedule.parallel {
            let axis = mapping.axis.index();
            if axis >= rank {
                return Err(ExecutableError::InvalidAxis(axis));
            }
            if kernel.as_kernel().iterators[axis] != IteratorKind::Parallel
                || !parallel.insert(axis)
            {
                return Err(ExecutableError::NonParallelAxis(axis));
            }
        }
        Ok(Self { kernel, schedule })
    }

    pub fn reference(kernel: ValidatedKernel) -> Self {
        let schedule = KernelSchedule::canonical(&kernel);
        Self { kernel, schedule }
    }
    pub fn kernel(&self) -> &ValidatedKernel {
        &self.kernel
    }
    pub fn schedule(&self) -> &KernelSchedule {
        &self.schedule
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableModule {
    name: String,
    kernels: Vec<Executable>,
}

impl ExecutableModule {
    pub fn reference(module: ValidatedModule) -> Self {
        let (name, kernels) = module.into_parts();
        Self {
            name,
            kernels: kernels.into_iter().map(Executable::reference).collect(),
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn kernels(&self) -> &[Executable] {
        &self.kernels
    }
    pub fn kernel(&self, name: &str) -> Option<&Executable> {
        self.kernels
            .iter()
            .find(|kernel| kernel.kernel.as_kernel().name == name)
    }
}
