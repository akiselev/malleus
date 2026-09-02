//! Deterministic identities for schedule-independent kernel artifacts.
//!
//! A digest is blake3 over the canonical serde JSON encoding of a typed payload that names its
//! schema. Typed structs fix field order and every collection in the IR is an ordered `Vec`, so
//! the encoding is deterministic without a general JSON canonicalizer. The wire shape
//! (`algorithm` plus lowercase `hex`) is the one the Sinbad federation already uses at repository
//! boundaries, so a Malleus digest can be carried verbatim inside consumer receipts.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{StructuredKernel, StructuredModule};

/// Schema of the payload behind [`kernel_digest`].
pub const KERNEL_DIGEST_SCHEMA: &str = "malleus-kernel-digest/1";
/// Schema of the payload behind [`module_digest`].
pub const MODULE_DIGEST_SCHEMA: &str = "malleus-module-digest/1";

/// A content digest that records its algorithm.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Digest {
    pub algorithm: String,
    pub hex: String,
}

impl Digest {
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            algorithm: "blake3".into(),
            hex: blake3::hash(bytes).to_hex().to_string(),
        }
    }

    pub(crate) fn of_payload(payload: &impl Serialize) -> Self {
        Self::blake3(&serde_json::to_vec(payload).expect("kernel artifacts serialize infallibly"))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hex)
    }
}

#[derive(Serialize)]
struct KernelPayload<'a> {
    schema: &'static str,
    kernel: &'a StructuredKernel,
}

/// Identity of one schedule-independent kernel: its name, iteration domain, operands, indexing
/// maps, body, and numeric policy. Schedules and executables never enter it.
pub fn kernel_digest(kernel: &StructuredKernel) -> Digest {
    Digest::of_payload(&KernelPayload {
        schema: KERNEL_DIGEST_SCHEMA,
        kernel,
    })
}

#[derive(Serialize)]
struct ModulePayload<'a> {
    schema: &'static str,
    name: &'a str,
    kernels: Vec<Digest>,
}

/// Identity of a module: its name plus the ordered kernel digests.
pub fn module_digest(module: &StructuredModule) -> Digest {
    Digest::of_payload(&ModulePayload {
        schema: MODULE_DIGEST_SCHEMA,
        name: &module.name,
        kernels: module.kernels.iter().map(kernel_digest).collect(),
    })
}
