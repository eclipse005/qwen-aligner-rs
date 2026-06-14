//! Qwen3 forced aligner — word-level alignment with hand-written engines.

pub mod config;
pub mod mel;
pub mod audio_io;
pub mod text_io;
pub mod tokenizer;
pub mod text;
pub mod prompt;
pub mod timestamp;
mod inference;
pub mod batch;

// Abstraction modules.
pub(crate) mod raw_tensor;
mod weights;
pub mod backend;

// The CUDA engine modules.
#[cfg(feature = "cuda")]
pub(crate) mod cudarc_engine;
#[cfg(feature = "cuda")]
pub(crate) mod gpu_audio_encoder;

// The CPU engine module.
#[cfg(feature = "cpu")]
pub(crate) mod cpu_engine;

pub use inference::{
    Qwen3ForcedAligner, load_model, release_model,
    AlignRequest, AudioInput, TextInput,
    ForcedAlignItem, ForcedAlignResult,
    write_forced_align_items_json,
};
pub use backend::DeviceRequest;
pub use inference::ModelOptions;
pub use batch::{BatchJob, load_manifest_jobs};
