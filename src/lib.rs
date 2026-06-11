//! Qwen3 forced aligner — word-level alignment with hand-written engines.
//!
//! Inference dispatch lives in `inference::Qwen3ForcedAligner::align`, which
//! delegates to one of the engine backends selected via `ModelOptions::device`:
//!
//!   * `DeviceRequest::Cuda(n)` — cudarc + cuBLAS + NVRTC-compiled fused
//!     kernels.  Requires the `cuda` Cargo feature (on by default).
//!   * `DeviceRequest::Cpu` — pure CPU engine (gemm + rayon).  Not yet
//!     implemented; the public type and `Cpu` variant are wired through so
//!     downstream code can stage support without an API churn.
//!   * `DeviceRequest::Auto` — probe CUDA first, fall back to CPU.
//!
//! No deep-learning framework (burn / candle / tch) participates in any
//! inference path; the engines are hand-rolled against driver-level APIs
//! (cudarc / future gemm+rayon).

pub mod config;
pub mod mel;
pub mod audio_io;
pub mod text_io;
pub mod tokenizer;
pub mod text;
pub mod prompt;
pub mod timestamp;
pub mod weight;
pub mod inference;
pub mod batch;

// The CUDA engine modules.  Only compile when the `cuda` feature is on; the
// inference dispatch in `inference.rs` gates the CUDA backend behind the
// same flag and falls back to the (future) CPU engine otherwise.
#[cfg(feature = "cuda")]
pub mod cudarc_engine;
#[cfg(feature = "cuda")]
pub mod gpu_audio_encoder;

pub use weight::WeightTensor;

pub use inference::{
    // Main type + free-function loaders (mirrors the candle crate's surface).
    Qwen3ForcedAligner, load_model, release_model,
    // Options.
    DeviceRequest, ModelOptions,
    // Request / response types.
    AlignRequest, AudioInput, TextInput,
    ForcedAlignItem, ForcedAlignResult,
    // Convenience.
    write_forced_align_items_json,
};
pub use batch::{BatchJob, load_manifest_jobs};
