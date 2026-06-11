pub mod config;
pub mod mel;
pub mod audio_io;
pub mod text_io;
pub mod tokenizer;
pub mod text;
pub mod prompt;
pub mod timestamp;
pub mod encoder;
pub mod decoder;
pub mod inference;
pub mod batch;

#[cfg(feature = "cuda")]
pub mod cudarc_engine;
#[cfg(feature = "cuda")]
pub mod gpu_audio_encoder;

pub use inference::{AlignerInference, ForcedAlignItem, ForcedAlignResult, AlignRequest, AudioInput, TextInput};

// ─── Backend / Device type aliases ─────────────────────────────────

#[cfg(feature = "cuda")]
pub type Backend = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, half::f16, i32, u8>;
#[cfg(feature = "cuda")]
pub type Device = burn::backend::cuda::CudaDevice;
#[cfg(feature = "cuda")]
pub fn best_device() -> Device { Device::new(0) }
