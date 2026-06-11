//! Qwen3 forced aligner — CUDA-only (cudarc + hand-written kernels).
//!
//! The single inference path lives in `inference::AlignerInference::align`
//! and dispatches every op (audio encoder, text decoder, MRoPE, lm_head
//! gather, argmax) through `cudarc_engine` + `gpu_audio_encoder`.  No burn
//! / candle / cubecl backend is used.

pub mod config;
pub mod mel;
pub mod audio_io;
pub mod text_io;
pub mod tokenizer;
pub mod text;
pub mod prompt;
pub mod timestamp;
pub mod inference;
pub mod batch;

pub mod cudarc_engine;
pub mod gpu_audio_encoder;

pub use inference::{AlignerInference, ForcedAlignItem, ForcedAlignResult, AlignRequest, AudioInput, TextInput};
