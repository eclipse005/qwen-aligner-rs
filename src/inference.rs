use anyhow::Context;
use burn::tensor::{Int, Tensor, TensorData};
use half::f16;
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::config::AlignerConfig;
use crate::decoder::{compute_mrope_cos_sin, extract_timestamp_logits, TextDecoder};
use crate::encoder::AudioEncoder;
use crate::mel::extract_log_mel_features;
use crate::{Backend, Device};

#[cfg(feature = "cuda")]
use crate::cudarc_engine::{CudaState, CpuTensor};
#[cfg(feature = "cuda")]
use crate::gpu_audio_encoder::GpuAudioEncoder;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
type F16Slice = CudaSlice<f16>;

const _AUDIO_PAD_TOKEN: &str = "<|video_pad|>";
const F16_TIMESTAMP_ARGMAX_TIE_EPS: f32 = 1.0 / 256.0;

// ─── Public types ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcedAlignItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
    pub output_ids: Vec<i64>,
    pub raw_timestamp_ms: Vec<i64>,
    pub fixed_timestamp_ms: Vec<i64>,
}

impl ForcedAlignResult {
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AudioInput {
    Path(PathBuf),
    Waveform16Khz(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextInput {
    Path(PathBuf),
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlignRequest {
    pub audio: AudioInput,
    pub text: TextInput,
    pub language: String,
}

impl AlignRequest {
    pub fn new(audio: AudioInput, text: TextInput, language: impl Into<String>) -> Self {
        Self { audio, text, language: language.into() }
    }
    pub fn from_paths(audio: impl Into<PathBuf>, text: impl Into<PathBuf>, language: impl Into<String>) -> Self {
        Self::new(AudioInput::Path(audio.into()), TextInput::Path(text.into()), language)
    }
}

// ─── Inference engine ──────────────────────────────────────────────

pub struct AlignerInference {
    /// Burn-side audio encoder.  Only used in the non-cuda build or when
    /// cudarc init fails.  On the cudarc path the cudarc engine owns the
    /// model and this is None, saving ~600MB of GPU memory.
    audio_encoder: Option<AudioEncoder<Backend>>,
    text_decoder: Option<TextDecoder<Backend>>,
    config: AlignerConfig,
    tokenizer: crate::tokenizer::QwenTokenizer,
    device: Device,
    /// Optional cudarc audio encoder (Step 2+). When present, the inference
    /// pipeline uses it instead of the burn `AudioEncoder` for the conv-stem +
    /// transformer forward pass.  Loaded eagerly so we fail fast on cudarc
    /// errors and so the cudarc state stays resident for repeat calls.
    #[cfg(feature = "cuda")]
    gpu_audio_encoder: Option<Arc<GpuAudioEncoder>>,
    #[cfg(feature = "cuda")]
    cuda_state: Option<Arc<CudaState>>,
    /// cudarc text decoder (Step 3+).  Mirrors the burn `TextDecoder` but
    /// dispatches fused cuBLAS+CUDA kernels for every op (no burn tensor
    /// allocations, no .clone()/slice/cat on the hot path).
    #[cfg(feature = "cuda")]
    gpu_text_decoder: Option<Arc<crate::cudarc_engine::GpuTextDecoder>>,
}

unsafe impl Send for AlignerInference {}

impl AlignerInference {
    pub fn load(model_dir: &Path, device: Device) -> anyhow::Result<Self> {
        info!("Loading config...");
        let config = AlignerConfig::from_file(&model_dir.join("config.json"))
            .context("load config")?;

        info!("Loading weights...");
        let weight_data = load_weights(model_dir)?;
        info!("Loaded {} weight tensors", weight_data.len());

        info!("Loading tokenizer...");
        let tokenizer = crate::tokenizer::load_qwen_tokenizer(model_dir)?;

        // Step 3+: when cudarc is available, skip uploading weights to the burn
        // backend entirely.  This saves ~600MB of GPU memory (f16 model weights
        // held on the device) and is essential for fitting long-audio KV caches
        // on the 8GB P104-100.
        //
        // The cudarc build's `align_waveform_text` always dispatches to
        // `align_cudarc`, so the burn-side encoder/decoder fields are unused
        // and stay None.
        #[cfg(feature = "cuda")]
        let (audio_encoder, text_decoder): (Option<AudioEncoder<Backend>>, Option<TextDecoder<Backend>>) = (None, None);

        #[cfg(not(feature = "cuda"))]
        let (audio_encoder, text_decoder) = {
            info!("Loading audio encoder (burn)...");
            let ae = AudioEncoder::load(
                &weight_data, "thinker.audio_tower",
                &config.thinker_config.audio_config, &device,
            ).context("load audio encoder")?;
            info!("Loading text decoder (burn)...");
            let td = TextDecoder::load(
                &weight_data, "thinker.model",
                &config.thinker_config.text_config, &device,
            ).context("load text decoder")?;
            (Some(ae), Some(td))
        };

        // ─── cudarc engine (used for the GpuAudioEncoder + GpuTextDecoder) ───
        #[cfg(feature = "cuda")]
        let (gpu_audio_encoder, cuda_state, gpu_text_decoder) = {
            info!("Initialising cudarc engine for GPU encoders + decoder...");
            let cuda = Arc::new(CudaState::new(0).context("cudarc init")?);
            let gpu_ae = GpuAudioEncoder::load(
                cuda.clone(), &weight_data, "thinker.audio_tower",
                &config.thinker_config.audio_config,
            ).context("load cudarc audio encoder")?;
            let gpu_td = crate::cudarc_engine::GpuTextDecoder::load_with(
                cuda.clone(), &weight_data, "thinker.model",
                &config.thinker_config.text_config,
            ).context("load cudarc text decoder")?;
            (Some(Arc::new(gpu_ae)), Some(cuda), Some(Arc::new(gpu_td)))
        };

        info!("Model loaded successfully.");
        Ok(Self {
            audio_encoder, text_decoder, config, tokenizer, device,
            #[cfg(feature = "cuda")] gpu_audio_encoder,
            #[cfg(feature = "cuda")] cuda_state,
            #[cfg(feature = "cuda")] gpu_text_decoder,
        })
    }

    pub fn align(&self, request: AlignRequest) -> anyhow::Result<ForcedAlignResult> {
        let waveform = match request.audio {
            AudioInput::Path(path) => crate::audio_io::load_wav_mono_16k(&path)?,
            AudioInput::Waveform16Khz(w) => w,
        };
        let text = match request.text {
            TextInput::Path(path) => crate::text_io::load_clean_text(&path)?,
            TextInput::Text(t) => t,
        };
        self.align_waveform_text(&waveform, &text, &request.language)
    }

    pub fn align_batch<I>(&self, requests: I) -> anyhow::Result<Vec<ForcedAlignResult>>
    where I: IntoIterator<Item = AlignRequest>,
    {
        requests.into_iter().enumerate().map(|(i, req)| {
            self.align(req).with_context(|| format!("failed align request {}", i + 1))
        }).collect()
    }

    fn align_waveform_text(&self, waveform: &[f32], text: &str, language: &str) -> anyhow::Result<ForcedAlignResult> {
        // Step 3+: cudarc text decoder path.  When the cudarc decoder is loaded,
        // dispatch the full forward pass (embed → MRoPE → 28 layers → lm_head →
        // timestamp gather → argmax) entirely through fused cuBLAS+CUDA kernels
        // with no burn tensor allocations in the hot path.
        #[cfg(feature = "cuda")]
        if self.gpu_text_decoder.is_some() && self.gpu_audio_encoder.is_some() && self.cuda_state.is_some() {
            return self.align_cudarc(waveform, text, language);
        }

        let mut profile = Profile::new();

        // 1. Text tokenization
        let (words, aligner_input) = crate::text::encode_timestamp(text, language)?;

        // 2. Mel feature extraction
        let features = extract_log_mel_features(waveform)?;
        profile.mark("prepare_input");

        // 3. Audio pad expansion
        let audio_pad_count = crate::prompt::feature_extract_output_len(features.frames as i64);
        let expanded_input = crate::prompt::expand_audio_pad_once(&aligner_input, audio_pad_count as usize)?;

        // 4. Tokenize
        let input_ids_u32 = crate::tokenizer::encode_to_ids(&self.tokenizer, &expanded_input)?;
        let input_ids: Vec<i64> = input_ids_u32.into_iter().map(i64::from).collect();

        // 5. Find timestamp positions
        let timestamp_token_id = self.config.timestamp_token_id as i64;
        let timestamp_positions: Vec<usize> = input_ids.iter().enumerate()
            .filter_map(|(i, &id)| if id == timestamp_token_id { Some(i) } else { None })
            .collect();
        profile.mark("tokenize");

        // 6. Audio encoder
        //    Step 2+ uses the cudarc path: pack mel into [b_chunks, 1, n_mels, cs] f16
        //    chunks, run the GpuAudioEncoder (conv stem + 24 transformer layers +
        //    ln_post + proj1 + gelu + proj2), and download the result.
        let audio_cfg = &self.config.thinker_config.audio_config;
        let n_window = audio_cfg.n_window;
        let cs = n_window * 2;
        let n_mels = audio_cfg.num_mel_bins;
        let nf = features.frames;
        let nfull = nf / cs;
        let tail = nf % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };

        // chunk_tokens[i] = how many conv-stem output tokens this chunk contributes
        // (full chunks → tpc; tail → feo(tail)).
        let tpc = AudioEncoder::<Backend>::feo(cs);
        let mut chunk_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for _ in 0..nfull { chunk_tokens.push(tpc); }
        if tail > 0 { chunk_tokens.push(AudioEncoder::<Backend>::feo(tail)); }

        // Pack mel into cudarc conv-stem layout: for each chunk, write
        // output[(chunk * n_mels + mel) * cs + t] = input[mel * nf + chunk*cs + t]
        // (zero-padding the tail chunk).  Same layout as asr-burn's build_padded_chunks.
        let mut mel_packed: Vec<f16> = vec![f16::ZERO; n_chunks * n_mels * cs];
        for chunk in 0..n_chunks {
            let start = chunk * cs;
            let len = cs.min(nf.saturating_sub(start));
            for mel in 0..n_mels {
                for t in 0..len {
                    let src = mel * nf + start + t;
                    let dst = (chunk * n_mels + mel) * cs + t;
                    mel_packed[dst] = f16::from_f32(features.values[src]);
                }
            }
        }

        let (audio_embeds_data, _out_dim, n_audio_tokens) = if let (Some(gpu_ae), Some(_cuda)) = (self.gpu_audio_encoder.as_ref(), self.cuda_state.as_ref()) {
            // ── cudarc path ──
            let (data, out_dim) = gpu_ae.run(&mel_packed, n_chunks, n_mels, cs, &chunk_tokens)
                .context("cudarc audio encoder")?;
            let n_total: usize = chunk_tokens.iter().sum();
            (data, out_dim, n_total)
        } else {
            // ── burn fallback (no cuda feature, or cudarc failed) ──
            let mel_tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(features.values.clone(), [features.mel_bins, features.frames]),
                &self.device,
            );
            let audio_embeds_tensor = self.audio_encoder.as_ref().expect("burn audio encoder").forward(&mel_tensor)?;
            let dims = audio_embeds_tensor.dims();
            let td = audio_embeds_tensor.into_data();
            // f16 if backend default, else f32
            let data: Vec<f16> = match td.to_vec::<f16>() {
                Ok(v) => v,
                Err(_) => td.to_vec::<f32>().unwrap().into_iter().map(f16::from_f32).collect(),
            };
            (data, dims[1], dims[0])
        };
        profile.mark("audio_encoder");

        // 7. Embedding merge
        if n_audio_tokens != audio_pad_count as usize {
            anyhow::bail!(
                "audio feature/token count mismatch: features={} placeholders={}",
                n_audio_tokens, audio_pad_count,
            );
        }

        let audio_token_id = self.config.thinker_config.audio_token_id;
        let hidden_size = self.config.thinker_config.text_config.hidden_size;
        let seq_len = input_ids.len();

        // Embed all input_ids
        let ids_tensor = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(input_ids.iter().map(|&id| id as i32).collect::<Vec<_>>(), [seq_len]),
            &self.device,
        );
        let embeds = self.text_decoder.as_ref().expect("burn text decoder").embed(&ids_tensor); // [seq_len, hidden]

        // Download embeddings, replace audio positions with audio features, re-upload.
        // audio_embeds_data is f16 (cudarc path) or f32-then-cast (burn path); both end up
        // as f16 in `embed_vals` to match the burn backend's default f16 storage.
        let embed_data = embeds.into_data();
        let mut embed_vals: Vec<f16> = embed_data.to_vec::<f16>().unwrap_or_else(|_| {
            embed_data.to_vec::<f32>().expect("embed dtype")
                .into_iter().map(f16::from_f32).collect()
        });

        let mut audio_idx = 0usize;
        for (tok_idx, &tok_id) in input_ids.iter().enumerate() {
            if tok_id == audio_token_id as i64 {
                let dst = tok_idx * hidden_size;
                let src = audio_idx * hidden_size;
                for j in 0..hidden_size {
                    // audio_embeds is already f16; the f16→f32→f16 round-trip
                    // here matches what the previous burn path did, preserving
                    // bit-exact equivalence with the candle baseline.
                    embed_vals[dst + j] = f16::from_f32(audio_embeds_data[src + j].to_f32());
                }
                audio_idx += 1;
            }
        }

        let inputs_embeds = Tensor::<Backend, 3>::from_data(
            TensorData::new(embed_vals, [1, seq_len, hidden_size]),
            &self.device,
        );
        profile.mark("merge_embeddings");

        // 8. MRoPE computation
        let text_cfg = &self.config.thinker_config.text_config;
        let all_pos: Vec<i64> = (0..seq_len as i64).collect();
        let pos_3d: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos, sin) = compute_mrope_cos_sin(
            &pos_3d, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
            &self.device,
        );
        profile.mark("rope_compute");

        // 9. Text decoder (single forward pass)
        let hidden_states = self.text_decoder.as_ref().expect("burn text decoder").forward_hidden(&inputs_embeds, &cos, &sin);
        profile.mark("text_decoder");

        // 10. Timestamp logits extraction
        let td = self.text_decoder.as_ref().expect("burn text decoder");
        let (logits, _n_timestamps, classify_num) = extract_timestamp_logits(
            &hidden_states, &timestamp_positions,
            &td.norm, &td.lm_head,
        );
        profile.mark("timestamp_logits");

        // 11. Argmax with f16 tie-breaking
        let output_ids = argmax_rows(&logits, classify_num);

        // 12. Timestamp fix
        let result = timestamp_ids_to_run(&words, &output_ids, self.config.timestamp_segment_time)?;
        profile.mark("total");

        Ok(result)
    }

    /// Step 3+: full cudarc inference pipeline.  Audio encoder already uses
    /// `gpu_audio_encoder`; here we also run the 28-layer text decoder, MRoPE,
    /// and the timestamp-head argmax entirely on the GPU.
    #[cfg(feature = "cuda")]
    fn align_cudarc(&self, waveform: &[f32], text: &str, language: &str) -> anyhow::Result<ForcedAlignResult> {
        use crate::cudarc_engine::{
            compute_mrope_cos_sin as cudarc_mrope, CpuTensor, GpuKvCache,
        };
        let mut profile = Profile::new();

        let cuda = self.cuda_state.as_ref().expect("cudarc engine");
        let gpu_td = self.gpu_text_decoder.as_ref().expect("cudarc decoder");
        let gpu_ae = self.gpu_audio_encoder.as_ref().expect("cudarc audio encoder");

        // 1. Text tokenization (CPU)
        let (words, aligner_input) = crate::text::encode_timestamp(text, language)?;
        // 2. Mel feature extraction (CPU)
        let features = extract_log_mel_features(waveform)?;
        profile.mark("prepare_input");

        // 3. Audio pad expansion (CPU)
        let audio_pad_count = crate::prompt::feature_extract_output_len(features.frames as i64);
        let expanded_input = crate::prompt::expand_audio_pad_once(&aligner_input, audio_pad_count as usize)?;
        // 4. Tokenize (CPU)
        let input_ids_u32 = crate::tokenizer::encode_to_ids(&self.tokenizer, &expanded_input)?;
        let input_ids: Vec<i64> = input_ids_u32.into_iter().map(i64::from).collect();
        // 5. Find timestamp positions
        let timestamp_token_id = self.config.timestamp_token_id as i64;
        let timestamp_positions: Vec<usize> = input_ids.iter().enumerate()
            .filter_map(|(i, &id)| if id == timestamp_token_id { Some(i) } else { None })
            .collect();
        profile.mark("tokenize");

        // 6. Audio encoder (cudarc) ─────────────────────────────
        let audio_cfg = &self.config.thinker_config.audio_config;
        let n_window = audio_cfg.n_window;
        let cs = n_window * 2;
        let n_mels = audio_cfg.num_mel_bins;
        let nf = features.frames;
        let nfull = nf / cs;
        let tail = nf % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };
        let tpc = AudioEncoder::<Backend>::feo(cs);
        let mut chunk_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for _ in 0..nfull { chunk_tokens.push(tpc); }
        if tail > 0 { chunk_tokens.push(AudioEncoder::<Backend>::feo(tail)); }

        // Pack mel into cudarc conv-stem layout (f16).
        let mut mel_packed: Vec<f16> = vec![f16::ZERO; n_chunks * n_mels * cs];
        for chunk in 0..n_chunks {
            let start = chunk * cs;
            let len = cs.min(nf.saturating_sub(start));
            for mel in 0..n_mels {
                for t in 0..len {
                    let src = mel * nf + start + t;
                    let dst = (chunk * n_mels + mel) * cs + t;
                    mel_packed[dst] = f16::from_f32(features.values[src]);
                }
            }
        }
        let (audio_embeds_data, _out_dim, n_audio_tokens) = {
            let (data, out_dim) = gpu_ae.run(&mel_packed, n_chunks, n_mels, cs, &chunk_tokens)?;
            let n_total: usize = chunk_tokens.iter().sum();
            (data, out_dim, n_total)
        };
        profile.mark("audio_encoder");

        if n_audio_tokens != audio_pad_count as usize {
            anyhow::bail!("audio feature/token count mismatch: features={} placeholders={}",
                n_audio_tokens, audio_pad_count);
        }

        // 7. Embedding merge (GPU-side, no CPU detour) ────────
        // Step 4: replace CPU-side scatter with two GPU ops:
        //   a) embed_lookup on GPU (returns GpuTensor)
        //   b) scatter_audio_rows: splice audio_embeds into the audio token
        //      positions directly on the device.
        let audio_token_id = self.config.thinker_config.audio_token_id;
        let hidden_size = self.config.thinker_config.text_config.hidden_size;
        let seq_len = input_ids.len();

        // a) Embed lookup on GPU.
        let ids_dev = cuda.upload_i64(&input_ids.iter().map(|&x| x).collect::<Vec<_>>())?;
        let mut embeds_gpu = cuda.embed_lookup(&gpu_td.embed_table, &ids_dev)?;
        // embeds_gpu shape: [seq_len, hidden]

        // b) Build the audio_token position list (in input_ids order).
        //    Upload audio_embeds_data to GPU and scatter.
        let audio_positions: Vec<i32> = input_ids.iter().enumerate()
            .filter_map(|(i, &id)| if id == audio_token_id as i64 { Some(i as i32) } else { None })
            .collect();
        assert_eq!(audio_positions.len(), n_audio_tokens);
        let audio_pos_dev = cuda.stream.clone_htod(&audio_positions)?;
        let audio_embeds_dev = cuda.upload_f16(&audio_embeds_data)?;
        let audio_embeds_tensor = crate::cudarc_engine::GpuTensor::new(
            audio_embeds_dev, vec![n_audio_tokens, hidden_size]
        );
        // embeds_gpu is currently [seq_len, hidden] but GpuTensor is conceptually
        // a 3D wrapper.  Reshape to [1, seq_len, hidden] then back to [seq_len, hidden]
        // is a no-op (just a shape metadata change), so we can call scatter directly.
        cuda.scatter_audio_rows(&mut embeds_gpu, &audio_embeds_tensor, &audio_pos_dev)?;

        // Wrap as [1, seq_len, hidden] for the decoder.
        let inputs_embeds_gpu = embeds_gpu.reshape(vec![1, seq_len, hidden_size]);
        profile.mark("merge_embeddings");

        // 8. MRoPE cos/sin (CPU compute → GPU upload) ───────────
        let text_cfg = &self.config.thinker_config.text_config;
        let all_pos: Vec<i64> = (0..seq_len as i64).collect();
        let pos_3d: [Vec<i64>; 3] = [all_pos.clone(), all_pos.clone(), all_pos.clone()];
        let (cos_cpu, sin_cpu) = cudarc_mrope(
            &pos_3d, text_cfg.head_dim, text_cfg.rope_theta,
            &text_cfg.mrope_section(), text_cfg.mrope_interleaved(),
        );
        // cudarc_engine's compute_mrope_cos_sin returns CpuTensor, but
        // GpuDecoderLayer expects &[CudaSlice<f16>] for cos/sin.  We re-upload.
        let cos_dev = cuda.upload_f16(&cos_cpu.data)?;
        let sin_dev = cuda.upload_f16(&sin_cpu.data)?;
        profile.mark("rope_compute");

        // 9. 28-layer text decoder forward (cudarc) ──────────────
        // For aligner: use_causal=true; kv_start=0 (full prefill).
        // max_seq must be >= seq_len so the KV cache can hold the full sequence.
        // Aligner 180s audio: seq_len=4567.  KV cache is 28×8×128×max_seq×2×2 bytes
        // (≤ 524MB for seq_len=4567).  P104-100 8GB has ~3GB headroom after
        // cudarc uploads, so we can hold 4567 + 64 padding.
        let max_seq: usize = seq_len + 64;
        let nkvh = text_cfg.num_key_value_heads;
        let hd = text_cfg.head_dim;
        let mut kv = GpuKvCache::new(cuda, text_cfg.num_hidden_layers, 1, nkvh, max_seq, hd)?;
        // GpuTextDecoder::forward runs the 28 transformer layers + final RMSNorm + lm_head
        // in one call.  Output shape: [1, seq_len, classify_num] (= 5000 for aligner).
        let logits_full = gpu_td.forward(inputs_embeds_gpu, &cos_dev, &sin_dev, &mut kv, 0, true, false)?;
        // Force sync so the `text_decoder` profile time reflects actual GPU work,
        // not just kernel-submit time.  Without this, downstream calls (which
        // do a sync via cudaMemcpy) appear to eat all the time, while the
        // 28-layer decoder and lm_head GEMM (the real heavy work) hide in
        // the gap.
        cuda.synchronize()?;
        profile.mark("text_decoder");

        // 10. Gather timestamp logits from the [1, seq_len, classify_num] tensor.
        //     Use cuda.embed_lookup with logits_full as a 2D weight matrix
        //     (rows=seq_len, cols=classify_num) to gather rows at the timestamp positions.
        let logits_2d = logits_full.reshape(vec![seq_len, logits_full.shape()[2]]);
        let ts_indices: Vec<i64> = timestamp_positions.iter().map(|&p| p as i64).collect();
        let ts_indices_dev = cuda.upload_i64(&ts_indices)?;
        let n_pos = timestamp_positions.len();
        let classify_num = logits_2d.shape()[1];
        let logits_gathered = cuda.embed_lookup(&crate::cudarc_engine::GpuWeight {
            data: logits_2d.data().clone(),
            rows: seq_len,
            cols: classify_num,
        }, &ts_indices_dev)?;
        let logits_data = cuda.download_tensor(&logits_gathered)?.data;
        let logits_f32: Vec<f32> = logits_data.iter().map(|v| v.to_f32()).collect();
        profile.mark("timestamp_logits");

        // 11. Argmax with f16 tie-breaking
        let output_ids = argmax_rows(&logits_f32, classify_num);

        // 12. Timestamp fix
        let result = timestamp_ids_to_run(&words, &output_ids, self.config.timestamp_segment_time)?;
        profile.mark("total");

        Ok(result)
    }
}

// ─── Helpers ───────────────────────────────────────────────────────

fn to_f16_f32(v: f32) -> f32 { f16::from_f32(v).to_f32() }

pub(crate) fn argmax_rows(values: &[f32], cols: usize) -> Vec<i64> {
    values.chunks(cols).map(|row| {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val { best_idx = i; best_val = v; }
        }
        let tie_floor = best_val - F16_TIMESTAMP_ARGMAX_TIE_EPS;
        row.iter().position(|&v| v >= tie_floor).unwrap_or(best_idx) as i64
    }).collect()
}

fn timestamp_ids_to_run(
    words: &[String], output_ids: &[i64], timestamp_segment_time_ms: f32,
) -> anyhow::Result<ForcedAlignResult> {
    if output_ids.len() != words.len() * 2 {
        anyhow::bail!(
            "timestamp count mismatch: words={} timestamps={}",
            words.len(), output_ids.len(),
        );
    }
    let raw_ms: Vec<i64> = output_ids.iter()
        .map(|id| (*id as f32 * timestamp_segment_time_ms) as i64)
        .collect();
    let fixed_ms = crate::timestamp::fix_timestamp(&raw_ms);
    let mut items = Vec::with_capacity(words.len());
    for (wi, word) in words.iter().enumerate() {
        items.push(ForcedAlignItem {
            text: word.clone(),
            start_time: fixed_ms[wi * 2] as f64 / 1000.0,
            end_time: fixed_ms[wi * 2 + 1] as f64 / 1000.0,
        });
    }
    Ok(ForcedAlignResult {
        items, output_ids: output_ids.to_vec(),
        raw_timestamp_ms: raw_ms, fixed_timestamp_ms: fixed_ms,
    })
}

pub fn write_forced_align_items_json(output: &Path, items: &[ForcedAlignItem]) -> anyhow::Result<()> {
    if let Some(parent) = output.parent() { std::fs::create_dir_all(parent)?; }
    let json = serde_json::to_string_pretty(items)?;
    std::fs::write(output, json)?;
    Ok(())
}

// ─── Weight loading ────────────────────────────────────────────────

fn load_weights(model_dir: &Path) -> anyhow::Result<HashMap<String, TensorData>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let wm = idx["weight_map"].as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid index.json"))?;
        let mut sf: std::collections::HashSet<String> = std::collections::HashSet::new();
        for v in wm.values() { if let Some(s) = v.as_str() { sf.insert(s.to_string()); } }
        let mut all = HashMap::new();
        for s in sf { all.extend(load_safetensors_file(&model_dir.join(&s))?); }
        return Ok(all);
    }
    load_safetensors_file(&model_dir.join("model.safetensors"))
}

fn load_safetensors_file(path: &Path) -> anyhow::Result<HashMap<String, TensorData>> {
    let buf = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| anyhow::anyhow!("safetensors: {}", e))?;
    let names = st.names();
    let tensors = st.tensors();
    let mut weights = HashMap::new();
    for i in 0..names.len() {
        let name = names[i];
        let view = &tensors[i];
        let raw = view.1.data();
        let shape: Vec<usize> = view.1.shape().to_vec();
        let f32_data: Vec<f32> = match view.1.dtype() {
            safetensors::Dtype::F32 => raw.chunks_exact(4).map(|c| {
                f32::from_ne_bytes([c[0], c[1], c[2], c[3]])
            }).collect(),
            safetensors::Dtype::BF16 => raw.chunks_exact(2).map(|c| {
                let b = u16::from_ne_bytes([c[0], c[1]]);
                f32::from_bits((b as u32) << 16)
            }).collect(),
            safetensors::Dtype::F16 => raw.chunks_exact(2).map(|c| {
                half::f16::from_ne_bytes([c[0], c[1]]).to_f32()
            }).collect(),
            other => anyhow::bail!("unsupported dtype: {:?} in {}", other, name),
        };
        weights.insert(name.to_string(), TensorData::new(f32_data, shape));
    }
    Ok(weights)
}

// ─── Profile helper ────────────────────────────────────────────────

struct Profile { enabled: bool, start: Instant, last: Instant }

impl Profile {
    fn new() -> Self {
        let now = Instant::now();
        Self { enabled: std::env::var_os("QFA_PROFILE").is_some(), start: now, last: now }
    }
    fn mark(&mut self, label: &str) {
        if !self.enabled { return; }
        let now = Instant::now();
        eprintln!("profile {label}: stage={:.3}s total={:.3}s",
            (now - self.last).as_secs_f64(), (now - self.start).as_secs_f64());
        self.last = now;
    }
}
