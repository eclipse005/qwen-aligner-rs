//! GPU audio encoder for Qwen3-ASR — cuBLAS + custom kernels.
//!
//! Architecture (Qwen3-ASR audio tower):
//!   conv2d stem (3 × conv2d + GELU) → conv_out (Linear) + sinusoidal PE
//!   → 18 × { LayerNorm + Self-attention (windowed) + LayerNorm + FFN (GELU) }
//!   → ln_post + proj1 + GELU + proj2
//!
//! The conv2d stem is run by the existing burn AudioEncoder (small fraction
//! of total cost) and the resulting [n_tokens, d_model] tensor is downloaded
//! once.  Everything downstream stays on GPU.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

use burn::tensor::TensorData;

use crate::config::AudioConfig;
use crate::cudarc_engine::{
    CpuTensor, CudaState, GpuTensor, GpuWeight, gpu_helpers,
};

// ─── Linear + LayerNorm primitives ─────────────────────────────────

pub(crate) struct GpuLinear {
    pub w: GpuWeight,
    pub bias: Option<CudaSlice<f16>>,
}

impl GpuLinear {
    fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        let w = gpu_helpers::load_gpu_weight(cuda, weights, &format!("{}.weight", prefix))?;
        let bias = if weights.contains_key(&format!("{}.bias", prefix)) {
            Some(gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.bias", prefix))?)
        } else {
            None
        };
        Ok(Self { w, bias })
    }

    fn forward(&self, cuda: &CudaState, x: &GpuTensor) -> Result<GpuTensor> {
        let mut y = cuda.linear_gpu(x, &self.w)?;
        if let Some(bias) = &self.bias {
            cuda.add_bias_inplace(&mut y, bias)?;
        }
        Ok(y)
    }
}

pub(crate) struct GpuLayerNorm {
    w: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    eps: f32,
}

impl GpuLayerNorm {
    fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            w: gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.weight", prefix))?,
            bias: gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.bias", prefix))?,
            eps,
        })
    }
    fn forward(&self, cuda: &CudaState, x: &GpuTensor) -> Result<GpuTensor> {
        cuda.layer_norm(x, &self.w, &self.bias, self.eps)
    }
}

// ─── Self-attention ────────────────────────────────────────────────

struct GpuAudioAttention {
    q_proj: GpuLinear,
    k_proj: GpuLinear,
    v_proj: GpuLinear,
    out_proj: GpuLinear,
    num_heads: usize,
    head_dim: usize,
}

impl GpuAudioAttention {
    fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize) -> Result<Self> {
        Ok(Self {
            q_proj: GpuLinear::load(cuda, weights, &format!("{}.q_proj", prefix))?,
            k_proj: GpuLinear::load(cuda, weights, &format!("{}.k_proj", prefix))?,
            v_proj: GpuLinear::load(cuda, weights, &format!("{}.v_proj", prefix))?,
            out_proj: GpuLinear::load(cuda, weights, &format!("{}.out_proj", prefix))?,
            num_heads: nh,
            head_dim: dm / nh,
        })
    }

    /// x [1, s, dm]; windowed attention over chunks of size `ws`.
    fn forward(&self, cuda: &CudaState, x: &GpuTensor, ws: Option<usize>) -> Result<GpuTensor> {
        let dims = x.shape();
        let b = dims[0]; let s = dims[1]; let _dm = dims[2];
        let nh = self.num_heads; let hd = self.head_dim;

        // Project Q, K, V.
        let q = self.q_proj.forward(cuda, x)?;
        let k = self.k_proj.forward(cuda, x)?;
        let v = self.v_proj.forward(cuda, x)?;

        // [b, s, nh, hd] → [b, nh, s, hd].
        let q = cuda.swap_dims_12(&q.reshape(vec![b, s, nh, hd]))?;
        let k = cuda.swap_dims_12(&k.reshape(vec![b, s, nh, hd]))?;
        let v = cuda.swap_dims_12(&v.reshape(vec![b, s, nh, hd]))?;

        let window = ws.filter(|&w| w > 0 && w < s);
        let scale = 1.0f32 / (hd as f32).sqrt();

        let attn_out = if let Some(w) = window {
            // Pre-allocate the output buffer [b, nh, s, hd]; each chunk writes its slice via kernel.
            let mut out_buf = cuda.alloc_zeros_f16(b * nh * s * hd)?;
            for st in (0..s).step_by(w) {
                let ln = w.min(s - st);
                let qw = cuda.slice_dim2(&q, st, ln)?;
                let kw = cuda.slice_dim2(&k, st, ln)?;
                let vw = cuda.slice_dim2(&v, st, ln)?;
                let scores = cuda.attention_qk(&qw, &kw)?;
                let attn = cuda.softmax_scaled_causal(&scores, scale, false)?;
                let o = cuda.attention_av(&attn, &vw)?;
                cuda.concat_dim2_write(&mut out_buf, &o, b, nh, s, hd, st)?;
            }
            GpuTensor::new(out_buf, vec![b, nh, s, hd])
        } else {
            let scores = cuda.attention_qk(&q, &k)?;
            let attn = cuda.softmax_scaled_causal(&scores, scale, false)?;
            cuda.attention_av(&attn, &v)?
        };

        let attn_flat = cuda.swap_dims_12(&attn_out)?.reshape(vec![b, s, nh * hd]);
        self.out_proj.forward(cuda, &attn_flat)
    }
}

// ─── FFN ───────────────────────────────────────────────────────────

struct GpuAudioFfn {
    fc1: GpuLinear,
    fc2: GpuLinear,
}

impl GpuAudioFfn {
    fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: GpuLinear::load(cuda, weights, &format!("{}.fc1", prefix))?,
            fc2: GpuLinear::load(cuda, weights, &format!("{}.fc2", prefix))?,
        })
    }
    fn forward(&self, cuda: &CudaState, x: &GpuTensor) -> Result<GpuTensor> {
        let mut h = self.fc1.forward(cuda, x)?;
        cuda.gelu_inplace(&mut h)?;
        self.fc2.forward(cuda, &h)
    }
}

// ─── Encoder layer ─────────────────────────────────────────────────

struct GpuAudioLayer {
    sln: GpuLayerNorm,
    attn: GpuAudioAttention,
    fln: GpuLayerNorm,
    ffn: GpuAudioFfn,
}

impl GpuAudioLayer {
    fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize) -> Result<Self> {
        Ok(Self {
            sln: GpuLayerNorm::load(cuda, weights, &format!("{}.self_attn_layer_norm", prefix), 1e-5)?,
            attn: GpuAudioAttention::load(cuda, weights, &format!("{}.self_attn", prefix), nh, dm)?,
            fln: GpuLayerNorm::load(cuda, weights, &format!("{}.final_layer_norm", prefix), 1e-5)?,
            ffn: GpuAudioFfn::load(cuda, weights, prefix)?,
        })
    }
    fn forward(&self, cuda: &CudaState, x: GpuTensor, ws: Option<usize>) -> Result<GpuTensor> {
        let normed = self.sln.forward(cuda, &x)?;
        let attn_out = self.attn.forward(cuda, &normed, ws)?;
        let mut x1 = cuda.add(&x, &attn_out)?;
        let normed2 = self.fln.forward(cuda, &x1)?;
        let ffn_out = self.ffn.forward(cuda, &normed2)?;
        cuda.add_inplace(&mut x1, &ffn_out)?;
        Ok(x1)
    }
}

// ─── Audio encoder transformer (post conv-stem) ────────────────────

pub(crate) struct GpuConvStem {
    c1_w: CudaSlice<f16>, c1_b: CudaSlice<f16>,
    c2_w: CudaSlice<f16>, c2_b: CudaSlice<f16>,
    c3_w: CudaSlice<f16>, c3_b: CudaSlice<f16>,
    co: GpuLinear,
    pe: CudaSlice<f16>,  // [max_source_positions, d_model]
    d_model: usize,
    max_pos: usize,
    c1_out: usize, c2_out: usize, c3_out: usize,
}

impl GpuConvStem {
    pub fn load(cuda: &CudaState, weights: &HashMap<String, TensorData>, prefix: &str, config: &AudioConfig) -> Result<Self> {
        let c1_w = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d1.weight", prefix))?;
        let c1_b = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d1.bias", prefix))?;
        let c2_w = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d2.weight", prefix))?;
        let c2_b = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d2.bias", prefix))?;
        let c3_w = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d3.weight", prefix))?;
        let c3_b = gpu_helpers::load_gpu_vec(cuda, weights, &format!("{}.conv2d3.bias", prefix))?;
        let co = GpuLinear::load(cuda, weights, &format!("{}.conv_out", prefix))?;

        // Get conv channel counts from weight shapes.
        let c1_out = weights.get(&format!("{}.conv2d1.weight", prefix)).unwrap().shape[0];
        let c2_out = weights.get(&format!("{}.conv2d2.weight", prefix)).unwrap().shape[0];
        let c3_out = weights.get(&format!("{}.conv2d3.weight", prefix)).unwrap().shape[0];

        // Compute sinusoidal PE on CPU then upload.
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0);
        let mut pe_f32 = vec![0.0f32; max_pos * dm];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe_f32[p * dm + i] = a.sin() as f32;
                pe_f32[p * dm + half + i] = a.cos() as f32;
            }
        }
        let pe_f16: Vec<f16> = pe_f32.into_iter().map(f16::from_f32).collect();
        let pe = cuda.upload_f16(&pe_f16)?;

        Ok(Self { c1_w, c1_b, c2_w, c2_b, c3_w, c3_b, co, pe, d_model: dm, max_pos, c1_out, c2_out, c3_out })
    }

    /// Run conv stem on chunked mel input [b_chunks, 1, 128, cs] → [b_chunks, t2, d_model] flat output.
    /// Returns (flat output, t2_per_chunk).
    pub fn forward(&self, cuda: &CudaState, mel_chunks: &[f16], b_chunks: usize, n_mels: usize, cs: usize) -> Result<(GpuTensor, usize)> {
        // Upload mel chunks as [b_chunks, 1, n_mels, cs].
        let x_cpu = CpuTensor::new(mel_chunks.to_vec(), vec![b_chunks, 1, n_mels, cs]);
        let x = cuda.upload_tensor(&x_cpu)?;

        let x = cuda.conv2d_3x3_s2p1_gelu(&x, &self.c1_w, self.c1_out, 1, &self.c1_b)?;
        let x = cuda.conv2d_3x3_s2p1_gelu(&x, &self.c2_w, self.c2_out, self.c1_out, &self.c2_b)?;
        let x = cuda.conv2d_3x3_s2p1_gelu(&x, &self.c3_w, self.c3_out, self.c2_out, &self.c3_b)?;
        // x: [b_chunks, c3_out, f2, t2]
        let s = x.shape();
        let (b2, c2_dim, f2, t2) = (s[0], s[1], s[2], s[3]);

        // Permute [b, c, f, t] → [b, t, c, f] → reshape [b, t, c*f]
        // We use a single swap_dims_12 doesn't fit here; do via download for now since
        // it's small (~31×16×13×480 bytes).  TODO: write permute kernel.
        let x_cpu = cuda.download_tensor(&x)?;
        let mut perm = vec![f16::ZERO; b2 * t2 * c2_dim * f2];
        for ib in 0..b2 {
            for it in 0..t2 {
                for ic in 0..c2_dim {
                    for f in 0..f2 {
                        let src = ((ib * c2_dim + ic) * f2 + f) * t2 + it;
                        let dst = ((ib * t2 + it) * c2_dim + ic) * f2 + f;
                        perm[dst] = x_cpu.data[src];
                    }
                }
            }
        }
        let r = cuda.upload_tensor(&CpuTensor::new(perm, vec![b2, t2, c2_dim * f2]))?;
        let co = self.co.forward(cuda, &r)?;
        // co: [b2, t2, d_model] — add PE [t2, d_model] broadcast over b2
        // We do this on CPU since t2 is small and we need to slice/concat after anyway.
        let co_cpu = cuda.download_tensor(&co)?;
        let pe_cpu = cuda.stream.clone_dtoh(&self.pe)?;
        let dm = self.d_model;
        let mut out = co_cpu.data.clone();
        for ib in 0..b2 {
            for it in 0..t2 {
                let base = (ib * t2 + it) * dm;
                let pe_base = it * dm;
                for j in 0..dm {
                    let v = f32::from(out[base + j]) + f32::from(pe_cpu[pe_base + j]);
                    out[base + j] = f16::from_f32(v);
                }
            }
        }

        let _ = self.max_pos;
        let final_gpu = cuda.upload_tensor(&CpuTensor::new(out, vec![b2, t2, dm]))?;
        Ok((final_gpu, t2))
    }
}

pub(crate) struct GpuAudioEncoder {
    layers: Vec<GpuAudioLayer>,
    ln_post: GpuLayerNorm,
    proj1: GpuLinear,
    proj2: GpuLinear,
    config: AudioConfig,
    pub conv_stem: GpuConvStem,
    pub cuda: Arc<CudaState>,
}

impl GpuAudioEncoder {
    pub fn load(cuda: Arc<CudaState>, weights: &HashMap<String, TensorData>, prefix: &str, config: &AudioConfig) -> Result<Self> {
        let dm = config.d_model;
        let nh = config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(GpuAudioLayer::load(&cuda, weights, &format!("{}.layers.{}", prefix, i), nh, dm)?);
        }
        let ln_post = GpuLayerNorm::load(&cuda, weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = GpuLinear::load(&cuda, weights, &format!("{}.proj1", prefix))?;
        let proj2 = GpuLinear::load(&cuda, weights, &format!("{}.proj2", prefix))?;
        let conv_stem = GpuConvStem::load(&cuda, weights, prefix, config)?;
        Ok(Self { layers, ln_post, proj1, proj2, config: config.clone(), conv_stem, cuda })
    }

    /// Run the full audio encoder on chunked mel input.
    /// mel_chunks: [b_chunks * 1 * n_mels * cs] f16 flat array (zero-padded tail chunk).
    /// chunk_tokens[i] = how many tokens the i-th chunk contributes (tpc for full, feo(tail) for partial).
    /// Returns (output_data, output_dim) — flat [n_tokens, output_dim] f16.
    ///
    /// Aligner uses **full attention** (no windowing), so we pass `None` to layer.forward.
    pub fn run(&self, mel_chunks: &[f16], b_chunks: usize, n_mels: usize, cs: usize,
               chunk_tokens: &[usize]) -> Result<(Vec<f16>, usize)>
    {
        let cuda = &self.cuda;
        // 1. Conv stem
        let (co_gpu, t2) = self.conv_stem.forward(cuda, mel_chunks, b_chunks, n_mels, cs)?;
        // co_gpu: [b_chunks, t2, d_model]
        let dm = self.config.d_model;
        let n_total: usize = chunk_tokens.iter().sum();

        // 2. Pack valid tokens of each chunk into a single contiguous [1, n_total, d_model] buffer
        //    (this is the slice loop, on GPU via download+pack since t2 differs per chunk).
        //    Tail chunk has chunk_tokens[i] < t2; full chunks have == t2.
        let co_cpu = cuda.download_tensor(&co_gpu)?;
        let mut packed = Vec::with_capacity(n_total * dm);
        for (idx, &v) in chunk_tokens.iter().enumerate() {
            let base = idx * t2 * dm;
            packed.extend_from_slice(&co_cpu.data[base..base + v * dm]);
        }
        let packed_gpu = cuda.upload_tensor(&CpuTensor::new(packed, vec![1, n_total, dm]))?;

        // 3. Transformer layers (aligner: full attention, no windowing)
        let mut h = packed_gpu;
        for layer in &self.layers {
            h = layer.forward(cuda, h, None)?;   // <-- None = full attention (aligner)
        }

        // 4. Final projection
        let h = self.ln_post.forward(cuda, &h)?;
        let mut h = self.proj1.forward(cuda, &h)?;
        cuda.gelu_inplace(&mut h)?;
        let h = self.proj2.forward(cuda, &h)?;

        let out = cuda.download_tensor(&h)?;
        let out_dim = out.shape[out.shape.len() - 1];
        Ok((out.data, out_dim))
    }

    /// Run the transformer stack on pre-computed conv output [n_tokens, d_model].
    /// Used when the conv stem runs on burn (legacy path).
    pub fn run_transformer(&self, conv_out: &[f16], n_tokens: usize) -> Result<(Vec<f16>, usize)> {
        let dm = self.config.d_model;

        let h_cpu = CpuTensor::new(conv_out.to_vec(), vec![1, n_tokens, dm]);
        let mut h = self.cuda.upload_tensor(&h_cpu)?;

        for layer in &self.layers {
            h = layer.forward(&self.cuda, h, None)?;   // <-- aligner: full attention
        }

        let h = self.ln_post.forward(&self.cuda, &h)?;
        let mut h = self.proj1.forward(&self.cuda, &h)?;
        self.cuda.gelu_inplace(&mut h)?;
        let h = self.proj2.forward(&self.cuda, &h)?;

        let out = self.cuda.download_tensor(&h)?;
        let out_dim = out.shape[out.shape.len() - 1];
        Ok((out.data, out_dim))
    }
}

fn feo(ifr: usize) -> usize {
    let f = |l: usize| -> usize { (l - 1) / 2 + 1 };
    f(f(f(ifr)))
}
