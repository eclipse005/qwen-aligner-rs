use anyhow::Result;
use burn::tensor::{activation, Bool, Tensor, TensorData};
use burn::tensor::backend::Backend;
use burn::tensor::module::{attention, conv2d};
use burn::tensor::ops::{AttentionModuleOptions, ConvOptions};
use std::collections::HashMap;

use crate::config::AudioConfig;

// ─── Matmul alignment ──────────────────────────────────────────────
const K_ALIGN: usize = 32;

pub(crate) fn safe_attention<B: Backend>(
    q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>, is_causal: bool,
) -> Tensor<B, 4> {
    let n = k.dims()[2];
    let m = q.dims()[2];
    if m > 1 || n % K_ALIGN == 0 {
        let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal };
        return attention(q, k, v, None::<Tensor<B, 4, Bool>>, None, opts);
    }
    let [b, h, _, d] = q.dims();
    let device = q.device();
    let scale = 1.0 / (d as f64).sqrt();
    let scores = q.matmul(k.swap_dims(2, 3)) * scale;
    let attn = activation::softmax(scores, 3);
    let n_padded = ((n + K_ALIGN - 1) / K_ALIGN) * K_ALIGN;
    let pad = n_padded - n;
    let attn_padded = Tensor::cat(vec![attn, Tensor::zeros([b, h, 1, pad], &device)], 3);
    let v_padded = Tensor::cat(vec![v, Tensor::zeros([b, h, pad, d], &device)], 2);
    attn_padded.matmul(v_padded)
}

// ─── Weight helpers ────────────────────────────────────────────────

fn get_w<B: Backend, const D: usize>(
    weights: &HashMap<String, TensorData>, name: &str, device: &B::Device,
) -> Result<Tensor<B, D>> {
    weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
        .map(|d| Tensor::from_data(d.clone(), device))
}

fn load_linear<B: Backend>(
    weights: &HashMap<String, TensorData>, prefix: &str, device: &B::Device,
) -> Result<LinearW<B>> {
    let weight = get_w(weights, &format!("{}.weight", prefix), device)?;
    let bias = weights.get(&format!("{}.bias", prefix))
        .map(|d| Tensor::<B, 1>::from_data(d.clone(), device));
    Ok(LinearW::new(weight, bias))
}

// ─── Linear (generic D-dim matmul + broadcast) ─────────────────────

pub(crate) struct LinearW<B: Backend> {
    weight_t: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> LinearW<B> {
    pub fn new(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self { weight_t: weight.transpose(), bias }
    }

    pub fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let wd = self.weight_t.dims();
        let mut ws = [1; D];
        ws[D - 2] = wd[0];
        ws[D - 1] = wd[1];
        let out = x.clone().matmul(self.weight_t.clone().reshape(ws));
        match &self.bias {
            Some(b) => {
                let bd = b.dims();
                let mut bs = [1; D];
                bs[D - 1] = bd[0];
                out + b.clone().reshape(bs)
            }
            None => out,
        }
    }
}

// ─── Conv2d ────────────────────────────────────────────────────────

fn conv2d_forward<B: Backend>(
    input: &Tensor<B, 4>, weight: &Tensor<B, 4>, bias: Option<&Tensor<B, 1>>,
    stride: usize, padding: usize,
) -> Tensor<B, 4> {
    let opts = ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1);
    conv2d(input.clone(), weight.clone(), bias.cloned(), opts)
}

// ─── Manual LayerNorm ──────────────────────────────────────────────

struct ManualLayerNorm<B: Backend> {
    weight: Tensor<B, 1>, bias: Tensor<B, 1>, eps: f64, size: usize,
}

impl<B: Backend> ManualLayerNorm<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, size: usize, eps: f64, device: &B::Device) -> Result<Self> {
        Ok(Self {
            weight: get_w(weights, &format!("{}.weight", prefix), device)?,
            bias: get_w(weights, &format!("{}.bias", prefix), device)?,
            eps, size,
        })
    }

    fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let last = D - 1;
        let mean = x.clone().mean_dim(last);
        let var = x.clone().var_bias(last);
        let xn = (x.clone() - mean) / (var + self.eps).sqrt();
        let mut ws = [1; D];
        ws[D - 1] = self.size;
        xn * self.weight.clone().reshape(ws) + self.bias.clone().reshape(ws)
    }
}

// ─── Fused QKV Linear (3 matmuls → 1) ────────────────────────────

struct FusedAudioQkv<B: Backend> {
    weight_t: Tensor<B, 3>, // [1, dm, 3*dm] (pre-transposed, pre-reshaped)
    bias_q: Option<Tensor<B, 1>>, bias_k: Option<Tensor<B, 1>>, bias_v: Option<Tensor<B, 1>>,
    dm: usize,
}

impl<B: Backend> FusedAudioQkv<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, dm: usize, device: &B::Device) -> Result<Self> {
        let qw: Tensor<B, 2> = get_w(weights, &format!("{}.q_proj.weight", prefix), device)?;
        let kw: Tensor<B, 2> = get_w(weights, &format!("{}.k_proj.weight", prefix), device)?;
        let vw: Tensor<B, 2> = get_w(weights, &format!("{}.v_proj.weight", prefix), device)?;
        let bq = weights.get(&format!("{}.q_proj.bias", prefix)).map(|d| Tensor::<B, 1>::from_data(d.clone(), device));
        let bk = weights.get(&format!("{}.k_proj.bias", prefix)).map(|d| Tensor::<B, 1>::from_data(d.clone(), device));
        let bv = weights.get(&format!("{}.v_proj.bias", prefix)).map(|d| Tensor::<B, 1>::from_data(d.clone(), device));
        let fused = Tensor::cat(vec![qw, kw, vw], 0); // [3*dm, dm]
        let wt = fused.transpose().reshape([1, dm, 3 * dm]); // [1, dm, 3*dm]
        Ok(Self { weight_t: wt, bias_q: bq, bias_k: bk, bias_v: bv, dm })
    }

    fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, _] = x.dims();
        let dm = self.dm;
        let qkv = x.clone().matmul(self.weight_t.clone()); // [b, s, 3*dm]
        let q = qkv.clone().slice([0..b, 0..s, 0..dm]);
        let k = qkv.clone().slice([0..b, 0..s, dm..2*dm]);
        let v = qkv.slice([0..b, 0..s, 2*dm..3*dm]);
        // Add biases
        let q = match &self.bias_q { Some(b) => q + b.clone().reshape([1, 1, dm]), None => q };
        let k = match &self.bias_k { Some(b) => k + b.clone().reshape([1, 1, dm]), None => k };
        let v = match &self.bias_v { Some(b) => v + b.clone().reshape([1, 1, dm]), None => v };
        (q, k, v)
    }
}

// ─── Audio Self-Attention (no windowed attention for aligner) ──────

struct AudioAttention<B: Backend> {
    fused_qkv: FusedAudioQkv<B>, out_proj: LinearW<B>,
    num_heads: usize, head_dim: usize,
}

impl<B: Backend> AudioAttention<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize, device: &B::Device) -> Result<Self> {
        Ok(Self {
            fused_qkv: FusedAudioQkv::load(weights, prefix, dm, device)?,
            out_proj: load_linear(weights, &format!("{}.out_proj", prefix), device)?,
            num_heads: nh, head_dim: dm / nh,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let nh = self.num_heads; let hd = self.head_dim;
        let (q, k, v) = self.fused_qkv.forward(x);
        let q = q.reshape([b, s, nh, hd]).swap_dims(1, 2);
        let k = k.reshape([b, s, nh, hd]).swap_dims(1, 2);
        let v = v.reshape([b, s, nh, hd]).swap_dims(1, 2);
        // Aligner uses full attention (no windowing)
        let out = safe_attention(q, k, v, false);
        self.out_proj.forward(&out.swap_dims(1, 2).reshape([b, s, nh * hd]))
    }
}

// ─── Audio FFN ─────────────────────────────────────────────────────

struct AudioFfn<B: Backend> { fc1: LinearW<B>, fc2: LinearW<B> }

impl<B: Backend> AudioFfn<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, device: &B::Device) -> Result<Self> {
        Ok(Self {
            fc1: load_linear(weights, &format!("{}.fc1", prefix), device)?,
            fc2: load_linear(weights, &format!("{}.fc2", prefix), device)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(&activation::gelu(self.fc1.forward(x)))
    }
}

// ─── Audio Encoder Layer ───────────────────────────────────────────

struct AudioEncoderLayer<B: Backend> {
    sln: ManualLayerNorm<B>, attn: AudioAttention<B>, fln: ManualLayerNorm<B>, ffn: AudioFfn<B>,
}

impl<B: Backend> AudioEncoderLayer<B> {
    fn load(weights: &HashMap<String, TensorData>, prefix: &str, nh: usize, dm: usize, device: &B::Device) -> Result<Self> {
        Ok(Self {
            sln: ManualLayerNorm::load(weights, &format!("{}.self_attn_layer_norm", prefix), dm, 1e-5, device)?,
            attn: AudioAttention::load(weights, &format!("{}.self_attn", prefix), nh, dm, device)?,
            fln: ManualLayerNorm::load(weights, &format!("{}.final_layer_norm", prefix), dm, 1e-5, device)?,
            ffn: AudioFfn::load(weights, prefix, device)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(&self.sln.forward(x));
        x.clone() + self.ffn.forward(&self.fln.forward(&x))
    }
}

// ─── Sinusoidal PE ─────────────────────────────────────────────────

fn create_sinusoidal_embedding<B: Backend>(max_len: usize, dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let half = dim / 2;
    let lt = (10000.0f64).ln() / (half as f64 - 1.0);
    let mut e = vec![0.0f32; max_len * dim];
    for p in 0..max_len {
        for i in 0..half {
            let a = p as f64 * (-(i as f64) * lt).exp();
            e[p * dim + i] = a.sin() as f32;
            e[p * dim + half + i] = a.cos() as f32;
        }
    }
    Tensor::from_data(TensorData::new(e, [max_len, dim]), device)
}

// ─── Audio Encoder ─────────────────────────────────────────────────

const CONV_BATCH_SIZE: usize = 64;

pub(crate) struct AudioEncoder<B: Backend> {
    c1w: Tensor<B, 4>, c1b: Tensor<B, 1>,
    c2w: Tensor<B, 4>, c2b: Tensor<B, 1>,
    c3w: Tensor<B, 4>, c3b: Tensor<B, 1>,
    co: LinearW<B>, pe: Tensor<B, 2>,
    layers: Vec<AudioEncoderLayer<B>>,
    lnp: ManualLayerNorm<B>, p1: LinearW<B>, p2: LinearW<B>,
    config: AudioConfig,
}

impl<B: Backend> AudioEncoder<B> {
    pub(crate) fn load(
        weights: &HashMap<String, TensorData>, prefix: &str,
        config: &AudioConfig, device: &B::Device,
    ) -> Result<Self> {
        let dm = config.d_model;
        let c1w = get_w(weights, &format!("{}.conv2d1.weight", prefix), device)?;
        let c1b = get_w(weights, &format!("{}.conv2d1.bias", prefix), device)?;
        let c2w = get_w(weights, &format!("{}.conv2d2.weight", prefix), device)?;
        let c2b = get_w(weights, &format!("{}.conv2d2.bias", prefix), device)?;
        let c3w = get_w(weights, &format!("{}.conv2d3.weight", prefix), device)?;
        let c3b = get_w(weights, &format!("{}.conv2d3.bias", prefix), device)?;
        let co = load_linear(weights, &format!("{}.conv_out", prefix), device)?;
        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                weights, &format!("{}.layers.{}", prefix, i),
                config.encoder_attention_heads, dm, device,
            )?);
        }
        let lnp = ManualLayerNorm::load(weights, &format!("{}.ln_post", prefix), dm, 1e-5, device)?;
        let p1 = load_linear(weights, &format!("{}.proj1", prefix), device)?;
        let p2 = load_linear(weights, &format!("{}.proj2", prefix), device)?;
        let pe = create_sinusoidal_embedding(config.max_source_positions, dm, device);
        Ok(Self { c1w, c1b, c2w, c2b, c3w, c3b, co, pe, layers, lnp, p1, p2, config: config.clone() })
    }

    /// Conv output frame count formula: 3 stride-2 convs.
    pub(crate) fn feo(ifr: usize) -> usize {
        let f = |l: usize| -> usize { (l - 1) / 2 + 1 };
        f(f(f(ifr)))
    }

    /// Run conv stem with batch processing (CONV_BATCH_SIZE chunks at a time).
    /// Returns flattened audio tokens [n_total_tokens, d_model].
    fn run_conv_stem(&self, mel: &Tensor<B, 2>, cs: usize) -> Tensor<B, 2> {
        let nf = mel.dims()[1]; // number of frames
        let tpc = Self::feo(cs); // tokens per chunk
        let nfull = nf / cs;
        let tail = nf % cs;
        let n_chunks = nfull + if tail > 0 { 1 } else { 0 };

        // Build valid token counts per chunk
        let mut chunk_valid_tokens: Vec<usize> = Vec::with_capacity(n_chunks);
        for _ in 0..nfull { chunk_valid_tokens.push(tpc); }
        if tail > 0 { chunk_valid_tokens.push(Self::feo(tail)); }

        // Process conv in batches to avoid GPU OOM
        let mut all_tokens: Vec<Tensor<B, 2>> = Vec::new();

        for batch_start in (0..n_chunks).step_by(CONV_BATCH_SIZE) {
            let batch_end = (batch_start + CONV_BATCH_SIZE).min(n_chunks);

            // Build batch of mel chunks: each [1, mel_bins, cs], then cat to [batch, mel_bins, cs]
            let mut chunk_mels: Vec<Tensor<B, 3>> = Vec::new();
            for i in batch_start..batch_end {
                let s = i * cs;
                if i < nfull {
                    chunk_mels.push(mel.clone().slice([0..mel.dims()[0], s..s + cs]).unsqueeze_dim::<3>(0));
                } else {
                    // Tail chunk: pad with zeros
                    let tm = mel.clone().slice([0..mel.dims()[0], s..s + tail]);
                    let pad = Tensor::zeros([mel.dims()[0], cs - tail], &mel.device());
                    chunk_mels.push(Tensor::cat(vec![tm, pad], 1).unsqueeze_dim::<3>(0));
                }
            }

            // [batch, mel_bins, cs] → [batch, 1, mel_bins, cs]
            let batch = Tensor::cat(chunk_mels, 0).unsqueeze_dim::<4>(1);

            // Conv stem: 3x Conv2d(stride=2, pad=1) + GELU
            let x = activation::gelu(conv2d_forward(&batch, &self.c1w, Some(&self.c1b), 2, 1));
            let x = activation::gelu(conv2d_forward(&x, &self.c2w, Some(&self.c2b), 2, 1));
            let x = activation::gelu(conv2d_forward(&x, &self.c3w, Some(&self.c3b), 2, 1));

            let [b2, c2, f2, t2] = x.dims();
            // Permute to [batch, time, channels*freq] then project
            let r = x.permute([0, 3, 1, 2]).reshape([b2, t2, c2 * f2]);
            let co = self.co.forward(&r); // [batch, t2, dm]

            // Add sinusoidal PE
            let pe = self.pe.clone().slice([0..t2]).unsqueeze_dim::<3>(0);
            let co = co + pe;

            // Extract valid tokens per chunk
            for (idx, &v) in chunk_valid_tokens[batch_start..batch_end].iter().enumerate() {
                all_tokens.push(co.clone().slice([idx..idx + 1, 0..v]).squeeze_dim(0));
            }
        }

        Tensor::cat(all_tokens, 0) // [n_total_tokens, dm]
    }

    /// Full forward: mel → conv_stem → transformer → projection
    pub(crate) fn forward(&self, mel: &Tensor<B, 2>) -> Result<Tensor<B, 2>> {
        let cs = self.config.n_window * 2;
        let h = self.run_conv_stem(mel, cs).unsqueeze_dim::<3>(0); // [1, n_tokens, dm]

        // 24 transformer encoder layers (full attention, no windowing)
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h);
        }

        // Post projection: LayerNorm → Linear(GELU) → Linear
        let h = self.lnp.forward(&h);
        let h = self.p2.forward(&activation::gelu(self.p1.forward(&h)));
        Ok(h.squeeze_dim(0)) // [n_tokens, dm]
    }
}
