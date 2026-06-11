use anyhow::Result;
use burn::tensor::{activation, DType, Int, Tensor, TensorData};
use burn::tensor::backend::Backend;
use std::collections::HashMap;

use crate::config::TextConfig;
use crate::encoder::{safe_attention, LinearW};

// ─── Weight helpers ────────────────────────────────────────────────

fn get_w<B: Backend, const D: usize>(
    weights: &HashMap<String, TensorData>, name: &str, device: &B::Device,
) -> Result<Tensor<B, D>> {
    weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
        .map(|d| Tensor::from_data(d.clone(), device))
}

fn load_linear_no_bias<B: Backend>(
    weights: &HashMap<String, TensorData>, prefix: &str, device: &B::Device,
) -> Result<LinearW<B>> {
    Ok(LinearW::new(get_w(weights, &format!("{}.weight", prefix), device)?, None))
}

// ─── Manual RmsNorm (f32-precision rms computation) ────────────────

pub(crate) struct ManualRmsNorm<B: Backend> {
    weight: Tensor<B, 1>, eps: f64, size: usize,
}

impl<B: Backend> ManualRmsNorm<B> {
    pub fn load(weights: &HashMap<String, TensorData>, prefix: &str, size: usize, eps: f64, device: &B::Device) -> Result<Self> {
        Ok(Self {
            weight: get_w(weights, &format!("{}.weight", prefix), device)?,
            eps, size,
        })
    }

    pub fn forward<const D: usize>(&self, x: &Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let last = D - 1;
        let rms = (x.clone().cast(DType::F32).square().mean_dim(last) + self.eps).sqrt();
        let mut ws = [1; D];
        ws[D - 1] = self.size;
        (x.clone() / rms.cast(dtype)) * self.weight.clone().reshape(ws)
    }
}

// ─── MRoPE ─────────────────────────────────────────────────────────

pub(crate) fn compute_mrope_cos_sin<B: Backend>(
    pos: &[Vec<i64>; 3], hd: usize, rt: f64, ms: &[usize], il: bool, device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let hh = hd / 2;
    let sl = pos[0].len();
    let inv: Vec<f64> = (0..hh).map(|i| 1.0 / rt.powf(2.0 * i as f64 / hd as f64)).collect();
    let dm = if il { build_interleaved_dim_map(ms, hh) } else { build_contiguous_dim_map(ms, hh) };
    let mut cv = vec![0.0f32; sl * hd];
    let mut sv = vec![0.0f32; sl * hd];
    for t in 0..sl {
        for j in 0..hh {
            let a = pos[dm[j]][t] as f64 * inv[j];
            cv[t * hd + j] = a.cos() as f32;
            sv[t * hd + j] = a.sin() as f32;
            cv[t * hd + j + hh] = a.cos() as f32;
            sv[t * hd + j + hh] = a.sin() as f32;
        }
    }
    (
        Tensor::from_data(TensorData::new(cv, [sl, hd]), device),
        Tensor::from_data(TensorData::new(sv, [sl, hd]), device),
    )
}

fn build_contiguous_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let mut m = Vec::with_capacity(t);
    for (d, &sz) in s.iter().enumerate() {
        for _ in 0..sz { if m.len() >= t { break; } m.push(d); }
    }
    while m.len() < t { m.push(s.len() - 1); }
    m
}

fn build_interleaved_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let nd = s.len();
    let mut m = Vec::with_capacity(t);
    let mut c = vec![0usize; nd];
    while m.len() < t {
        let pv = m.len();
        for d in 0..nd {
            if m.len() >= t { break; }
            if c[d] < s[d] { m.push(d); c[d] += 1; }
        }
        if m.len() == pv { break; }
    }
    m
}

fn apply_rotary_emb<B: Backend>(x: &Tensor<B, 4>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>) -> Tensor<B, 4> {
    let xr = rotate_half(x);
    x.clone() * cos.clone() + xr * sin.clone()
}

fn rotate_half<B: Backend>(x: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [b0, b1, b2, l] = x.dims();
    let h = l / 2;
    let x1 = x.clone().slice([0..b0, 0..b1, 0..b2, 0..h]);
    let x2 = x.clone().slice([0..b0, 0..b1, 0..b2, h..l]);
    Tensor::cat(vec![x2 * (-1.0f64), x1], 3)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    if n == 1 { return x; }
    let [b, nkv, s, hd] = x.dims();
    x.unsqueeze_dim::<5>(2).expand([b, nkv, n, s, hd]).reshape([b, nkv * n, s, hd])
}

// ─── Fused QKV Linear ─────────────────────────────────────────────

struct FusedQkv<B: Backend> {
    weight_t: Tensor<B, 3>,
    q_dim: usize, kv_dim: usize,
}

impl<B: Backend> FusedQkv<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, device: &B::Device) -> Result<Self> {
        let qw: Tensor<B, 2> = get_w(w, &format!("{}.q_proj.weight", p), device)?;
        let kw: Tensor<B, 2> = get_w(w, &format!("{}.k_proj.weight", p), device)?;
        let vw: Tensor<B, 2> = get_w(w, &format!("{}.v_proj.weight", p), device)?;
        let fused = Tensor::cat(vec![qw, kw, vw], 0);
        let [out_dim, inp_dim] = fused.dims();
        Ok(Self {
            weight_t: fused.transpose().reshape([1, inp_dim, out_dim]),
            q_dim: nqh * hd, kv_dim: nkvh * hd,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, _] = x.dims();
        let mut ws = [1; 3];
        ws[1] = self.weight_t.dims()[1];
        ws[2] = self.weight_t.dims()[2];
        let qkv = x.clone().matmul(self.weight_t.clone().reshape(ws));
        let q = qkv.clone().slice([0..b, 0..s, 0..self.q_dim]);
        let k = qkv.clone().slice([0..b, 0..s, self.q_dim..self.q_dim + self.kv_dim]);
        let v = qkv.clone().slice([0..b, 0..s, self.q_dim + self.kv_dim..self.q_dim + 2 * self.kv_dim]);
        (q, k, v)
    }
}

// ─── Fused Gate-Up Linear ─────────────────────────────────────────

struct FusedGateUp<B: Backend> {
    weight_t: Tensor<B, 3>,
    intermediate_size: usize,
}

impl<B: Backend> FusedGateUp<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, device: &B::Device) -> Result<Self> {
        let gw: Tensor<B, 2> = get_w(w, &format!("{}.gate_proj.weight", p), device)?;
        let uw: Tensor<B, 2> = get_w(w, &format!("{}.up_proj.weight", p), device)?;
        let fused = Tensor::cat(vec![gw, uw], 0);
        let [out_dim, inp_dim] = fused.dims();
        Ok(Self {
            weight_t: fused.transpose().reshape([1, inp_dim, out_dim]),
            intermediate_size: out_dim / 2,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, s, _] = x.dims();
        let mut ws = [1; 3];
        ws[1] = self.weight_t.dims()[1];
        ws[2] = self.weight_t.dims()[2];
        let gu = x.clone().matmul(self.weight_t.clone().reshape(ws));
        let gate = gu.clone().slice([0..b, 0..s, 0..self.intermediate_size]);
        let up = gu.clone().slice([0..b, 0..s, self.intermediate_size..2 * self.intermediate_size]);
        (gate, up)
    }
}

// ─── Text Attention (no KV cache for aligner) ──────────────────────

struct TextAttention<B: Backend> {
    qkv: FusedQkv<B>, op: LinearW<B>,
    qn: ManualRmsNorm<B>, kn: ManualRmsNorm<B>,
    nqh: usize, nkvh: usize, hd: usize,
}

impl<B: Backend> TextAttention<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, eps: f64, d: &B::Device) -> Result<Self> {
        Ok(Self {
            qkv: FusedQkv::load(w, p, nqh, nkvh, hd, d)?,
            op: load_linear_no_bias(w, &format!("{}.o_proj", p), d)?,
            qn: ManualRmsNorm::load(w, &format!("{}.q_norm", p), hd, eps, d)?,
            kn: ManualRmsNorm::load(w, &format!("{}.k_norm", p), hd, eps, d)?,
            nqh, nkvh, hd,
        })
    }

    fn forward(&self, x: &Tensor<B, 3>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>, use_causal: bool) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (q, k, v) = self.qkv.forward(x);
        let q = q.reshape([b, s, self.nqh, self.hd]).swap_dims(1, 2);
        let k = k.reshape([b, s, self.nkvh, self.hd]).swap_dims(1, 2);
        let v = v.reshape([b, s, self.nkvh, self.hd]).swap_dims(1, 2);
        let q = apply_rotary_emb(&self.qn.forward(&q), cos, sin);
        let k = apply_rotary_emb(&self.kn.forward(&k), cos, sin);

        let nr = self.nqh / self.nkvh;
        let k_rep = repeat_kv(k, nr);
        let v_rep = repeat_kv(v, nr);

        let out = safe_attention(q, k_rep, v_rep, use_causal && s > 1);
        self.op.forward(&out.swap_dims(1, 2).reshape([b, s, self.nqh * self.hd]))
    }
}

// ─── SwiGLU MLP ────────────────────────────────────────────────────

struct TextMlp<B: Backend> {
    gate_up: FusedGateUp<B>, dp: LinearW<B>,
}

impl<B: Backend> TextMlp<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, d: &B::Device) -> Result<Self> {
        Ok(Self {
            gate_up: FusedGateUp::load(w, p, d)?,
            dp: load_linear_no_bias(w, &format!("{}.down_proj", p), d)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let (gate, up) = self.gate_up.forward(x);
        self.dp.forward(&(activation::silu(gate) * up))
    }
}

// ─── Text Decoder Layer ────────────────────────────────────────────

struct TextDecoderLayer<B: Backend> {
    iln: ManualRmsNorm<B>, attn: TextAttention<B>, pln: ManualRmsNorm<B>, mlp: TextMlp<B>,
}

impl<B: Backend> TextDecoderLayer<B> {
    fn load(w: &HashMap<String, TensorData>, p: &str, nqh: usize, nkvh: usize, hd: usize, hs: usize, eps: f64, d: &B::Device) -> Result<Self> {
        Ok(Self {
            iln: ManualRmsNorm::load(w, &format!("{}.input_layernorm", p), hs, eps, d)?,
            attn: TextAttention::load(w, &format!("{}.self_attn", p), nqh, nkvh, hd, eps, d)?,
            pln: ManualRmsNorm::load(w, &format!("{}.post_attention_layernorm", p), hs, eps, d)?,
            mlp: TextMlp::load(w, &format!("{}.mlp", p), d)?,
        })
    }
    fn forward(&self, x: &Tensor<B, 3>, cos: &Tensor<B, 4>, sin: &Tensor<B, 4>, use_causal: bool) -> Tensor<B, 3> {
        let normed = self.iln.forward(x);
        let h = self.attn.forward(&normed, cos, sin, use_causal);
        let x = x.clone() + h;
        x.clone() + self.mlp.forward(&self.pln.forward(&x))
    }
}

// ─── Text Decoder ─────────────────────────────────────────────────

pub(crate) struct TextDecoder<B: Backend> {
    pub embed_tokens: Tensor<B, 2>,
    layers: Vec<TextDecoderLayer<B>>,
    pub norm: ManualRmsNorm<B>,
    pub lm_head: LinearW<B>,
}

impl<B: Backend> TextDecoder<B> {
    pub(crate) fn load(
        weights: &HashMap<String, TensorData>, prefix: &str,
        config: &TextConfig, device: &B::Device,
    ) -> Result<Self> {
        let et: Tensor<B, 2> = get_w(weights, &format!("{}.embed_tokens.weight", prefix), device)?;
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(TextDecoderLayer::load(
                weights, &format!("{}.layers.{}", prefix, i),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, config.hidden_size, config.rms_norm_eps, device,
            )?);
        }
        let norm = ManualRmsNorm::load(
            weights, &format!("{}.norm", prefix),
            config.hidden_size, config.rms_norm_eps, device,
        )?;
        // lm_head uses INDEPENDENT weight: thinker.lm_head.weight [classify_num, hidden_size]
        let lm_head_w: Tensor<B, 2> = get_w(weights, "thinker.lm_head.weight", device)?;
        let lm_head = LinearW::new(lm_head_w, None);

        Ok(Self { embed_tokens: et, layers, norm, lm_head })
    }

    pub(crate) fn embed(&self, input_ids: &Tensor<B, 1, Int>) -> Tensor<B, 2> {
        self.embed_tokens.clone().select(0, input_ids.clone())
    }

    /// Single-pass forward through all layers. Returns hidden states [1, seq_len, hidden].
    pub(crate) fn forward_hidden(
        &self, hs: &Tensor<B, 3>, cos: &Tensor<B, 2>, sin: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let cos4 = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin4 = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let mut h = hs.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &cos4, &sin4, true); // causal mask
        }
        h
    }
}

/// Extract timestamp logits from hidden states.
/// hidden_states: [1, seq_len, hidden_size]
/// positions: indices of timestamp tokens in the sequence
/// Returns logits as f32 vec, (n_timestamps, classify_num)
pub(crate) fn extract_timestamp_logits<B: Backend>(
    hidden_states: &Tensor<B, 3>,
    positions: &[usize],
    norm: &ManualRmsNorm<B>,
    lm_head: &LinearW<B>,
) -> (Vec<f32>, usize, usize) {
    let [_, seq_len, hidden] = hidden_states.dims();
    let n_pos = positions.len();

    // Use GPU-side gather: flatten to [seq_len, hidden], then index_select
    let hs2d = hidden_states.clone().reshape([seq_len, hidden]); // [seq_len, hidden]
    let device = hidden_states.device();

    // Build index tensor for timestamp positions
    let idx_data: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let idx = Tensor::<B, 1, Int>::from_data(TensorData::new(idx_data, [n_pos]), &device);

    // Gather selected rows on GPU
    let selected = hs2d.select(0, idx); // [n_pos, hidden]
    let selected = selected.unsqueeze_dim::<3>(0); // [1, n_pos, hidden]

    // Apply final RMSNorm
    let sel = norm.forward(&selected);

    // Apply lm_head
    let logits = lm_head.forward(&sel); // [1, n_pos, classify_num]
    let classify_num = logits.dims()[2];

    // Download logits to CPU for argmax
    let logits_data = logits.into_data();
    let logits_vals: Vec<f32> = logits_data.to_vec::<f32>().unwrap_or_else(|_| {
        logits_data.to_vec::<half::f16>().expect("logits dtype")
            .into_iter().map(|v| v.to_f32()).collect()
    });

    (logits_vals, n_pos, classify_num)
}
