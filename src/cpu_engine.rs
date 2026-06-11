//! CPU-resident text decoder for the Qwen3-ForcedAligner.
//!
//! Mirror of `cudarc_engine.rs` running on the host CPU:
//!   * `gemm` crate handles every matmul, with `Parallelism::Rayon(0)` forced
//!     even for the m=1 lm_head GEMV (gemm's internal threshold would
//!     otherwise leave decode single-threaded).
//!   * `rayon` parallelises every hand-written elementwise / reduction op
//!     (rms_norm, silu_mul, prefill attention) across heads or rows.
//!   * Tensors are f32 (Vec<f32>) — modern x86 has no native f16 SIMD outside
//!     Sapphire Rapids, so f32 ends up faster than f16 with upcast.
//!   * KV cache is pre-allocated per layer.
//!
//! Audio encoder on CPU is **not yet implemented**; `DeviceRequest::Cpu`
//! currently fails fast at `run_audio_encoder` with a clear "use CUDA for
//! now" message.  The decoder (this file's main body) is the part that's
//! 80% of the runtime for the aligner and is what the CPU path actually
//! exercises.  See the project's handoff.md for the planned audio encoder
//! port.

// `#[allow(dead_code)]` is attached to items that are part of the audio
// encoder CPU port stub (mirroring cudarc_engine.rs's signature, but with
// the body returning an error).  These will be exercised once the audio
// encoder lands.
#![allow(dead_code)]

use anyhow::Result;
use gemm::{gemm, Parallelism};
use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::{AudioConfig, TextConfig};
use crate::weight::WeightTensor;

// ═══════════════════════════════════════════════════════════════════════
//  Host-side f32 tensors
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch (shape {:?})", shape);
        Self { data, shape }
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn numel(&self) -> usize { self.data.len() }
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        assert_eq!(self.numel(), shape.iter().product::<usize>());
        self.shape = shape; self
    }
}

pub(crate) struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,   // = out_features (N)
    pub cols: usize,   // = in_features  (K)
}

// ═══════════════════════════════════════════════════════════════════════
//  Matmul: y = x @ W^T
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn linear(x: &CpuTensor, w: &CpuWeight) -> CpuTensor {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
    let mut out_shape = x.shape.clone();
    out_shape[nd - 1] = n;
    if m == 1 {
        let out = linear_gemv(&x.data, w);
        return CpuTensor::new(out, out_shape);
    }
    let mut out = vec![0.0f32; m * n];
    gemm_row_major(&mut out, &x.data, w, m, 0.0);
    CpuTensor::new(out, out_shape)
}

pub(crate) fn linear_accum(out: &mut CpuTensor, x: &CpuTensor, w: &CpuWeight) {
    let nd = x.shape.len();
    let m: usize = x.shape[..nd - 1].iter().product();
    let k = x.shape[nd - 1];
    let n = w.rows;
    assert_eq!(k, w.cols);
    assert_eq!(out.numel(), m * n);
    if m == 1 {
        let add = linear_gemv(&x.data, w);
        for (o, a) in out.data.iter_mut().zip(add.iter()) { *o += *a; }
        return;
    }
    gemm_row_major(&mut out.data, &x.data, w, m, 1.0);
}

fn gemm_row_major(out: &mut [f32], x: &[f32], w: &CpuWeight, m: usize, beta: f32) {
    let n = w.rows;
    let k = w.cols;
    unsafe {
        gemm(
            m, n, k,
            out.as_mut_ptr(),
            1,                  // dst_cs
            n as isize,         // dst_rs
            beta != 0.0,
            x.as_ptr(),
            1,                  // lhs_cs
            k as isize,         // lhs_rs
            w.data.as_ptr(),
            k as isize,         // rhs_cs (B is W^T; j+1 advances by k in W's row-major layout)
            1,                  // rhs_rs
            beta,
            1.0,
            false, false, false,
            Parallelism::Rayon(0),
        );
    }
}

/// Hand-written m=1 GEMV optimised for the lm_head case.
fn linear_gemv(x: &[f32], w: &CpuWeight) -> Vec<f32> {
    let n = w.rows;
    let k = w.cols;
    debug_assert_eq!(x.len(), k);
    let mut out = vec![0.0f32; n];
    let chunk = (n / (rayon::current_num_threads() * 4)).max(64).min(2048);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, slab)| {
        let row0 = ci * chunk;
        for (offset, o) in slab.iter_mut().enumerate() {
            let row = row0 + offset;
            let w_row = &w.data[row * k..(row + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k { acc += x[j] * w_row[j]; }
            *o = acc;
        }
    });
    out
}

// ═══════════════════════════════════════════════════════════════════════
//  Elementwise / reduction ops
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn rms_norm(x: &CpuTensor, w: &[f32], eps: f32) -> CpuTensor {
    let nd = x.shape.len();
    let last = x.shape[nd - 1];
    let outer: usize = x.shape[..nd - 1].iter().product();
    assert_eq!(w.len(), last);
    let mut out = vec![0.0f32; outer * last];
    out.par_chunks_mut(last)
        .zip(x.data.par_chunks(last))
        .for_each(|(o, xrow)| {
            let mut ss = 0.0f64;
            for &v in xrow { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / last as f64 + eps as f64).sqrt() as f32);
            for j in 0..last {
                o[j] = xrow[j] * inv_rms * w[j];
            }
        });
    CpuTensor::new(out, x.shape.clone())
}

pub(crate) fn silu_mul_split(gu: &CpuTensor) -> CpuTensor {
    let nd = gu.shape.len();
    let two_inter = gu.shape[nd - 1];
    let inter = two_inter / 2;
    let outer: usize = gu.shape[..nd - 1].iter().product();
    let mut out = vec![0.0f32; outer * inter];
    out.par_chunks_mut(inter)
        .zip(gu.data.par_chunks(two_inter))
        .for_each(|(o, row)| {
            let (gate, up) = row.split_at(inter);
            for j in 0..inter {
                let g = gate[j];
                let sig = 1.0 / (1.0 + (-g).exp());
                o[j] = g * sig * up[j];
            }
        });
    let mut shape = gu.shape.clone();
    shape[nd - 1] = inter;
    CpuTensor::new(out, shape)
}

pub(crate) fn embed_lookup(table: &CpuWeight, ids: &[i64]) -> CpuTensor {
    let n = ids.len();
    let d = table.cols;
    let mut out = vec![0.0f32; n * d];
    out.par_chunks_mut(d)
        .zip(ids.par_iter())
        .for_each(|(o, &id)| {
            let src = &table.data[(id as usize) * d..(id as usize + 1) * d];
            o.copy_from_slice(src);
        });
    CpuTensor::new(out, vec![n, d])
}

pub(crate) fn argmax(x: &[f32]) -> i32 {
    const CHUNK: usize = 4096;
    let n = x.len();
    let (idx, _) = (0..n).step_by(CHUNK).collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + CHUNK).min(n);
            let mut best_idx = start;
            let mut best_val = x[start];
            for i in (start + 1)..end {
                if x[i] > best_val { best_val = x[i]; best_idx = i; }
            }
            (best_idx, best_val)
        })
        .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
    idx as i32
}

// ═══════════════════════════════════════════════════════════════════════
//  Rotary embedding (in-place on a head row)
// ═══════════════════════════════════════════════════════════════════════

#[inline]
fn apply_rotary_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let half = d / 2;
    let mut tmp = vec![0.0f32; d];
    tmp.copy_from_slice(x);
    for i in 0..half {
        x[i]        = tmp[i]        * cos[i]        - tmp[i + half] * sin[i];
        x[i + half] = tmp[i + half] * cos[i + half] + tmp[i]        * sin[i + half];
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Fused QKV extract + Q/K norm + rotary + KV cache write
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn qkv_extract_qkv_norm_rotary_cache(
    qkv: &CpuTensor,
    qn_w: &[f32], kn_w: &[f32],
    cos_table: &[f32], sin_table: &[f32],
    k_cache: &mut [f32], v_cache: &mut [f32],
    b: usize, nqh: usize, nkvh: usize, hd: usize,
    q_dim: usize, kv_dim: usize,
    max_seq: usize, start: usize, pos_offset: usize, eps: f32,
) -> CpuTensor {
    let s = qkv.shape[1];
    let total_cols = qkv.shape[2];
    let mut q_out = vec![0.0f32; b * nqh * s * hd];

    q_out.par_chunks_mut(nqh * hd).enumerate().for_each(|(token_idx, q_dst)| {
        let ib = token_idx / s;
        let is = token_idx % s;
        let cs_row = pos_offset + is;
        let cos_row = &cos_table[cs_row * hd..(cs_row + 1) * hd];
        let sin_row = &sin_table[cs_row * hd..(cs_row + 1) * hd];

        let row_base = (ib * s + is) * total_cols;
        let qkv_row = &qkv.data[row_base..row_base + total_cols];
        let q_src = &qkv_row[..q_dim];
        let k_src = &qkv_row[q_dim..q_dim + kv_dim];
        let v_src = &qkv_row[q_dim + kv_dim..];

        for h in 0..nqh {
            let head_in = &q_src[h * hd..(h + 1) * hd];
            let head_out = &mut q_dst[h * hd..(h + 1) * hd];
            let mut ss = 0.0f64;
            for &v in head_in { ss += (v as f64) * (v as f64); }
            let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
            for j in 0..hd { head_out[j] = head_in[j] * inv_rms * qn_w[j]; }
            apply_rotary_row(head_out, cos_row, sin_row);
        }

        // K: per kv-head, RMSNorm(kn_w) → rotary → write to k_cache[ib, h, start+is, :]
        // V: raw copy to v_cache.
        let k_cache_ptr = k_cache.as_ptr() as *mut f32;
        let v_cache_ptr = v_cache.as_ptr() as *mut f32;
        for h in 0..nkvh {
            let k_in = &k_src[h * hd..(h + 1) * hd];
            let v_in = &v_src[h * hd..(h + 1) * hd];
            let cache_idx = ((ib * nkvh + h) * max_seq + (start + is)) * hd;
            unsafe {
                let k_dst = std::slice::from_raw_parts_mut(k_cache_ptr.add(cache_idx), hd);
                let v_dst = std::slice::from_raw_parts_mut(v_cache_ptr.add(cache_idx), hd);
                let mut ss = 0.0f64;
                for &v in k_in { ss += (v as f64) * (v as f64); }
                let inv_rms = 1.0 / ((ss / hd as f64 + eps as f64).sqrt() as f32);
                for j in 0..hd { k_dst[j] = k_in[j] * inv_rms * kn_w[j]; }
                apply_rotary_row(k_dst, cos_row, sin_row);
                v_dst.copy_from_slice(v_in);
            }
        }
    });

    CpuTensor::new(q_out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Prefill attention (s > 1, full causal mask, f32 throughout)
// ═══════════════════════════════════════════════════════════════════════

fn prefill_attention(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    causal: bool,
) -> CpuTensor {
    let s = q.shape[2];
    let n_rep = nqh / nkvh;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let out = vec![0.0f32; b * s * nqh * hd];

    (0..b * nqh).into_par_iter().for_each(|idx| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let k_base = (ib * nkvh + kh) * max_seq * hd;
        let v_base = (ib * nkvh + kh) * max_seq * hd;

        let mut q_qh = vec![0.0f32; s * hd];
        for i in 0..s {
            let src = ((ib * s + i) * nqh + qh) * hd;
            q_qh[i * hd..(i + 1) * hd].copy_from_slice(&q.data[src..src + hd]);
        }

        let mut scores = vec![0.0f32; s * cur_len];
        for i in 0..s {
            let qi = &q_qh[i * hd..(i + 1) * hd];
            let limit = if causal { i + (cur_len - s) + 1 } else { cur_len };
            for t in 0..cur_len {
                if t >= limit {
                    scores[i * cur_len + t] = f32::NEG_INFINITY;
                } else {
                    let kt = &k_cache[k_base + t * hd..k_base + (t + 1) * hd];
                    let mut dot = 0.0f32;
                    for j in 0..hd { dot += qi[j] * kt[j]; }
                    scores[i * cur_len + t] = dot * scale;
                }
            }
        }
        for i in 0..s {
            let row = &mut scores[i * cur_len..(i + 1) * cur_len];
            let mut mx = f32::NEG_INFINITY;
            for &v in row.iter() { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        let out_ptr = out.as_ptr() as *mut f32;
        for i in 0..s {
            let dst_off = ((ib * s + i) * nqh + qh) * hd;
            unsafe {
                let out_i = std::slice::from_raw_parts_mut(out_ptr.add(dst_off), hd);
                for j in 0..hd { out_i[j] = 0.0; }
                let row = &scores[i * cur_len..(i + 1) * cur_len];
                for t in 0..cur_len {
                    let w = row[t];
                    if w == 0.0 { continue; }
                    let vt = &v_cache[v_base + t * hd..v_base + (t + 1) * hd];
                    for j in 0..hd { out_i[j] += w * vt[j]; }
                }
            }
        }
    });

    CpuTensor::new(out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  Audio encoder — stub
// ═══════════════════════════════════════════════════════════════════════

/// The audio encoder is not yet implemented on CPU.  The conv stem
/// (3 × stride-2 conv2d) plus 24 transformer layers is ~2x more work than
/// the 28-layer text decoder, and the per-chunk conv_out reshape that the
/// CUDA path does dynamically is fiddly to port.  See handoff.md for the
/// plan; in the meantime `DeviceRequest::Cpu` fails fast here.
pub(crate) struct CpuAudioEncoder;

impl CpuAudioEncoder {
    pub fn load(_weights: &HashMap<String, WeightTensor>, _prefix: &str, _config: &AudioConfig) -> Result<Self> {
        Ok(Self)
    }
    pub fn run(&self, _mel_packed: &[f16], _b_chunks: usize, _n_mels: usize, _cs: usize) -> Result<(Vec<f16>, usize)> {
        anyhow::bail!(
            "CPU audio encoder is not yet implemented.  Use DeviceRequest::Cuda(n) \
             for now.  See handoff.md for the planned port (conv stem + 24 layers)."
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  KV cache (pre-allocated per layer, f32 host memory)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuKvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub cur_len: usize,
    pub max_seq: usize,
}

impl CpuKvCache {
    pub fn new(num_layers: usize, b: usize, nkvh: usize, max_seq: usize, hd: usize) -> Self {
        let cap = b * nkvh * max_seq * hd;
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(vec![0.0; cap]);
            v.push(vec![0.0; cap]);
        }
        Self { k, v, cur_len: 0, max_seq }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Decoder layer
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuDecoderLayer {
    pub iln_w: Vec<f32>,
    pub pln_w: Vec<f32>,
    pub qn_w: Vec<f32>,
    pub kn_w: Vec<f32>,
    pub qkv_w: CpuWeight,
    pub o_w: CpuWeight,
    pub gu_w: CpuWeight,
    pub dp_w: CpuWeight,
    pub nqh: usize, pub nkvh: usize, pub hd: usize, pub eps: f32,
}

impl CpuDecoderLayer {
    pub fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            iln_w: load_vec(weights, &format!("{}.input_layernorm.weight", prefix))?,
            pln_w: load_vec(weights, &format!("{}.post_attention_layernorm.weight", prefix))?,
            qn_w: load_vec(weights, &format!("{}.self_attn.q_norm.weight", prefix))?,
            kn_w: load_vec(weights, &format!("{}.self_attn.k_norm.weight", prefix))?,
            qkv_w: load_fused_qkv(weights, &format!("{}.self_attn", prefix))?,
            o_w: load_weight(weights, &format!("{}.self_attn.o_proj.weight", prefix))?,
            gu_w: load_fused_gate_up(weights, &format!("{}.mlp", prefix))?,
            dp_w: load_weight(weights, &format!("{}.mlp.down_proj.weight", prefix))?,
            nqh: cfg.num_attention_heads,
            nkvh: cfg.num_key_value_heads,
            hd: cfg.head_dim,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// x: [b, s, hidden] consumed; returns h (post all residuals).
    pub fn forward(
        &self,
        x: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, layer_idx: usize,
        kv_start: usize, use_causal: bool,
    ) -> CpuTensor {
        let b = x.shape[0]; let s = x.shape[1];
        let normed = rms_norm(&x, &self.iln_w, self.eps);
        let qkv = linear(&normed, &self.qkv_w);
        drop(normed);
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        let q = qkv_extract_qkv_norm_rotary_cache(
            &qkv, &self.qn_w, &self.kn_w, cos_table, sin_table,
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            b, self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        );
        drop(qkv);
        let cur_len = kv_start + s;

        let attn_out = prefill_attention(
            &q, &kv.k[layer_idx], &kv.v[layer_idx],
            b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, use_causal,
        );
        // attn_out is laid out as [b, nqh, s, hd] (logical) but bytes are [b, s, nqh, hd].
        // Reshape directly to [b, s, nqh*hd] for the O projection (no swap needed).
        let attn_flat = attn_out.reshape(vec![b, s, self.nqh * self.hd]);
        let mut h = x;
        linear_accum(&mut h, &attn_flat, &self.o_w);
        drop(attn_flat);

        let normed2 = rms_norm(&h, &self.pln_w, self.eps);
        let gu = linear(&normed2, &self.gu_w);
        drop(normed2);
        let activated = silu_mul_split(&gu);
        drop(gu);
        linear_accum(&mut h, &activated, &self.dp_w);
        h
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text decoder
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTextDecoder {
    pub embed_table: CpuWeight,
    pub lm_head: CpuWeight,        // independent, NOT tied to embed
    pub layers: Vec<CpuDecoderLayer>,
    pub norm_w: Vec<f32>,
    pub eps: f32,
}

impl CpuTextDecoder {
    pub fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, config: &TextConfig) -> Result<Self> {
        let embed_table = load_weight(weights, &format!("{}.embed_tokens.weight", prefix))?;
        // lm_head is INDEPENDENT for aligner: the safetensors key is
        // "thinker.lm_head.weight" (sibling of the model namespace), not
        // "thinker.model.lm_head.weight".
        let lm_head = load_weight(weights, "thinker.lm_head.weight")?;
        let norm_w = load_vec(weights, &format!("{}.norm.weight", prefix))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(CpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config)?);
        }
        Ok(Self { embed_table, lm_head, layers, norm_w, eps: config.rms_norm_eps as f32 })
    }

    /// Forward pass.
    /// hs: [1, sl, hidden].  cos/sin_table: full [total_positions, hd] tables.
    /// kv_start: how many positions already in cache.
    /// Returns logits as [1, sl, classify_num] (aligner keeps all positions).
    pub fn forward(
        &self,
        hs: CpuTensor,
        cos_table: &[f32], sin_table: &[f32],
        kv: &mut CpuKvCache, kv_start: usize, use_causal: bool, _llo: bool,
    ) -> CpuTensor {
        let sl = hs.shape[1];
        let mut h = hs;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(h, cos_table, sin_table, kv, i, kv_start, use_causal);
        }
        kv.cur_len = kv_start + sl;

        // Final RMSNorm (aligner wants the full [1, sl, hidden] back, not last-token).
        let h = rms_norm(&h, &self.norm_w, self.eps);
        // lm_head: y = h @ W^T  where h is [1, sl, hidden], W is [classify_num, hidden].
        // m=sl which is > 1 in prefill, so use linear() (the gemm path).
        linear(&h, &self.lm_head)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  MRoPE cos/sin precompute
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn compute_mrope_cos_sin(
    pos: &[Vec<i64>; 3], hd: usize, rt: f64, ms: &[usize], il: bool,
) -> (Vec<f32>, Vec<f32>) {
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
    (cv, sv)
}

fn build_contiguous_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let mut m = Vec::with_capacity(t);
    for (d, &sz) in s.iter().enumerate() { for _ in 0..sz { if m.len() >= t { break; } m.push(d); } }
    while m.len() < t { m.push(s.len() - 1); } m
}

fn build_interleaved_dim_map(s: &[usize], t: usize) -> Vec<usize> {
    let nd = s.len(); let mut m = Vec::with_capacity(t); let mut c = vec![0usize; nd];
    while m.len() < t {
        let pv = m.len();
        for d in 0..nd {
            if m.len() >= t { break; }
            if c[d] < s[d] { m.push(d); c[d] += 1; }
        }
        if m.len() == pv { break; }
    } m
}

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers
// ═══════════════════════════════════════════════════════════════════════

fn load_f32(weights: &HashMap<String, WeightTensor>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    let shape = td.shape.clone();
    let data: Vec<f32> = td.data.clone();
    Ok((data, shape))
}

fn load_vec(weights: &HashMap<String, WeightTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = load_f32(weights, name)?;
    Ok(data)
}

fn load_weight(weights: &HashMap<String, WeightTensor>, name: &str) -> Result<CpuWeight> {
    let (data, shape) = load_f32(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    Ok(CpuWeight { data, rows: shape[0], cols: shape[1] })
}

fn load_fused_qkv(weights: &HashMap<String, WeightTensor>, prefix: &str) -> Result<CpuWeight> {
    let (qw, qs) = load_f32(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = load_f32(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = load_f32(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw);
    fused.extend_from_slice(&kw);
    fused.extend_from_slice(&vw);
    Ok(CpuWeight { data: fused, rows: q_dim + 2 * kv_dim, cols: hidden })
}

fn load_fused_gate_up(weights: &HashMap<String, WeightTensor>, prefix: &str) -> Result<CpuWeight> {
    let (gw, gs) = load_f32(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = load_f32(weights, &format!("{}.up_proj.weight", prefix))?;
    let inter = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], inter); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * inter * hidden);
    fused.extend_from_slice(&gw);
    fused.extend_from_slice(&uw);
    Ok(CpuWeight { data: fused, rows: 2 * inter, cols: hidden })
}
