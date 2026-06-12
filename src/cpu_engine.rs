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
    let mut out = if m == 1 {
        linear_gemv(&x.data, w)
    } else {
        let mut o = vec![0.0f32; m * n];
        gemm_row_major(&mut o, &x.data, w, m, 0.0);
        o
    };
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

/// LayerNorm (with affine, no residual).  out[i, j] = (x[i, j] - mean) / sqrt(var + eps) * w[j] + b[j].
pub(crate) fn layer_norm(x: &CpuTensor, w: &[f32], b: &[f32], eps: f32) -> CpuTensor {
    let nd = x.shape.len();
    let last = x.shape[nd - 1];
    let outer: usize = x.shape[..nd - 1].iter().product();
    assert_eq!(w.len(), last);
    assert_eq!(b.len(), last);
    let mut out = vec![0.0f32; outer * last];
    out.par_chunks_mut(last)
        .zip(x.data.par_chunks(last))
        .for_each(|(o, xrow)| {
            let mut mean = 0.0f64;
            for &v in xrow { mean += v as f64; }
            mean /= last as f64;
            let mut var = 0.0f64;
            for &v in xrow { let d = v as f64 - mean; var += d * d; }
            var /= last as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt() as f32;
            for j in 0..last {
                let v = xrow[j] as f64;
                o[j] = (((v - mean) * inv_std as f64) as f32) * w[j] + b[j];
            }
        });
    CpuTensor::new(out, x.shape.clone())
}

/// out[i] = a[i] + b[i], broadcasts over matching shapes.
pub(crate) fn add(a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
    assert_eq!(a.shape, b.shape);
    let mut out = vec![0.0f32; a.numel()];
    out.par_iter_mut()
        .zip(a.data.par_iter().zip(b.data.par_iter()))
        .for_each(|(o, (x, y))| *o = x + y);
    CpuTensor::new(out, a.shape.clone())
}

/// a += b, in place.
pub(crate) fn add_inplace(a: &mut CpuTensor, b: &CpuTensor) {
    assert_eq!(a.shape, b.shape);
    a.data.par_iter_mut()
        .zip(b.data.par_iter())
        .for_each(|(x, y)| *x += *y);
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

/// Swap dims 1 and 2 of a 4D tensor: [d0, d1, d2, d3] → [d0, d2, d1, d3].
pub(crate) fn swap_dims_12(x: &CpuTensor) -> CpuTensor {
    assert_eq!(x.shape.len(), 4);
    let (d0, d1, d2, d3) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let src = x.data.clone();
    let mut out = vec![0.0f32; d0 * d2 * d1 * d3];
    // Parallelise over (i0, i2): each job writes a d1*d3 slab into `out`.
    // Within the slab, the inner i1 loop writes d3 bytes per iter at the
    // correct offset (i1 * d3) — NOT into the whole slab at once (which
    // would mismatch sizes).
    out.par_chunks_mut(d1 * d3).enumerate().for_each(|(idx, slab)| {
        let i0 = idx / d2;
        let i2 = idx % d2;
        for i1 in 0..d1 {
            let src_off = ((i0 * d1 + i1) * d2 + i2) * d3;
            let dst_in_slab = i1 * d3;
            slab[dst_in_slab..dst_in_slab + d3]
                .copy_from_slice(&src[src_off..src_off + d3]);
        }
    });
    CpuTensor::new(out, vec![d0, d2, d1, d3])
}

/// Q @ K^T for [b, nh, s, hd] layout, returns [b, nh, s, s].
/// Per (b, nh), one gemm call (m=s, n=s, k=hd).
pub(crate) fn matmul_qk(q: &CpuTensor, k: &CpuTensor) -> CpuTensor {
    let qs = q.shape();
    let (b, nh, s, hd) = (qs[0], qs[1], qs[2], qs[3]);
    let mut out = vec![0.0f32; b * nh * s * s];
    out.par_chunks_mut(s * s).enumerate().for_each(|(idx, slab)| {
        let ib = idx / nh;
        let ih = idx % nh;
        let q_off = (ib * nh + ih) * s * hd;
        let k_off = (ib * nh + ih) * s * hd;
        // gemm: m=s, n=s, k=hd.  Output is row-major [s, s]: out[i, j] = sum_k q[i, k] * k[j, k].
        // cs for j+1 is 1, rs for i+1 is s.  For k (b^T, i.e. transposed), element (i, j) = k[j, i]?
        // — no, gemm computes A @ B (no transpose).  We want Q @ K^T, so B is conceptually
        // [hd, s] with element (i, j) = K[j, i].  Since K is row-major [s, hd] with
        // element (i, k) at i*hd + k, the B^T of that is [hd, s] with element (k, j) = K[j, k].
        // We pass rhs as a [hd, s] view: rhs_cs=1, rhs_rs=hd is wrong for that; actually
        // for rhs as [hd, s] row-major, element (i, j) = i*s + j: i+1 advances s, j+1 advances 1.
        // K (row-major [s, hd]) data layout is identical to K^T ([hd, s] col-major) only if we
        // swap cs and rs.  Concretely: pass rhs as the K data pointer with cs=hd, rs=1, which
        // is what a transposed matrix view looks like in BLAS.
        unsafe {
            gemm(
                s, s, hd,
                slab.as_mut_ptr(),
                1, s as isize,
                false,
                q.data.as_ptr().add(q_off),
                1, hd as isize,
                k.data.as_ptr().add(k_off),
                hd as isize, 1,
                0.0, 1.0,
                false, false, false,
                Parallelism::None,
            );
        }
    });
    CpuTensor::new(out, vec![b, nh, s, s])
}

/// Attention @ V: [b, nh, s, t] × [b, nh, t, hd] → [b, nh, s, hd] (bytes laid out
/// [b, s, nh, hd] so the caller can reshape directly to [b, s, nqh*hd]).
pub(crate) fn matmul_av(attn: &CpuTensor, v: &CpuTensor) -> CpuTensor {
    let vs = v.shape();
    let (b, nh, t, hd) = (vs[0], vs[1], vs[2], vs[3]);
    let s = attn.shape()[2];
    let mut out = vec![0.0f32; b * s * nh * hd];
    out.par_chunks_mut(s * hd).enumerate().for_each(|(bn, slab)| {
        let ib = bn / nh;
        let ih = bn % nh;
        let a_off = (ib * nh + ih) * s * t;
        let v_off = (ib * nh + ih) * t * hd;
        // m=s, n=hd, k=t.  Output is row-major [s, hd] with strides (1, hd).
        // V is row-major [t, hd] with strides (1, hd), so pass as-is.
        unsafe {
            gemm(
                s, hd, t,
                slab.as_mut_ptr(),
                1, hd as isize,
                false,
                attn.data.as_ptr().add(a_off),
                1, t as isize,
                v.data.as_ptr().add(v_off),
                1, hd as isize,
                0.0, 1.0,
                false, false, false,
                Parallelism::None,
            );
        }
    });
    CpuTensor::new(out, vec![b, nh, s, hd])
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
//  Audio encoder — full CPU port mirroring gpu_audio_encoder.rs
// ═══════════════════════════════════════════════════════════════════════
//
// Architecture:
//   conv2d stem (3 × 3x3 s=2 p=1 + GELU) → conv_out (Linear) + sinusoidal PE
//   → 24 × { LayerNorm + Self-attention (full) + LayerNorm + FFN (GELU) }
//   → ln_post + proj1 (GELU) + proj2
//
// f16 input (mel spectrogram chunks) → f32 internal (faster on modern x86) → f16 output.
// Same `run(mel_packed, b_chunks, n_mels, cs, chunk_tokens) → (Vec<f16>, out_dim)` signature
// as the GPU encoder, so the dispatch in `inference::align_waveform_text_cpu` can
// mirror the CUDA path's data flow.

// ─── Conv stem ──────────────────────────────────────────────────────

struct CpuConv2d {
    weight: CpuWeight,   // [c_out, c_in, kh, kw]
    bias: Option<Vec<f32>>,
    in_channels: usize,
    out_channels: usize,
}

impl CpuConv2d {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str) -> Result<Self> {
        let (w, ws) = load_f32(weights, &format!("{prefix}.weight"))?;
        assert_eq!(ws.len(), 4, "conv weight {prefix}.weight should be 4D [c_out, c_in, kh, kw]");
        let c_out = ws[0];
        let c_in = ws[1];
        let kh = ws[2]; let kw = ws[3];
        assert_eq!(kh, 3, "expected 3x3 conv, got kh={}", kh);
        assert_eq!(kw, 3, "expected 3x3 conv, got kw={}", kw);
        let bias = if weights.contains_key(&format!("{prefix}.bias")) {
            Some(load_vec(weights, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self {
            weight: CpuWeight { data: w, rows: c_out, cols: c_in * 9 },
            bias,
            in_channels: c_in,
            out_channels: c_out,
        })
    }

    /// x: [b, c_in, f, t] NCHW  →  out: [b, c_out, f_out, t_out]
    /// stride=2, padding=1, kernel=3, with GELU.
    fn forward_gelu(&self, x: &CpuTensor) -> CpuTensor {
        let s = x.shape();
        let b = s[0];
        let c_in = s[1];
        let f = s[2];
        let t = s[3];
        let c_out = self.out_channels;
        // Conv2d with stride=2, pad=1, kernel=3 (PyTorch convention):
        //   out_len = floor((in_len + 2*1 - 3) / 2) + 1 = floor((in_len - 1) / 2) + 1
        let f_out = (f - 1) / 2 + 1;
        let t_out = (t - 1) / 2 + 1;
        // Im2col: for each (b, f_out, t_out), gather a 3x3 patch of all c_in channels
        // → row of length c_in*9.  Total rows = b * f_out * t_out.
        // gemm: y[c_out, c_in*9] @ X[c_in*9, b*f_out*t_out] = out[c_out, b*f_out*t_out]
        // then add bias, GELU, reshape to [b, c_out, f_out, t_out].
        let cols_per_row = c_in * 9;
        let n_rows = b * f_out * t_out;
        let mut im2col = vec![0.0f32; n_rows * cols_per_row];
        for ib in 0..b {
            for ifo in 0..f_out {
                let f0 = ifo * 2;       // top-left of the receptive window in input
                for ito in 0..t_out {
                    let t0 = ito * 2;
                    let row = (ib * f_out + ifo) * t_out + ito;
                    let mut col = 0;
                    for c in 0..c_in {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let f_in = f0 + kh - 1;   // pad=1
                                let t_in = t0 + kw - 1;
                                let v = if f_in < f && t_in < t {
                                    x.data[((ib * c_in + c) * f + f_in) * t + t_in]
                                } else {
                                    0.0
                                };
                                im2col[row * cols_per_row + col] = v;
                                col += 1;
                            }
                        }
                    }
                }
            }
        }
        // gemm: W is [c_out, c_in*9] (row-major).  Compute y = W @ X → [c_out, n_rows].
        // Use gemm crate with non-parallel (im2col is already parallelised via outer loop,
        // and the matmul dims are small enough that single-threaded gemm is fine here).
        // gemm API: gemm(m, n, k, c_ptr, c_cs, c_rs, transpose_c, a_ptr, a_cs, a_rs,
        //                              b_ptr, b_cs, b_rs, beta, alpha, ...).
        // For row-major [m, k] matrix, cs (column stride) = 1, rs (row stride) = k.
        let mut out = vec![0.0f32; c_out * n_rows];
        unsafe {
            gemm(
                c_out, n_rows, cols_per_row,
                out.as_mut_ptr(),
                1, n_rows as isize,            // dst [c_out, n_rows]: cs=1, rs=n_rows
                false,
                self.weight.data.as_ptr(),
                1, cols_per_row as isize,      // lhs [c_out, c_in*9]: cs=1, rs=c_in*9
                im2col.as_ptr(),
                cols_per_row as isize, 1,      // rhs = im2col^T [c_in*9, n_rows]: cs=cols_per_row, rs=1
                0.0, 1.0,
                false, false, false,
                Parallelism::None,
            );
        }
        // Add bias and GELU(tanh approx).  After gemm, out[i, j] = sum_k W[i,k] * X[k,j].
        // Layout is [c_out, n_rows], so out[(i * n_rows) + j] is the (i, j) element.
        // We want to reshape to [b, c_out, f_out, t_out], so dst[ib, c, fo, to] = out[c, row]
        // where row = (ib * f_out + fo) * t_out + to.
        let mut result = vec![0.0f32; b * c_out * f_out * t_out];
        for ib in 0..b {
            for c in 0..c_out {
                for fo in 0..f_out {
                    for to in 0..t_out {
                        let row = (ib * f_out + fo) * t_out + to;
                        let v = out[c * n_rows + row] + self.bias.as_ref().unwrap()[c];
                        // GELU (exact erf, matching GPU kernel's erff)
                        result[((ib * c_out + c) * f_out + fo) * t_out + to] =
                            0.5 * v * (1.0 + libm::erff(v * std::f32::consts::FRAC_1_SQRT_2));
                    }
                }
            }
        }
        CpuTensor::new(result, vec![b, c_out, f_out, t_out])
    }
}

struct CpuLinear {
    weight: CpuWeight,   // [out, in]
    bias: Option<Vec<f32>>,
}

impl CpuLinear {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str) -> Result<Self> {
        let (w, ws) = load_f32(weights, &format!("{prefix}.weight"))?;
        assert_eq!(ws.len(), 2, "linear {prefix}.weight should be 2D");
        let bias = if weights.contains_key(&format!("{prefix}.bias")) {
            Some(load_vec(weights, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Self { weight: CpuWeight { data: w, rows: ws[0], cols: ws[1] }, bias })
    }

    /// x: [..., in_dim]  →  out: [..., out_dim]
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let mut y = linear(x, &self.weight);
        // Bias add — broadcast bias over all leading dims.  Some linears
        // (e.g. conv_out in the audio tower) have no bias.
        if let Some(bias) = &self.bias {
            let nd = x.shape.len();
            let outer: usize = x.shape[..nd - 1].iter().product();
            let out_dim = y.shape[nd - 1];
            for i in 0..outer {
                for j in 0..out_dim {
                    y.data[i * out_dim + j] += bias[j];
                }
            }
        }
        y
    }
}

struct CpuLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl CpuLayerNorm {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: load_vec(weights, &format!("{prefix}.weight"))?,
            bias: load_vec(weights, &format!("{prefix}.bias"))?,
            eps,
        })
    }
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}

struct CpuConvStem {
    conv1: CpuConv2d,
    conv2: CpuConv2d,
    conv3: CpuConv2d,
    conv_out: CpuLinear,
    pe: Vec<f32>,    // [max_pos, d_model]
    d_model: usize,
    max_pos: usize,
    n_mels_out: usize,
    c3_out: usize,
}

impl CpuConvStem {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, config: &AudioConfig) -> Result<Self> {
        let conv1 = CpuConv2d::load(weights, &format!("{prefix}.conv2d1"))?;
        let conv2 = CpuConv2d::load(weights, &format!("{prefix}.conv2d2"))?;
        let conv3 = CpuConv2d::load(weights, &format!("{prefix}.conv2d3"))?;
        let conv_out = CpuLinear::load(weights, &format!("{prefix}.conv_out"))?;
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        // Conv stem downsamples 3x by stride=2 → n_mels_out = f(f(f(n_mels)))
        let f = |l: usize| -> usize { l / 2 };
        let n_mels_out = f(f(f(config.num_mel_bins)));
        let c3_out = conv3.out_channels;
        // Sinusoidal PE (matches asr-burn's CPU encoder, identical math to the CUDA path).
        let half = dm / 2;
        let lt = (10000.0f64).ln() / (half as f64 - 1.0).max(1.0);
        let mut pe_f32 = vec![0.0f32; max_pos * dm];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 * (-(i as f64) * lt).exp();
                pe_f32[p * dm + i] = a.sin() as f32;
                pe_f32[p * dm + half + i] = a.cos() as f32;
            }
        }
        Ok(Self {
            conv1, conv2, conv3, conv_out, pe: pe_f32,
            d_model: dm, max_pos, n_mels_out, c3_out,
        })
    }

    /// Run conv stem on chunked mel input [b_chunks, 1, n_mels, cs].
    /// Returns (output, t2) where output is [b_chunks, t2, d_model] (with PE added).
    fn forward(&self, mel_chunks: &[f16], b_chunks: usize, n_mels: usize, cs: usize) -> Result<(CpuTensor, usize)> {
        // mel_packed: [b_chunks, 1, n_mels, cs] in NCHW.  Convert to f32.
        let x_data: Vec<f32> = mel_chunks.iter().map(|v| v.to_f32()).collect();
        let x = CpuTensor::new(x_data, vec![b_chunks, 1, n_mels, cs]);

        let x = self.conv1.forward_gelu(&x);
        let x = self.conv2.forward_gelu(&x);
        let x = self.conv3.forward_gelu(&x);
        // x: [b_chunks, c3_out, f2, t2].  Permute → [b_chunks, t2, c3_out*f2] for the linear.
        let xs = x.shape();
        let (b, c, fo, t2) = (xs[0], xs[1], xs[2], xs[3]);
        assert_eq!(fo, self.n_mels_out, "conv stem spatial out mismatch: got {}, expected {}", fo, self.n_mels_out);
        // Pack [b, t2, c*f2] in row-major.
        let mut perm = vec![0.0f32; b * t2 * c * fo];
        for ib in 0..b {
            for it in 0..t2 {
                for ic in 0..c {
                    for f in 0..fo {
                        // src is [b, c, f, t] = ((ib * c + ic) * f + f) * t2 + it
                        let src = ((ib * c + ic) * fo + f) * t2 + it;
                        let dst = ((ib * t2 + it) * c + ic) * fo + f;
                        perm[dst] = x.data[src];
                    }
                }
            }
        }
        // conv_out linear: W is [d_model, c*f2].  out = perm @ W^T.
        // The conv_out weight's second dim is c3 * n_mels_out (per the standard fixed
        // size), so we can build a CpuWeight once and reuse.
        let in_dim = c * fo;
        // Per-chunk we need a [t2, in_dim] @ [in_dim, d_model] matmul.  Reshape
        // perm as [b*t2, in_dim] and use our linear() which is GEMV-fast for m=1.
        // Actually [b*t2, in_dim] is multi-row, so it's the gemm path.  Group the
        // chunks together for one big gemm.
        let perm_2d = CpuTensor::new(perm, vec![b * t2, in_dim]);
        // Conv_out weight: in load() we have [d_model, c*f2].  For each chunk the
        // *actual* in_dim is c * fo (== c * n_mels_out).  This is constant across
        // chunks because the conv stem always produces the same f_out = n_mels_out
        // for both full and tail chunks (the t-dim shrinks, not the f-dim).
        // So the same weight applies for every chunk.  Good — we built it once in load.
        let co = self.conv_out.forward(&perm_2d);
        // co: [b*t2, d_model].  Add PE row it (broadcast over b).
        // Match candle's f16_add: quantise both operands and the result through f16.
        let mut out = co.data.clone();
        for ib in 0..b {
            for it in 0..t2 {
                let base = (ib * t2 + it) * self.d_model;
                let pe_base = it * self.d_model;
                for j in 0..self.d_model {
                    let a = f16::from_f32(out[base + j]).to_f32();
                    let b = f16::from_f32(self.pe[pe_base + j]).to_f32();
                    out[base + j] = f16::from_f32(a + b).to_f32();
                }
            }
        }
        Ok((CpuTensor::new(out, vec![b, t2, self.d_model]), t2))
    }
}

// ─── Audio attention (full — aligner doesn't use windowed) ─────────

struct CpuAudioAttention {
    q_proj: CpuLinear,
    k_proj: CpuLinear,
    v_proj: CpuLinear,
    out_proj: CpuLinear,
    num_heads: usize,
    head_dim: usize,
}

impl CpuAudioAttention {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, num_heads: usize, d_model: usize) -> Result<Self> {
        Ok(Self {
            q_proj: CpuLinear::load(weights, &format!("{prefix}.q_proj"))?,
            k_proj: CpuLinear::load(weights, &format!("{prefix}.k_proj"))?,
            v_proj: CpuLinear::load(weights, &format!("{prefix}.v_proj"))?,
            out_proj: CpuLinear::load(weights, &format!("{prefix}.out_proj"))?,
            num_heads,
            head_dim: d_model / num_heads,
        })
    }

    /// x: [1, s, d_model]; full causal self-attention (aligner).
    /// Returns [1, s, d_model].
    fn forward(&self, x: &CpuTensor) -> Result<CpuTensor> {
        let s = x.shape[1];
        let dm = x.shape[2];
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // Reshape [1, s, dm] → [1, s, nh, hd] and transpose to [1, nh, s, hd].
        let q = swap_dims_12(&q.reshape(vec![1, s, nh, hd]));
        let k = swap_dims_12(&k.reshape(vec![1, s, nh, hd]));
        let v = swap_dims_12(&v.reshape(vec![1, s, nh, hd]));

        let scale = 1.0f32 / (hd as f32).sqrt();
        let scores = matmul_qk(&q, &k);  // [1, nh, s, s]
        let attn = softmax_scaled(&scores, scale, false);  // full (non-causal) attention for audio encoder
        let attn_out = matmul_av(&attn, &v);  // [1, nh, s, hd]
        let attn_flat = swap_dims_12(&attn_out).reshape(vec![1, s, dm]);
        Ok(self.out_proj.forward(&attn_flat))
    }
}

/// Scaled softmax over the last two dims of a 4D [b, nh, s, t] tensor.
/// If causal=true, row i masks out columns > i.
/// out[i, j] = exp(in[i, j]*scale - max) / sum;  0 for j > i (if causal).
fn softmax_scaled(scores: &CpuTensor, scale: f32, causal: bool) -> CpuTensor {
    let s = scores.shape();
    let (b, nh, sl, t) = (s[0], s[1], s[2], s[3]);
    let mut out = vec![0.0f32; b * nh * sl * t];
    out.par_chunks_mut(t).enumerate().for_each(|(idx, slab)| {
        let i = idx % sl;
        let valid_t = if causal { i + 1 } else { t };
        let row = &scores.data[idx * t..(idx + 1) * t];
        let mut max_v = f32::NEG_INFINITY;
        for j in 0..valid_t {
            let v = row[j] * scale;
            if v > max_v { max_v = v; }
        }
        let mut sum = 0.0f32;
        for j in 0..t {
            let v = if j < valid_t {
                ((row[j] * scale) - max_v).exp()
            } else { 0.0 };
            slab[j] = v;
            sum += v;
        }
        let inv = 1.0 / sum;
        for j in 0..t { slab[j] *= inv; }
    });
    CpuTensor::new(out, vec![b, nh, sl, t])
}

// ─── Audio FFN (GELU) ──────────────────────────────────────────────

struct CpuAudioFfn {
    fc1: CpuLinear,
    fc2: CpuLinear,
}

impl CpuAudioFfn {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: CpuLinear::load(weights, &format!("{prefix}.fc1"))?,
            fc2: CpuLinear::load(weights, &format!("{prefix}.fc2"))?,
        })
    }
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let mut h = self.fc1.forward(x);
        // In-place GELU (exact erf, matching GPU kernel's erff).
        h.data.par_iter_mut().for_each(|v| {
            let x = *v;
            *v = 0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
        });
        self.fc2.forward(&h)
    }
}

// ─── Audio layer ──────────────────────────────────────────────────

struct CpuAudioLayer {
    sln: CpuLayerNorm,
    attn: CpuAudioAttention,
    fln: CpuLayerNorm,
    ffn: CpuAudioFfn,
}

impl CpuAudioLayer {
    fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, num_heads: usize, d_model: usize) -> Result<Self> {
        Ok(Self {
            sln: CpuLayerNorm::load(weights, &format!("{prefix}.self_attn_layer_norm"), 1e-5)?,
            attn: CpuAudioAttention::load(weights, &format!("{prefix}.self_attn"), num_heads, d_model)?,
            fln: CpuLayerNorm::load(weights, &format!("{prefix}.final_layer_norm"), 1e-5)?,
            ffn: CpuAudioFfn::load(weights, &format!("{prefix}"))?,
        })
    }
    fn forward(&self, x: CpuTensor) -> Result<CpuTensor> {
        // x: [1, s, d_model]
        // Block 1: LN → attn → residual
        let normed = self.sln.forward(&x);
        let attn_out = self.attn.forward(&normed)?;
        let mut h = add(&x, &attn_out);
        // Block 2: LN → FFN → residual
        let normed2 = self.fln.forward(&h);
        let ffn_out = self.ffn.forward(&normed2);
        add_inplace(&mut h, &ffn_out);
        Ok(h)
    }
}

pub(crate) struct CpuAudioEncoder {
    conv_stem: CpuConvStem,
    layers: Vec<CpuAudioLayer>,
    ln_post: CpuLayerNorm,
    proj1: CpuLinear,
    proj2: CpuLinear,
    d_model: usize,
    output_dim: usize,
}

impl CpuAudioEncoder {
    pub fn load(weights: &HashMap<String, WeightTensor>, prefix: &str, config: &AudioConfig) -> Result<Self> {
        let dm = config.d_model;
        let nh = config.encoder_attention_heads;
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(CpuAudioLayer::load(weights, &format!("{prefix}.layers.{}", i), nh, dm)?);
        }
        let ln_post = CpuLayerNorm::load(weights, &format!("{prefix}.ln_post"), 1e-5)?;
        let proj1 = CpuLinear::load(weights, &format!("{prefix}.proj1"))?;
        let proj2 = CpuLinear::load(weights, &format!("{prefix}.proj2"))?;
        let conv_stem = CpuConvStem::load(weights, prefix, config)?;
        let output_dim = config.output_dim;
        Ok(Self { conv_stem, layers, ln_post, proj1, proj2, d_model: dm, output_dim })
    }

    /// Run the full audio encoder on chunked mel input.  Same signature as
    /// `GpuAudioEncoder::run` so the dispatch in `inference::align_waveform_text_cpu`
    /// can mirror the CUDA path's data flow exactly.
    /// `mel_packed`: flat f16 [b_chunks * 1 * n_mels * cs], NCHW.
    /// `chunk_tokens[i]`: how many valid tokens chunk i contributes (≤ t2).
    /// Returns: (f16 [n_total, d_model_proj], output_dim) where d_model_proj is
    /// the final projection's output dimension.
    pub fn run(&self, mel_packed: &[f16], b_chunks: usize, n_mels: usize, cs: usize,
               chunk_tokens: &[usize]) -> Result<(Vec<f16>, usize)> {
        // 1. Conv stem → [b_chunks, t2, d_model] with PE
        let (stem_out, t2) = self.conv_stem.forward(mel_packed, b_chunks, n_mels, cs)?;
        let n_total: usize = chunk_tokens.iter().sum();

        // 2. Pack valid tokens across chunks into a single [1, n_total, d_model] tensor.
        // Each chunk's first `chunk_tokens[i]` rows of its t2-row block are valid; the rest
        // are discarded (the tail chunk can have v < t2 since the input window was
        // smaller than cs).
        let mut packed = Vec::with_capacity(n_total * self.d_model);
        for (idx, &v) in chunk_tokens.iter().enumerate() {
            let base = idx * t2 * self.d_model;
            packed.extend_from_slice(&stem_out.data[base..base + v * self.d_model]);
        }
        let mut h = CpuTensor::new(packed, vec![1, n_total, self.d_model]);

        // 3. 24 × transformer layers (aligner: full attention, no windowing)
        for layer in self.layers.iter() {
            h = layer.forward(h)?;
        }

        // 4. ln_post → proj1 (GELU) → proj2
        let h = self.ln_post.forward(&h);
        let mut h = self.proj1.forward(&h);
        // In-place GELU (exact erf, matching GPU kernel's erff).
        h.data.par_iter_mut().for_each(|v| {
            let x = *v;
            *v = 0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
        });
        let h = self.proj2.forward(&h);

        // Cast f32 → f16 for the output.
        let out: Vec<f16> = h.data.iter().map(|&v| f16::from_f32(v)).collect();
        Ok((out, self.output_dim))
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

    /// Embed ids into [n, hidden] CpuTensor (f32).
    pub fn embed_ids(&self, ids: &[i64]) -> CpuTensor {
        embed_lookup(&self.embed_table, ids)
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
    // Match candle: f32 inv_freq + f16 round-trip on cos/sin.
    let inv: Vec<f32> = (0..hh)
        .map(|i| (rt as f32).powf(-(2.0 * i as f32) / hd as f32))
        .collect();
    let dm = if il { build_interleaved_dim_map(ms, hh) } else { build_contiguous_dim_map(ms, hh) };
    let mut cv = vec![0.0f32; sl * hd];
    let mut sv = vec![0.0f32; sl * hd];
    for t in 0..sl {
        for j in 0..hh {
            let a = pos[dm[j]][t] as f32 * inv[j];
            let c = f16::from_f32(a.cos()).to_f32();
            let s = f16::from_f32(a.sin()).to_f32();
            cv[t * hd + j] = c;
            sv[t * hd + j] = s;
            cv[t * hd + j + hh] = c;
            sv[t * hd + j + hh] = s;
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
