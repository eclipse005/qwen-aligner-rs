//! CPU-resident engine for the Qwen3-ForcedAligner (text decoder + audio encoder).
//!
//! Mirror of `cudarc_engine.rs` / `gpu_audio_encoder.rs` running on the host CPU:
//!   * `gemm` crate handles every matmul, with `Parallelism::Rayon(0)` forced
//!     even for the m=1 lm_head GEMV (gemm's internal threshold would
//!     otherwise leave decode single-threaded).
//!   * `rayon` parallelises every hand-written elementwise / reduction op
//!     (rms_norm, silu_mul, prefill attention) across heads or rows.
//!   * Tensors are f32 (Vec<f32>) — Arrow Lake / Zen-class x86 has no native
//!     f16 SIMD outside Sapphire Rapids, so f32 ends up faster than f16 with
//!     upcast. All weights are pre-converted f16→f32 once at load time.
//!   * KV cache is pre-allocated per layer.
//!   * The two hot attention inner loops (Q@K dot and A@V axpy) are written as
//!     AVX2+FMA intrinsics (`dot_qk_avx2` / `axpy_avx2`), and the conv stem
//!     uses a 1-row × 8-col NHWC direct-conv microkernel (`conv_nhwc_direct`).
//!
//! Both the 28-layer text decoder and the 24-layer audio encoder (conv stem +
//! transformer) are fully implemented here; `DeviceRequest::Cpu` runs the
//! whole pipeline end-to-end with no CUDA dependency.

use anyhow::Result;
use gemm::{gemm, Parallelism};
use half::f16;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::config::{AudioConfig, TextConfig};
use crate::raw_tensor::RawTensor;
// ─── Sub-profile helpers (mirrors CUDA's QFA_SUB_PROFILE) ─────────────────────
// QFA_SUB_PROFILE=1: print per-op time per layer for text decoder and
// per-section time for audio encoder.  Zero overhead when env var is unset
// (Option<Instant> branch is inlined and skipped).

static SUB_PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn sub_profile_enabled() -> bool {
    *SUB_PROFILE.get_or_init(|| std::env::var_os("QFA_SUB_PROFILE").is_some())
}

#[inline(always)]
fn sub_t0() -> Option<std::time::Instant> {
    if sub_profile_enabled() { Some(std::time::Instant::now()) } else { None }
}

#[inline(always)]
fn sub_ms(t0: Option<std::time::Instant>) -> f64 {
    match t0 { Some(t) => t.elapsed().as_secs_f64() * 1000.0, None => 0.0 }
}

// FlashAttention-style tiled prefill_attention is the DEFAULT (online softmax,
// no 110 MB scores scratch, ~30% attn/layer reduction).  Accuracy gate is
// timestamp-level, not bit-exact: the online softmax rescale changes the f32
// sum-reduction order by ULPs, which *could* flip argmax at near-tie logits.
// Validated against the 15s EN smoke (40/40) and the 180s golden (909/909)
// timestamp equality — no flips observed.  Set QFA_NO_FLASH=1 to fall back to
// the materialised-softmax path (kept for debugging / regression A/B).
static FLASH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
fn flash_enabled() -> bool {
    *FLASH.get_or_init(|| std::env::var_os("QFA_NO_FLASH").is_none())
}


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

/// Weight matrix stored as f16 — halved memory vs f32.
/// Read directly by m=1 GEMV (halved bandwidth for memory-bound decode).
/// Converted to f32 on-the-fly for m>1 GEMM (prefill).
pub(crate) struct CpuWeightF16 {
    pub data: Vec<f16>,
    pub rows: usize,
    pub cols: usize,
}

impl CpuWeightF16 {
    /// Consuming f16→f32 with rayon. Used at load time to hoist conversion out
    /// of the hot loop. The 0.6B model costs ~1.2 GB extra RAM, but eliminates
    /// repeated per-call conversions (text decoder: ~250 calls per forward).
    pub(crate) fn into_f32(self) -> CpuWeight {
        let data: Vec<f32> = self.data.into_par_iter().map(|v| v.to_f32()).collect();
        CpuWeight { data, rows: self.rows, cols: self.cols }
    }
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
    let out = if m == 1 {
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

/// Q @ K^T for [b, nh, s, hd] layout, returns [b, nh, s, s] (head-contig bytes).
/// Per (b, nh) head, inner loop is `dot_qk_avx2` (8-wide FMA over the hd dim).
/// Used by the audio encoder (s ≈ 80, hd = 64 → 8 microkernel iters per dot).
pub(crate) fn matmul_qk(q: &CpuTensor, k: &CpuTensor) -> CpuTensor {
    let qs = q.shape();
    let (b, nh, s, hd) = (qs[0], qs[1], qs[2], qs[3]);
    let mut out = vec![0.0f32; b * nh * s * s];
    out.par_chunks_mut(s * s).enumerate().for_each(|(idx, slab)| {
        let ib = idx / nh;
        let ih = idx % nh;
        let q_off = (ib * nh + ih) * s * hd;
        let k_off = (ib * nh + ih) * s * hd;
        for i in 0..s {
            let q_row = &q.data[q_off + i * hd..q_off + (i + 1) * hd];
            for t in 0..s {
                let k_row = &k.data[k_off + t * hd..k_off + (t + 1) * hd];
                slab[i * s + t] = unsafe { dot_qk_avx2(q_row, k_row) };
            }
        }
    });
    CpuTensor::new(out, vec![b, nh, s, s])
}

/// Attention @ V: [b, nh, s, t] × [b, nh, t, hd] → [b, nh, s, hd] (head-contig bytes;
/// reshape + swap_dims_12 in the caller converts to [b, s, nqh*hd] for the next op).
/// Per (b, nh) head, inner loop is `axpy_avx2` (8-wide FMA: out[i, :] += w * V[t, :])
/// over t with one hd-wide FMA per t — for hd=64 that's 8 microkernel iters per axpy.
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
        for i in 0..s {
            let attn_row = &attn.data[a_off + i * t..a_off + (i + 1) * t];
            let out_i = &mut slab[i * hd..(i + 1) * hd];
            for k in 0..t {
                let w = attn_row[k];
                if w == 0.0 { continue; }
                let v_row = &v.data[v_off + k * hd..v_off + (k + 1) * hd];
                unsafe { axpy_avx2(out_i, w, v_row); }
            }
        }
    });
    CpuTensor::new(out, vec![b, nh, s, hd])
}
//  Rotary embedding (in-place on a head row)
// ═══════════════════════════════════════════════════════════════════════

#[inline]
fn apply_rotary_row(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let half = d / 2;
    // Rotary is a pairwise rotation of (x[i], x[i+half]) using the *original*
    // values of both.  Previously this hoisted a vec![d] copy per call; with
    // (nqh+nkvh)=24 heads × b*s=5240 tokens × 28 layers = 3.5M allocs/forward
    // that was pure waste.  Capture both into locals first, then write back —
    // the fmul/fsub/fadd are the identical operations on the identical inputs
    // in the identical order, so bit-exact.  No allocation.
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i + half] + a * sin[i + half];
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

// Thread-local scratch buffers for `prefill_attention`.  The per-head work
// previously allocated `q_qh` (s*hd) and `scores` (s*cur_len) on every one of
// the 28 layers × nqh heads = 448 heads/forward.  With rayon's work-stealing,
// each head runs on some worker thread; parking the scratch behind a
// thread-local (Cell<Vec<f32>>) means each worker reuses one growing buffer
// across all heads it processes, instead of alloc/free per head.  We take()
// the Vec out of the cell, use it as an owned value for this head, then put it
// back — so the body has a normal `&mut Vec<f32>` with no aliasing at all.
thread_local! {
    static PREFILL_Q: std::cell::Cell<Vec<f32>> = const { std::cell::Cell::new(Vec::new()) };
    static PREFILL_SCORES: std::cell::Cell<Vec<f32>> = const { std::cell::Cell::new(Vec::new()) };
}

/// Take the thread-local scratch Vec (or empty if first use on this thread),
/// leaving an empty Vec in its place.  Caller MUST `set` it back when done.
fn take_scratch(cell: &std::cell::Cell<Vec<f32>>) -> Vec<f32> {
    cell.take()
}

fn prefill_attention(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    causal: bool,
) -> CpuTensor {
    if flash_enabled() {
        return prefill_attention_flash(
            q, k_cache, v_cache,
            b, nqh, nkvh, max_seq, hd, cur_len, causal,
        );
    }
    let s = q.shape[2];
    let n_rep = nqh / nkvh;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let out = vec![0.0f32; b * s * nqh * hd];
    // kv_start = number of cached positions before this prefill chunk.
    // For the aligner's single-prefill use (kv_start=0) cur_len == s.
    let kv_start = cur_len - s;

    // ───────────────────────────────────────────────────────────────────
    //  Causal work-skip rewrite.
    //
    //  P1-2 (v2): parallelise over (b * nqh) query heads (16-way on this
    //  model — good width for a 20-core box; a prior attempt parallelised
    //  over (b * nkvh = 8) KV heads to share K/V between GQA siblings, but
    //  that *halved* the parallel width and regressed ~13% on a many-core
    //  host.  Per-head K reads are already LLC-resident (one KV head = 2.7 MB
    //  < ~36 MB LLC), so the second sibling's K read is an LLC hit anyway —
    //  the GQA-share theory didn't pay off.  Keep 16-way.
    //
    //  The actual win here is causal work-skip: query row i only attends to
    //  K/V[0..=i+kv_start].  The masked tail never enters the inner loop.
    //  Each scores row is pre-filled with NEG_INFINITY (byte-identical to the
    //  old code that wrote -inf into each masked slot), so softmax max/sum/exp
    //  see the exact same row bytes → bit-exact.  A@V likewise iterates only
    //  the live tail instead of scanning all cur_len with `if w==0 continue`.
    //
    //  BIT-EXACT CONTRACT: softmax reduction order (single-pass max, then
    //  left-to-right sum, then normalize) is preserved verbatim.  Online /
    //  block-wise softmax was rejected: it changes the sum reduction order,
    //  and the CPU path has no f16 round-trip to absorb the resulting ULPs
    //  before argmax (inference::argmax_rows, 1/256 tie band).  Each
    //  dot_qk_avx2(qi, kt) is deterministic in its inputs, and every
    //  scores[i][t] for t < limit is computed with the identical (qi, kt)
    //  pair as the old per-head code → every softmax input is bit-identical.
    // ───────────────────────────────────────────────────────────────────
    (0..b * nqh).into_par_iter().for_each(|idx| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let kv_base = (ib * nkvh + kh) * max_seq * hd;

        // Borrow the thread-local scratch for this head: resize (no realloc if
        // it already fits from a previous head on this worker), gather Q rows,
        // run Q@K + softmax + A@V, then return the Vec to the thread-local.
        let mut q_qh = PREFILL_Q.with(take_scratch);
        q_qh.resize(s * hd, 0.0);
        for i in 0..s {
            let src = ((ib * s + i) * nqh + qh) * hd;
            q_qh[i * hd..(i + 1) * hd].copy_from_slice(&q.data[src..src + hd]);
        }

        // Q @ K^T: per-head [s, hd] @ [hd, s] -> [s, s].  Inner dot product is
        // 8-wide FMA.  i-outer: each Q row reads its live K rows; Q[i] stays
        // L1-hot across the t loop, and scores writes are contiguous per row.
        // (A t-outer variant was tested to share K[t] reads across Q rows but
        // regressed ~17%: strided scores writes at stride=cur_len thrash the
        // cache worse than the K-reuse saves.  Same lesson as the rejected
        // 4-row microkernel.  Keep i-outer.)
        // Causal work-skip: only [0..limit) per row is written; the masked tail
        // [limit..cur_len] is left at 0.0 (resize) and never read again —
        // softmax and A@V below both scan only [0..limit).  Bit-exact: the
        // tail's value never influenced the result (see softmax note below).
        let mut scores = PREFILL_SCORES.with(take_scratch);
        scores.resize(s * cur_len, 0.0);
        for i in 0..s {
            let qi = &q_qh[i * hd..(i + 1) * hd];
            let limit = if causal { i + kv_start + 1 } else { cur_len };
            let row = &mut scores[i * cur_len..i * cur_len + limit];
            for t in 0..limit {
                let kt = &k_cache[kv_base + t * hd..kv_base + (t + 1) * hd];
                let dot = unsafe { dot_qk_avx2(qi, kt) };
                row[t] = dot * scale;
            }
        }

        // Softmax (single-pass max, left-to-right sum, normalize — order
        // preserved verbatim for bit-exactness).  Causal: the live region of
        // row i is [0..limit=i+kv_start+1]; the masked tail [limit..cur_len]
        // was never written (still 0.0 from resize).  Scanning only [0..limit)
        // is bit-exact because the tail never affected the result anyway:
        //   max pass: -inf/0.0 < any real score (rows always have ≥1 live t).
        //   exp/sum: a 0.0 tail element would give exp(0-mx) (nonzero!) which
        //   WOULD corrupt the sum — so we must NOT leave 0.0 in the tail when
        //   softmax could touch it.  Since we now scan only [0..limit), the
        //   tail is never read, so its value is irrelevant.  But the OLD code
        //   wrote -inf to the tail and scanned the full row; its exp(-inf-mx)=0
        //   contributed nothing.  Restricting softmax to [0..limit) produces
        //   identical mx, identical exp(x-mx) for live x, identical sum, and
        //   identical normalized weights on the live region.  Tail bytes are
        //   never consumed (A@V below also scans only [0..limit)).  Bit-exact.
        for i in 0..s {
            let limit = if causal { i + kv_start + 1 } else { cur_len };
            let row = &mut scores[i * cur_len..i * cur_len + limit];
            let mut mx = f32::NEG_INFINITY;
            for &v in row.iter() { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }

        // Scores @ V: 1-row microkernel.  The 4-row version saves V cache
        // reads but the 4 OUT[i,:] rows are stride-hd apart (each at its own
        // head's [b,s,nqh,hd] slot) and the L1 prefetcher handles the single
        // sequential row better.  Net: 1-row is faster.  Causal: only
        // [0..limit) can be non-zero (the rest were -inf → softmax weight 0),
        // so iterate the live tail only.
        let out_ptr = out.as_ptr() as *mut f32;
        for i in 0..s {
            let dst_off = ((ib * s + i) * nqh + qh) * hd;
            let limit = if causal { i + kv_start + 1 } else { cur_len };
            let scan = if causal { limit } else { cur_len };
            unsafe {
                let out_i = std::slice::from_raw_parts_mut(out_ptr.add(dst_off), hd);
                for j in 0..hd { out_i[j] = 0.0; }
                let row = &scores[i * cur_len..(i + 1) * cur_len];
                for t in 0..scan {
                    let w = row[t];
                    if w == 0.0 { continue; }
                    let vt = &v_cache[kv_base + t * hd..kv_base + (t + 1) * hd];
                    axpy_avx2(out_i, w, vt);
                }
            }
        }

        // Return the scratch buffers to their thread-locals for reuse by the
        // next head on this worker.  `shrink_to_fit` would defeat the reuse,
        // so we deliberately leave the capacity pinned high.
        PREFILL_Q.with(|c| c.set(q_qh));
        PREFILL_SCORES.with(|c| c.set(scores));
    });

    CpuTensor::new(out, vec![b, nqh, s, hd])
}

// ═══════════════════════════════════════════════════════════════════════
//  FlashAttention-style tiled prefill attention.
//
//  Replaces the materialised [s, cur_len] scores scratch (110 MB/head on the
//  180s fixture — way past LLC) with a tiled online-softmax that keeps a
//  [Bq, Bk] score tile in L1 and never materialises the full scores matrix.
//  K is streamed once per Q-block-row instead of being re-read once per Q row.
//
//  Algorithm (per query head, per Q-block-row of Bq rows):
//    for each K-block of Bk rows in the live region:
//        S[q, :] = Q[q] · K[k_block]            (Bq × Bk, in L1)
//        m_new   = max(m_old, rowmax(S[q, :]))
//        renorm  = exp(m_old - m_new)
//        s       = s * renorm + sum_q(exp(S[q,:] - m_new))
//        O       = O * renorm + exp(S[q,:] - m_new) · V[k_block]
//    O /= s                                       (final normalise)
//
//  `f32::exp` per element is preserved (same exp call as the materialised
//  path on each (S-m_new) value); what changes vs bit-exact is the sum
//  reduction order (per-K-block partial sums + a renormalisation rescale),
//  which introduces f32 ULPs.  Accuracy gate is timestamp-level, not bit-
//  level: see QFA_FLASH note above.  Output layout is identical to the
//  materialised path ([b, s, nqh, hd] token-major) so o_proj is unchanged.
//
//  Block sizes: Bq=32 Q rows per tile (32 × hd × 4 = 16 KB Q tile in L1),
//  Bk=128 K rows per tile (128 × hd × 4 = 64 KB K tile, fits L2; streamed).
//  Per Q-block, a [Bq, Bk] S scratch = 32 × 128 × 4 = 16 KB stays in L1.
//  Per-row online state m[Bq], s[Bq], O[Bq, hd] (32 × (1+1+128) × 4 ≈ 16.5 KB).
//  Both fit comfortably in L1 (32-48 KB).
// ═══════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
fn prefill_attention_flash(
    q: &CpuTensor,
    k_cache: &[f32], v_cache: &[f32],
    b: usize, nqh: usize, nkvh: usize, max_seq: usize, hd: usize, cur_len: usize,
    causal: bool,
) -> CpuTensor {
    const BQ: usize = 32;
    const BK: usize = 128;
    let s = q.shape[2];
    let n_rep = nqh / nkvh;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let out = vec![0.0f32; b * s * nqh * hd];
    let kv_start = cur_len - s;

    (0..b * nqh).into_par_iter().for_each(|idx| {
        let ib = idx / nqh;
        let qh = idx % nqh;
        let kh = qh / n_rep;
        let kv_base = (ib * nkvh + kh) * max_seq * hd;

        // Per-row online softmax state across K-blocks, sized for one Q tile.
        let mut m_state = [f32::NEG_INFINITY; BQ];
        let mut s_state = [0.0f32; BQ];
        // O accumulator: [Bq, hd].  32 × 128 × 4 = 16 KB, stays in L1.
        let mut o_state = vec![0.0f32; BQ * hd];
        // S tile scratch: [Bq, Bk] = 32 × 128 = 4096 f32 = 16 KB, stays in L1.
        let mut s_tile = vec![0.0f32; BQ * BK];
        // Q tile: gather Bq Q rows once per Q-block (contiguous, L1-hot).
        let mut q_tile = [0.0f32; BQ * 128]; // hd always 128 in this model
        debug_assert_eq!(hd, 128);

        // Process Q in blocks of Bq rows.
        for q_start in (0..s).step_by(BQ) {
            let q_end = (q_start + BQ).min(s);
            let bq = q_end - q_start;

            // (Re)initialise per-tile state.
            for r in 0..bq {
                m_state[r] = f32::NEG_INFINITY;
                s_state[r] = 0.0;
                let qsrc = ((ib * s + q_start + r) * nqh + qh) * hd;
                q_tile[r * hd..(r + 1) * hd].copy_from_slice(&q.data[qsrc..qsrc + hd]);
                for j in 0..hd { o_state[r * hd + j] = 0.0; }
            }

            // Causal: row (q_start+r) only attends to K/V[0 .. q_start+r+kv_start+1].
            // So the K-block range is [0 .. max_row_limit) where
            // max_row_limit = q_end-1 + kv_start + 1 = q_end + kv_start.
            let max_k = if causal { q_end + kv_start } else { cur_len };

            // Stream K/V in blocks of Bk.  Each K row read once per Q-block-row.
            for k_start in (0..max_k).step_by(BK) {
                let k_end = (k_start + BK).min(max_k);
                let bk = k_end - k_start;

                // S[r, c] = (Q[q_start+r] · K[k_start+c]) * scale  for r<bq, c<bk.
                for r in 0..bq {
                    let qr = q_start + r;
                    let qi = &q_tile[r * hd..(r + 1) * hd];
                    // Per-Q-row causal limit: K indices t in [0 .. qr+kv_start+1).
                    let row_limit = if causal { qr + kv_start + 1 } else { cur_len };
                    let srow = &mut s_tile[r * BK..r * BK + bk];
                    for c in 0..bk {
                        let t = k_start + c;
                        if causal && t >= row_limit {
                            srow[c] = f32::NEG_INFINITY;
                        } else {
                            let kt = &k_cache[kv_base + t * hd..kv_base + (t + 1) * hd];
                            srow[c] = unsafe { dot_qk_avx2(qi, kt) } * scale;
                        }
                    }
                }

                // Online softmax update per Q row in the tile.
                for r in 0..bq {
                    let srow = &s_tile[r * BK..r * BK + bk];
                    // m_new = max(m_old, max(srow))
                    let mut m_new = m_state[r];
                    for &v in srow.iter() { if v > m_new { m_new = v; } }
                    let renorm = (m_state[r] - m_new).exp(); // exp(0)=1 first block
                    // s = s * renorm + sum(exp(srow - m_new))
                    let mut block_sum = 0.0f32;
                    let mut exp_vals = [0.0f32; BK];
                    for c in 0..bk {
                        exp_vals[c] = (srow[c] - m_new).exp();
                        block_sum += exp_vals[c];
                    }
                    s_state[r] = s_state[r] * renorm + block_sum;
                    // O = O * renorm + sum_c exp_vals[c] * V[k_start+c]
                    let orow = &mut o_state[r * hd..(r + 1) * hd];
                    for j in 0..hd { orow[j] *= renorm; }
                    for c in 0..bk {
                        let w = exp_vals[c];
                        if w == 0.0 { continue; }
                        let t = k_start + c;
                        let vt = &v_cache[kv_base + t * hd..kv_base + (t + 1) * hd];
                        unsafe { axpy_avx2(orow, w, vt); }
                    }
                    m_state[r] = m_new;
                }
            }

            // Final normalise: O /= s, write to output tensor.
            let out_ptr = out.as_ptr() as *mut f32;
            for r in 0..bq {
                let qr = q_start + r;
                let dst_off = ((ib * s + qr) * nqh + qh) * hd;
                let inv = 1.0 / s_state[r];
                unsafe {
                    let out_i = std::slice::from_raw_parts_mut(out_ptr.add(dst_off), hd);
                    let orow = &o_state[r * hd..(r + 1) * hd];
                    for j in 0..hd { out_i[j] = orow[j] * inv; }
                }
            }
        }
    });

    CpuTensor::new(out, vec![b, nqh, s, hd])
}

#[cfg(not(target_arch = "x86_64"))]
fn prefill_attention_flash(
    _q: &CpuTensor, _k_cache: &[f32], _v_cache: &[f32],
    _b: usize, _nqh: usize, _nkvh: usize, _max_seq: usize, _hd: usize, _cur_len: usize,
    _causal: bool,
) -> CpuTensor {
    unimplemented!("prefill_attention_flash requires x86_64 AVX2")
}


/// ensure len is a multiple of 8 (the text decoder's hd=128 always is).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_qk_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len() % 8, 0);
    let mut acc = _mm256_setzero_ps();
    let n = a.len() / 8;
    for j in 0..n {
        let va = _mm256_loadu_ps(a.as_ptr().add(j * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(j * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum of 8 lanes: hsum = (acc[0]+acc[1]+...+acc[7]).
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    _mm_cvtss_f32(s)
}

/// AVX2+FMA: out[i] += w * v[i] for i in 0..out.len().  Caller must ensure
/// len is a multiple of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(out: &mut [f32], w: f32, v: &[f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(out.len(), v.len());
    debug_assert_eq!(out.len() % 8, 0);
    let wv = _mm256_set1_ps(w);
    let n = out.len() / 8;
    for j in 0..n {
        let vo = _mm256_loadu_ps(out.as_ptr().add(j * 8));
        let vv = _mm256_loadu_ps(v.as_ptr().add(j * 8));
        let r = _mm256_fmadd_ps(wv, vv, vo);
        _mm256_storeu_ps(out.as_mut_ptr().add(j * 8), r);
    }
}

/// P4-1: NHWC direct 3x3 conv kernel — 1-row x 8-col microkernel.
/// Computes c_out values for one (b, fo, to) by gathering the 9 input
/// patches from the NHWC tensor and FMA-accumulating against the
/// transposed weight (laid out as [c_in*9, c_out] at load, which is
/// also [c_in, kh*kw, c_out] in row-major).  Returns the full c_out in
/// a stack buffer of fixed size 512 (c_out must be <= 512) using the
/// load+FMA+store pattern.  P4-2 attempt with [[f32;8]; 16] scratch
/// buffers (intended to live in YMM registers) was slower than this
/// version because the [f32; 8] copy_from_slice + loadu_ps overhead
/// exceeds the savings from keeping the accumulator out of L1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_nhwc_direct(
    x_ptr: *const f32,
    w_ptr: *const f32,
    c_in: usize, c_out: usize, n_full: usize,
    b: usize, f: usize, t: usize,
    ifo: usize, ito: usize,
) -> [f32; 512] {
    use std::arch::x86_64::*;
    debug_assert!(c_out <= 512);
    let mut accs = [0.0f32; 512];
    let f0 = ifo * 2;
    let t0 = ito * 2;
    for ic in 0..c_in {
        for kk in 0..9usize {
            let kh = kk / 3;
            let kw = kk % 3;
            let f_in_signed = f0 as i32 + kh as i32 - 1;
            let t_in_signed = t0 as i32 + kw as i32 - 1;
            let x_val = if f_in_signed >= 0 && f_in_signed < f as i32
                && t_in_signed >= 0 && t_in_signed < t as i32
            {
                *x_ptr.add((b * f + f_in_signed as usize) * t * c_in
                    + t_in_signed as usize * c_in + ic)
            } else {
                0.0
            };
            for c_out_base in (0..n_full).step_by(8) {
                let w_vec = _mm256_loadu_ps(w_ptr.add((ic * 9 + kk) * c_out + c_out_base));
                let acc_vec = _mm256_loadu_ps(accs.as_ptr().add(c_out_base));
                let new_acc = _mm256_fmadd_ps(w_vec, _mm256_set1_ps(x_val), acc_vec);
                _mm256_storeu_ps(accs.as_mut_ptr().add(c_out_base), new_acc);
            }
            for c_out_idx in n_full..c_out {
                accs[c_out_idx] += x_val * *w_ptr.add((ic * 9 + kk) * c_out + c_out_idx);
            }
        }
    }
    accs
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
    /// Weight stored as f32 (pre-converted at load time).
    weight: CpuWeight,   // [c_out, c_in*9] (im2col-friendly layout)
    bias: Option<Vec<f32>>,
    out_channels: usize,
}

impl CpuConv2d {
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        let (w, ws) = load_f16_vec(weights, &format!("{prefix}.weight"))?;
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
        // Convert f16 → f32 at load time (one-time cost), then transpose to
        // [c_in*9, c_out] layout so the SIMD matmul can read 8 c_out values
        // at a fixed k with one contiguous 8-wide load.  In the natural
        // [c_out, c_in*9] layout those 8 values are strided by c_in*9, and
        // the SIMD load would read 7 wrong k values (silent bit-corruption).
        let weight_f32 = CpuWeightF16 { data: w, rows: c_out, cols: c_in * 9 }.into_f32();
        let cols = c_in * 9;
        let mut w_t = vec![0.0f32; cols * c_out];
        for c_out_idx in 0..c_out {
            for k in 0..cols {
                w_t[k * c_out + c_out_idx] = weight_f32.data[c_out_idx * cols + k];
            }
        }
        let weight = CpuWeight { data: w_t, rows: cols, cols: c_out };
        Ok(Self { weight, bias, out_channels: c_out })
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
        // The NCHW input is transposed to NHWC (~0.2ms for 180s conv2),
        // then the 1-row x 8-col direct conv (conv_nhwc_direct) replaces
        // both the im2col buffer and the gemm-style matmul.  Inner FMA
        // loop is c_in*9 = 270 terms; the 8-wide broadcast of x and
        // 8-wide load of w reads 8 c_out values per (ic, kh*kw) — same
        // memory pattern as P2-1 but with no im2col materialization
        // (saves ~389ms of scalar gather on 180s conv2) and no row-tile
        // cache-pressure issues.  Output is (n_rows, c_out) row-major
        // matching the old matmul_out, so the bias + GELU tail is
        // unchanged.
        let n_rows = b * f_out * t_out;
        let mut matmul_out = vec![0.0f32; n_rows * c_out];
        // NCHW [b, c_in, f, t] → NHWC [b, f, t, c_in].  Parallel over
        // (b, f) chunks of t*c_in elements.
        let mut x_nhwc = vec![0.0f32; b * f * t * c_in];
        x_nhwc.par_chunks_mut(t * c_in).enumerate().for_each(|(bf, x_bf)| {
            let ib = bf / f;
            let if_ = bf % f;
            for it in 0..t {
                for ic in 0..c_in {
                    x_bf[it * c_in + ic] = x.data[((ib * c_in + ic) * f + if_) * t + it];
                }
            }
        });
        let w_t = &self.weight;            // [c_in*9, c_out] row-major (transposed at load)
        let n_full = (c_out / 8) * 8;
        // Pass pointers as usize so the closure satisfies Sync (raw
        // pointers are !Sync, blocking rayon parallelism).
        let w_t_addr = w_t.data.as_ptr() as usize;
        let x_addr = x_nhwc.as_ptr() as usize;
        use rayon::prelude::*;
        matmul_out.par_chunks_mut(c_out).enumerate().for_each(|(row, row_slice)| {
            let ib = row / (f_out * t_out);
            let rem = row % (f_out * t_out);
            let ifo = rem / t_out;
            let ito = rem % t_out;
            let accs = unsafe {
                conv_nhwc_direct(
                    x_addr as *const f32, w_t_addr as *const f32,
                    c_in, c_out, n_full, ib, f, t, ifo, ito,
                )
            };
            let row_out = row_slice.as_mut_ptr();
            for c_out_idx in 0..c_out {
                unsafe { *row_out.add(c_out_idx) = accs[c_out_idx]; }
            }
        });
        // Bias add + GELU (exact erf via libm) into the NCHW output, parallel
        // over (ib, c) chunks of f_out * t_out elements each.  Output shape
        // is [b, c_out, f_out, t_out] (NCHW), so c_slab is one (ib, c) plane.
        let bias = self.bias.as_ref().unwrap();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut result = vec![0.0f32; b * c_out * f_out * t_out];
        result.par_chunks_mut(f_out * t_out).enumerate().for_each(|(ic, c_slab)| {
            let ib = ic / c_out;
            let c = ic % c_out;
            let b_val = bias[c];
            for fo in 0..f_out {
                for to in 0..t_out {
                    let row = (ib * f_out + fo) * t_out + to;
                    let v = matmul_out[row * c_out + c] + b_val;
                    c_slab[fo * t_out + to] =
                        0.5 * v * (1.0 + libm::erff(v * inv_sqrt2));
                }
            }
        });
        CpuTensor::new(result, vec![b, c_out, f_out, t_out])
    }
}

struct CpuLinear {
    /// Weight stored as f32 (pre-converted at load time).
    weight: CpuWeight,   // [out, in]
    bias: Option<Vec<f32>>,
}

impl CpuLinear {
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        let (w, ws) = load_f16_vec(weights, &format!("{prefix}.weight"))?;
        assert_eq!(ws.len(), 2, "linear {prefix}.weight should be 2D");
        let bias = if weights.contains_key(&format!("{prefix}.bias")) {
            Some(load_vec(weights, &format!("{prefix}.bias"))?)
        } else {
            None
        };
        // Convert f16 → f32 at load time.
        let weight = CpuWeightF16 { data: w, rows: ws[0], cols: ws[1] }.into_f32();
        Ok(Self { weight, bias })
    }

    /// x: [..., in_dim]  →  out: [..., out_dim]
    /// Audio encoder: weight is already f32 (pre-converted at load time).
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
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: load_vec(weights, &format!("{prefix}.weight"))?,
            bias: load_vec(weights, &format!("{prefix}.bias"))?,
            eps,
        })
    }
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let t = sub_t0();
        let y = layer_norm(x, &self.weight, &self.bias, self.eps);
        if sub_profile_enabled() {
            eprintln!("    layer_norm ({}) dt={:.1} ms", x.shape[0] * x.shape[1], sub_ms(t));
        }
        y
    }
}

struct CpuConvStem {
    conv1: CpuConv2d,
    conv2: CpuConv2d,
    conv3: CpuConv2d,
    conv_out: CpuLinear,
    pe: Vec<f32>,    // [max_pos, d_model]
    d_model: usize,
    n_mels_out: usize,
}

impl CpuConvStem {
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str, config: &AudioConfig) -> Result<Self> {
        let conv1 = CpuConv2d::load(weights, &format!("{prefix}.conv2d1"))?;
        let conv2 = CpuConv2d::load(weights, &format!("{prefix}.conv2d2"))?;
        let conv3 = CpuConv2d::load(weights, &format!("{prefix}.conv2d3"))?;
        let conv_out = CpuLinear::load(weights, &format!("{prefix}.conv_out"))?;
        let dm = config.d_model;
        let max_pos = config.max_source_positions;
        // Conv stem downsamples 3x by stride=2 → n_mels_out = f(f(f(n_mels)))
        let f = |l: usize| -> usize { l / 2 };
        let n_mels_out = f(f(f(config.num_mel_bins)));
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
            d_model: dm, n_mels_out,
        })
    }

    /// Run conv stem on chunked mel input [b_chunks, 1, n_mels, cs].
    /// Returns (output, t2) where output is [b_chunks, t2, d_model] (with PE added).
    fn forward(&self, mel_chunks: &[f16], b_chunks: usize, n_mels: usize, cs: usize) -> Result<(CpuTensor, usize)> {
        let t = sub_t0();
        // mel_packed: [b_chunks, 1, n_mels, cs] in NCHW.  Convert to f32.
        let x_data: Vec<f32> = mel_chunks.iter().map(|v| v.to_f32()).collect();
        let x = CpuTensor::new(x_data, vec![b_chunks, 1, n_mels, cs]);
        let t0 = sub_t0();
        let x = self.conv1.forward_gelu(&x);
        let dt_conv1 = sub_ms(t0);
        let t0 = sub_t0();
        let x = self.conv2.forward_gelu(&x);
        let dt_conv2 = sub_ms(t0);
        let t0 = sub_t0();
        let x = self.conv3.forward_gelu(&x);
        let dt_conv3 = sub_ms(t0);
        let t0 = sub_t0();
        // x: [b_chunks, c3_out, f2, t2].  Permute → [b_chunks, t2, c3_out*f2] for the linear.
        let xs = x.shape();
        let (b, c, fo, t2) = (xs[0], xs[1], xs[2], xs[3]);
        assert_eq!(fo, self.n_mels_out, "conv stem spatial out mismatch: got {}, expected {}", fo, self.n_mels_out);
        // Pack [b, t2, c*f2] in row-major. Parallel over (ib, it) rows — each
        // job writes a disjoint c*fo row into `perm`, so no cross-job aliasing.
        let mut perm = vec![0.0f32; b * t2 * c * fo];
        perm.par_chunks_mut(c * fo).enumerate().for_each(|(row, row_out)| {
            let ib = row / t2;
            let it = row % t2;
            for ic in 0..c {
                for f in 0..fo {
                    // src is [b, c, f, t] = ((ib * c + ic) * fo + f) * t2 + it
                    let src = ((ib * c + ic) * fo + f) * t2 + it;
                    row_out[ic * fo + f] = x.data[src];
                }
            }
        });
        let dt_perm = sub_ms(t0);
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
        let dt_conv_out = sub_ms(t0);
        // co: [b*t2, d_model].  Add PE row it (broadcast over b).
        // Match candle's f16_add: quantise both operands and the result through f16.
        // Parallel over rows — each (ib, it) row is d_model contiguous and disjoint.
        // The f16::from_f32/round-trip order is preserved exactly (bit-exact).
        let t0 = sub_t0();
        let mut out = co.data.clone();
        let dm = self.d_model;
        out.par_chunks_mut(dm).enumerate().for_each(|(row, row_out)| {
            let it = row % t2;
            let pe_base = it * dm;
            for j in 0..dm {
                let a = f16::from_f32(row_out[j]).to_f32();
                let b = f16::from_f32(self.pe[pe_base + j]).to_f32();
                row_out[j] = f16::from_f32(a + b).to_f32();
            }
        });
        let dt_pe = sub_ms(t0);
        if sub_profile_enabled() {
            eprintln!("  conv_stem dt: conv1={:.1} conv2={:.1} conv3={:.1} perm={:.1} conv_out={:.1} pe={:.1} ms (total {:.1})",
                dt_conv1, dt_conv2, dt_conv3, dt_perm, dt_conv_out, dt_pe, sub_ms(t));
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
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str, num_heads: usize, d_model: usize) -> Result<Self> {
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

        let t = sub_t0();
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        let dt_qkv = sub_ms(t);

        // Reshape [1, s, dm] → [1, s, nh, hd] and transpose to [1, nh, s, hd].
        let q = swap_dims_12(&q.reshape(vec![1, s, nh, hd]));
        let k = swap_dims_12(&k.reshape(vec![1, s, nh, hd]));
        let v = swap_dims_12(&v.reshape(vec![1, s, nh, hd]));

        let scale = 1.0f32 / (hd as f32).sqrt();
        let t = sub_t0();
        let scores = matmul_qk(&q, &k);  // [1, nh, s, s]
        let attn = softmax_scaled(&scores, scale, false);  // full (non-causal) attention for audio encoder
        let attn_out = matmul_av(&attn, &v);  // [1, nh, s, hd]
        let dt_attn = sub_ms(t);
        let t = sub_t0();
        let attn_flat = swap_dims_12(&attn_out).reshape(vec![1, s, dm]);
        let dt_post = sub_ms(t);
        let t_out = sub_t0();
        let out = self.out_proj.forward(&attn_flat);
        if sub_profile_enabled() {
            eprintln!("    audio_attn qkv={:.1} attn={:.1} post={:.1} out={:.1} ms",
                dt_qkv, dt_attn, dt_post, sub_ms(t_out));
        }
        Ok(out)
    }
}

/// Scaled softmax over the last two dims of a 4D [b, nh, s, t] tensor.
/// If causal=true, row i masks out columns > i.
/// out[i, j] = exp(in[i, j]*scale - max) / sum;  0 for j > i (if causal).
///
/// Non-causal path (the audio encoder's only call site, causal=false): the
/// `valid_t == t` always, so the masked branch is dead.  This fast path scales
/// each score once into `out` (reused as the exp scratch), finds the row max,
/// then does the in-place exp + left-to-right sum + normalize — bit-identical
/// to the general path because every fmul/fsub/fexp sees the same inputs in
/// the same order; we just don't recompute `row[j]*scale` twice and skip the
/// dead `else` branch.
fn softmax_scaled(scores: &CpuTensor, scale: f32, causal: bool) -> CpuTensor {
    let s = scores.shape();
    let (b, nh, sl, t) = (s[0], s[1], s[2], s[3]);
    let mut out = vec![0.0f32; b * nh * sl * t];
    if !causal {
        // Non-causal fast path: scale once into out, single max pass, single
        // exp+sum pass, single normalize pass.
        out.par_chunks_mut(t).enumerate().for_each(|(idx, slab)| {
            let row = &scores.data[idx * t..(idx + 1) * t];
            // Scale into slab (this is `row[j]*scale`, computed once not twice).
            for j in 0..t { slab[j] = row[j] * scale; }
            let mut max_v = f32::NEG_INFINITY;
            for &v in slab.iter() { if v > max_v { max_v = v; } }
            let mut sum = 0.0f32;
            for v in slab.iter_mut() { *v = (*v - max_v).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in slab.iter_mut() { *v *= inv; }
        });
        return CpuTensor::new(out, vec![b, nh, sl, t]);
    }
    // General causal path (unchanged for bit-exactness on any future causal
    // caller; currently only the non-causal fast path above is exercised).
    out.par_chunks_mut(t).enumerate().for_each(|(idx, slab)| {
        let i = idx % sl;
        let valid_t = i + 1;
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
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: CpuLinear::load(weights, &format!("{prefix}.fc1"))?,
            fc2: CpuLinear::load(weights, &format!("{prefix}.fc2"))?,
        })
    }
    fn forward(&self, x: &CpuTensor) -> CpuTensor {
        let t = sub_t0();
        let mut h = self.fc1.forward(x);
        let dt_fc1 = sub_ms(t);
        let t = sub_t0();
        h.data.par_iter_mut().for_each(|v| {
            let x = *v;
            *v = 0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2));
        });
        let dt_gelu = sub_ms(t);
        let t = sub_t0();
        let out = self.fc2.forward(&h);
        let dt_fc2 = sub_ms(t);
        if sub_profile_enabled() {
            eprintln!("    audio_ffn  fc1={:.1} gelu={:.1} fc2={:.1} ms", dt_fc1, dt_gelu, dt_fc2);
        }
        out
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
    fn load(weights: &HashMap<String, RawTensor>, prefix: &str, num_heads: usize, d_model: usize) -> Result<Self> {
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
    pub fn load(weights: &HashMap<String, RawTensor>, prefix: &str, config: &AudioConfig) -> Result<Self> {
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

// ─── Decoder layer ─────────────────────────────────────────────────────────────────────────────────────────────────────────
//  Decoder layer
// ═════════════════════════════════════════════════════════════════════

pub(crate) struct CpuDecoderLayer {
    pub iln_w: Vec<f32>,
    pub pln_w: Vec<f32>,
    pub qn_w: Vec<f32>,
    pub kn_w: Vec<f32>,
    /// All matmul weights pre-converted to f32 at load time (eliminates
    /// per-call f16→f32 in the hot loop).  Memory cost: +1.2 GB for 0.6B model.
    pub qkv_w: CpuWeight,
    pub o_w: CpuWeight,
    pub gu_w: CpuWeight,
    pub dp_w: CpuWeight,
    pub nqh: usize, pub nkvh: usize, pub hd: usize, pub eps: f32,
}

impl CpuDecoderLayer {
    pub fn load(weights: &HashMap<String, RawTensor>, prefix: &str, cfg: &TextConfig) -> Result<Self> {
        Ok(Self {
            iln_w: load_vec(weights, &format!("{}.input_layernorm.weight", prefix))?,
            pln_w: load_vec(weights, &format!("{}.post_attention_layernorm.weight", prefix))?,
            qn_w: load_vec(weights, &format!("{}.self_attn.q_norm.weight", prefix))?,
            kn_w: load_vec(weights, &format!("{}.self_attn.k_norm.weight", prefix))?,
            qkv_w: load_fused_qkv_f16(weights, &format!("{}.self_attn", prefix))?.into_f32(),
            o_w: load_weight_f16(weights, &format!("{}.self_attn.o_proj.weight", prefix))?.into_f32(),
            gu_w: load_fused_gate_up_f16(weights, &format!("{}.mlp", prefix))?.into_f32(),
            dp_w: load_weight_f16(weights, &format!("{}.mlp.down_proj.weight", prefix))?.into_f32(),
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

        let t = sub_t0();
        let normed = rms_norm(&x, &self.iln_w, self.eps);
        let dt_rmsn = sub_ms(t);

        let t = sub_t0();
        let qkv = linear(&normed, &self.qkv_w);
        let dt_qkv = sub_ms(t);
        drop(normed);
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        let t = sub_t0();
        let q = qkv_extract_qkv_norm_rotary_cache(
            &qkv, &self.qn_w, &self.kn_w, cos_table, sin_table,
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            b, self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        );
        let dt_qkv_prep = sub_ms(t);
        drop(qkv);
        let cur_len = kv_start + s;

        let t = sub_t0();
        let attn_out = prefill_attention(
            &q, &kv.k[layer_idx], &kv.v[layer_idx],
            b, self.nqh, self.nkvh, kv.max_seq, self.hd, cur_len, use_causal,
        );
        let dt_attn = sub_ms(t);
        // attn_out is laid out as [b, nqh, s, hd] (logical) but bytes are [b, s, nqh, hd].
        // Reshape directly to [b, s, nqh*hd] for the O projection (no swap needed).
        let attn_flat = attn_out.reshape(vec![b, s, self.nqh * self.hd]);
        let mut h = x;
        let t = sub_t0();
        linear_accum(&mut h, &attn_flat, &self.o_w);
        let dt_o = sub_ms(t);
        drop(attn_flat);

        let t = sub_t0();
        let normed2 = rms_norm(&h, &self.pln_w, self.eps);
        let dt_rmsn2 = sub_ms(t);

        let t = sub_t0();
        let gu = linear(&normed2, &self.gu_w);
        let dt_gu = sub_ms(t);
        drop(normed2);
        let activated = silu_mul_split(&gu);
        drop(gu);
        let t = sub_t0();
        linear_accum(&mut h, &activated, &self.dp_w);
        let dt_dp = sub_ms(t);

        if sub_profile_enabled() {
            eprintln!("  text_dec.layer[{:02}] rmsn={:.1} qkv={:.1} qkv_prep={:.1} attn={:.1} o={:.1} rmsn2={:.1} gu={:.1} dp={:.1} ms",
                layer_idx, dt_rmsn, dt_qkv, dt_qkv_prep, dt_attn, dt_o, dt_rmsn2, dt_gu, dt_dp);
        }
        h
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text decoder
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct CpuTextDecoder {
    /// Embedding table stored as f32 (pre-converted at load time).
    pub embed_table: CpuWeight,
    /// LM head stored as f32 (pre-converted at load time).
    pub lm_head: CpuWeight,
    pub layers: Vec<CpuDecoderLayer>,
    pub norm_w: Vec<f32>,
    pub eps: f32,
}

impl CpuTextDecoder {
    pub fn load(weights: &HashMap<String, RawTensor>, prefix: &str, config: &TextConfig) -> Result<Self> {
        let embed_table = load_weight_f16(weights, &format!("{}.embed_tokens.weight", prefix))?.into_f32();
        let lm_head = load_weight_f16(weights, "thinker.lm_head.weight")?.into_f32();
        let norm_w = load_vec(weights, &format!("{}.norm.weight", prefix))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(CpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config)?);
        }
        Ok(Self { embed_table, lm_head, layers, norm_w, eps: config.rms_norm_eps as f32 })
    }

    /// Embed ids into [n, hidden] CpuTensor (f32). The embed table is already
    /// f32 (pre-converted at load), so this is a straight copy of the row.
    pub fn embed_ids(&self, ids: &[i64]) -> CpuTensor {
        let hidden = self.embed_table.cols;
        let n = ids.len();
        let mut data = vec![0.0f32; n * hidden];
        data.par_chunks_mut(hidden).enumerate().for_each(|(i, slab)| {
            let row = ids[i] as usize;
            let src = &self.embed_table.data[row * hidden..(row + 1) * hidden];
            slab.copy_from_slice(src);
        });
        CpuTensor::new(data, vec![n, hidden])
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
        let t = sub_t0();
        let h = rms_norm(&h, &self.norm_w, self.eps);
        let dt_final_rmsn = sub_ms(t);
        // lm_head: y = h @ W^T  where h is [1, sl, hidden], W is [classify_num, hidden].
        // m=sl which is > 1 in prefill, so use linear() (the gemm path).
        let t = sub_t0();
        let logits = linear(&h, &self.lm_head);
        let dt_lm_head = sub_ms(t);
        if sub_profile_enabled() {
            eprintln!("  text_dec.tail  final_rmsn={:.1} lm_head={:.1} ms", dt_final_rmsn, dt_lm_head);
        }
        logits
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

fn load_f32(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    td.as_f32().map_err(|e| anyhow::anyhow!("weight {} dtype error: {}", name, e))
}

fn load_vec(weights: &HashMap<String, RawTensor>, name: &str) -> Result<Vec<f32>> {
    let (data, _) = load_f32(weights, name)?;
    Ok(data)
}

fn load_f16_vec(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<f16>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    td.as_f16().map_err(|e| anyhow::anyhow!("weight {} dtype error: {}", name, e))
}

fn load_weight_f16(weights: &HashMap<String, RawTensor>, name: &str) -> Result<CpuWeightF16> {
    let (data, shape) = load_f16_vec(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    Ok(CpuWeightF16 { data, rows: shape[0], cols: shape[1] })
}

fn load_fused_qkv_f16(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<CpuWeightF16> {
    let (qw, qs) = load_f16_vec(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = load_f16_vec(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = load_f16_vec(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw);
    fused.extend_from_slice(&kw);
    fused.extend_from_slice(&vw);
    Ok(CpuWeightF16 { data: fused, rows: q_dim + 2 * kv_dim, cols: hidden })
}

fn load_fused_gate_up_f16(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<CpuWeightF16> {
    let (gw, gs) = load_f16_vec(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = load_f16_vec(weights, &format!("{}.up_proj.weight", prefix))?;
    let inter = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], inter); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * inter * hidden);
    fused.extend_from_slice(&gw);
    fused.extend_from_slice(&uw);
    Ok(CpuWeightF16 { data: fused, rows: 2 * inter, cols: hidden })
}
