//! GPU-resident text decoder for Qwen3-ASR.
//!
//! All operational tensors live in GPU memory as `GpuTensor` (CudaSlice<f16>).
//! cuBLAS handles matmul; hand-written CUDA kernels handle every element-wise
//! op (RMSNorm, SiLU·up, softmax, rotary, repeat-KV, embed, argmax, etc.).
//! There are no CPU↔GPU round-trips in the steady-state decode loop —
//! weights, KV cache, MRoPE tables, and the embedding table are all uploaded
//! once at load time and stay on the device.

use anyhow::Result;
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::TextConfig;
pub(crate) use crate::raw_tensor::RawTensor;

const KERNEL_SRC: &str = include_str!("kernels/kernels.cu");

// ═══════════════════════════════════════════════════════════════════════
//  GpuTensor — owned f16 tensor on the GPU
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuTensor {
    data: CudaSlice<f16>,
    shape: Vec<usize>,
}

impl Clone for GpuTensor {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), shape: self.shape.clone() }
    }
}

impl GpuTensor {
    pub fn new(data: CudaSlice<f16>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "GpuTensor data len mismatch");
        Self { data, shape }
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn numel(&self) -> usize { self.data.len() }
    pub fn data(&self) -> &CudaSlice<f16> { &self.data }
    /// Reshape without moving data.
    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert_eq!(self.data.len(), shape.iter().product::<usize>());
        Self { data: self.data.clone(), shape }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  CpuTensor — used only for weight loading + initial input staging
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
pub(crate) struct CpuTensor {
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f16>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "CpuTensor len mismatch");
        Self { data, shape }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  CudaState — context, stream, cuBLAS handle, kernel registry
// ═══════════════════════════════════════════════════════════════════════

// Some kernel slots are inherited from the ASR project's autoregressive
// decode loop (single-token kv-cache writes, qkv split, rotary, etc.) and
// aren't called by the aligner's pure-prefill pipeline.  They stay here so
// the NVRTC kernels.cu source stays in sync with both projects.
#[allow(dead_code)]
pub(crate) struct CudaKernels {
    pub rms_norm: CudaFunction,
    pub add_residual_rms_norm: CudaFunction,
    pub add: CudaFunction,
    pub add_inplace: CudaFunction,
    pub silu_mul: CudaFunction,
    pub silu_mul_split: CudaFunction,
    pub softmax_causal: CudaFunction,
    pub softmax_causal_online: CudaFunction,
    pub rotary_emb: CudaFunction,
    pub rms_norm_rotary: CudaFunction,
    pub repeat_kv_from_cache: CudaFunction,
    pub embed_lookup: CudaFunction,
    pub embed_lookup_single_i32: CudaFunction,
    pub argmax: CudaFunction,
    pub argmax_into_slot: CudaFunction,
    pub lm_head_gemv_argmax: CudaFunction,
    pub swap_dims_12: CudaFunction,
    pub qkv_split: CudaFunction,
    pub qkv_extract_q_norm_rotary: CudaFunction,
    pub qkv_extract_kv_norm_rotary_cache: CudaFunction,
    pub qkv_extract_qkv_norm_rotary_cache: CudaFunction,
    pub kv_cache_write: CudaFunction,
    pub kv_cache_write_pair: CudaFunction,
    pub gelu: CudaFunction,
    pub gelu_inplace: CudaFunction,
    pub layer_norm: CudaFunction,
    pub add_bias_inplace: CudaFunction,
    pub slice_dim2: CudaFunction,
    pub concat_dim2_write: CudaFunction,
    pub im2col_3x3: CudaFunction,
    pub conv_postprocess: CudaFunction,
    pub fused_gqa_decode: CudaFunction,
    pub fused_gqa_decode_split_p1: CudaFunction,
    pub fused_gqa_decode_split_p2: CudaFunction,
    pub scatter_audio_rows: CudaFunction,
}

pub(crate) struct CudaState {
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub k: CudaKernels,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        Self::new_with_ctx(&ctx)
    }

    pub fn new_with_ctx(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;

        // CUDA toolkit include for cuda_fp16.h
        let cuda_include = std::env::var("CUDA_PATH")
            .map(|p| format!("{}/include", p))
            .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
        let opts = CompileOptions {
            arch: None,
            include_paths: vec![cuda_include],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)
            .map_err(|e| anyhow::anyhow!("kernel compile failed: {:?}", e))?;
        let module = ctx.load_module(ptx)?;

        let k = CudaKernels {
            rms_norm: module.load_function("rms_norm_f16")?,
            add_residual_rms_norm: module.load_function("add_residual_rms_norm_f16")?,
            add: module.load_function("add_f16")?,
            add_inplace: module.load_function("add_inplace_f16")?,
            silu_mul: module.load_function("silu_mul_f16")?,
            silu_mul_split: module.load_function("silu_mul_split_f16")?,
            softmax_causal: module.load_function("softmax_scaled_causal_f16")?,
            softmax_causal_online: module.load_function("softmax_scaled_causal_online_f16")?,
            rotary_emb: module.load_function("rotary_emb_f16")?,
            rms_norm_rotary: module.load_function("rms_norm_rotary_f16")?,
            repeat_kv_from_cache: module.load_function("repeat_kv_from_cache_f16")?,
            embed_lookup: module.load_function("embed_lookup_f16")?,
            embed_lookup_single_i32: module.load_function("embed_lookup_single_i32_f16")?,
            argmax: module.load_function("argmax_f16")?,
            argmax_into_slot: module.load_function("argmax_into_slot_f16")?,
            lm_head_gemv_argmax: module.load_function("lm_head_gemv_argmax_f16")?,
            swap_dims_12: module.load_function("swap_dims_12_f16")?,
            qkv_split: module.load_function("qkv_split_f16")?,
            qkv_extract_q_norm_rotary: module.load_function("qkv_extract_q_norm_rotary_f16")?,
            qkv_extract_kv_norm_rotary_cache: module.load_function("qkv_extract_kv_norm_rotary_cache_f16")?,
            qkv_extract_qkv_norm_rotary_cache: module.load_function("qkv_extract_qkv_norm_rotary_cache_f16")?,
            kv_cache_write: module.load_function("kv_cache_write_f16")?,
            kv_cache_write_pair: module.load_function("kv_cache_write_pair_f16")?,
            gelu: module.load_function("gelu_f16")?,
            gelu_inplace: module.load_function("gelu_inplace_f16")?,
            layer_norm: module.load_function("layer_norm_f16")?,
            add_bias_inplace: module.load_function("add_bias_inplace_f16")?,
            slice_dim2: module.load_function("slice_dim2_f16")?,
            concat_dim2_write: module.load_function("concat_dim2_write_f16")?,
            im2col_3x3: module.load_function("im2col_3x3_s2p1_f16")?,
            conv_postprocess: module.load_function("conv_postprocess_f16")?,
            fused_gqa_decode: module.load_function("fused_gqa_decode_f16")?,
            fused_gqa_decode_split_p1: module.load_function("fused_gqa_decode_split_p1_f16")?,
            fused_gqa_decode_split_p2: module.load_function("fused_gqa_decode_split_p2_f16")?,
            scatter_audio_rows: module.load_function("scatter_audio_rows_f16")?,
        };

        Ok(Self { stream, blas, k })
    }

    pub fn upload_f16(&self, data: &[f16]) -> Result<CudaSlice<f16>> {
        Ok(self.stream.clone_htod(data)?)
    }
    pub fn upload_i64(&self, data: &[i64]) -> Result<CudaSlice<i64>> {
        Ok(self.stream.clone_htod(data)?)
    }
    pub fn alloc_zeros_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        Ok(self.stream.alloc_zeros::<f16>(n)?)
    }
    /// Allocate uninitialized f16 — caller MUST ensure every byte is written before read.
    /// Saves one memset_d8_async vs `alloc_zeros_f16` for cuBLAS/kernel outputs that are
    /// fully overwritten (beta=0 GEMM, fused attention writing all of `out`, etc.).
    pub fn alloc_uninit_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        Ok(unsafe { self.stream.alloc::<f16>(n)? })
    }
    pub fn download_f16(&self, slice: &CudaSlice<f16>) -> Result<Vec<f16>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }

    pub fn upload_tensor(&self, t: &CpuTensor) -> Result<GpuTensor> {
        let d = self.upload_f16(&t.data)?;
        Ok(GpuTensor::new(d, t.shape.clone()))
    }
    pub fn download_tensor(&self, t: &GpuTensor) -> Result<CpuTensor> {
        let d = self.download_f16(&t.data)?;
        Ok(CpuTensor::new(d, t.shape.clone()))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  cuBLAS wrappers (GPU-resident)
// ═══════════════════════════════════════════════════════════════════════

impl CudaState {
    /// y = x @ W^T   (x: [..., K], W: [N, K], y: [..., N])
    pub fn linear_gpu(&self, x: &GpuTensor, w: &GpuWeight) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(k, w.cols, "linear K mismatch: x last={} vs W cols={}", k, w.cols);
        let mut y = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: f16::from_f32(0.0), ldc: n as i32,
                },
                &w.data, &x.data, &mut y,
            )?;
        }
        let mut out_shape = x.shape().to_vec();
        out_shape[nd - 1] = n;
        Ok(GpuTensor::new(y, out_shape))
    }

    /// y = y + x @ W^T  — cuBLAS GEMM with beta=1, in-place accumulation on `y`.
    /// Used to fuse a residual add into a linear projection (saves an add_inplace launch).
    pub fn linear_gpu_accum(&self, y: &mut GpuTensor, x: &GpuTensor, w: &GpuWeight) -> Result<()> {
        let nd = x.shape().len();
        let m: usize = x.shape()[..nd - 1].iter().product();
        let k = x.shape()[nd - 1];
        let n = w.rows;
        assert_eq!(k, w.cols, "linear_gpu_accum K mismatch: x last={} vs W cols={}", k, w.cols);
        assert_eq!(y.numel(), m * n, "linear_gpu_accum y size mismatch");
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: f16::from_f32(1.0), ldc: n as i32,
                },
                &w.data, &x.data, &mut y.data,
            )?;
        }
        Ok(())
    }


    /// Scatter `audio[i] -> embeds[positions[i], :]` in place.  Aligner uses
    /// this to splice audio encoder outputs into the embedding table at the
    /// audio token positions, entirely on GPU.
    pub fn scatter_audio_rows(
        &self, embeds: &mut GpuTensor, audio: &GpuTensor,
        positions_dev: &CudaSlice<i32>,
    ) -> Result<()> {
        let n_audio = audio.shape()[0];
        let hidden = audio.shape()[1];
        assert_eq!(embeds.shape()[1], hidden, "scatter: hidden dim mismatch");
        let bs = (hidden as u32).min(1024);
        let cfg = LaunchConfig {
            grid_dim: (n_audio as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n_audio as i32;
        let h_i = hidden as i32;
        let mut bb = self.stream.launch_builder(&self.k.scatter_audio_rows);
        bb.arg(&mut embeds.data); bb.arg(&audio.data); bb.arg(positions_dev);
        bb.arg(&n_i); bb.arg(&h_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// scores = Q @ K^T  (Q: [b,h,m,d], K: [b,h,n,d] → [b,h,m,n])
    pub fn attention_qk(&self, q: &GpuTensor, k: &GpuTensor) -> Result<GpuTensor> {
        let b = q.shape()[0]; let h = q.shape()[1]; let m = q.shape()[2]; let n = k.shape()[2];
        let s = self.alloc_uninit_f16(b * h * m * n)?;
        let mut s_t = GpuTensor::new(s, vec![b, h, m, n]);
        self.attention_qk_into(q, k, &mut s_t)?;
        Ok(s_t)
    }

    /// Same as `attention_qk` but writes into a pre-allocated output buffer.
    /// Use in 28-layer prefill loops to avoid 28 × ~50ms cudaMalloc calls.
    pub fn attention_qk_into(&self, q: &GpuTensor, k: &GpuTensor, s: &mut GpuTensor) -> Result<()> {
        let b = q.shape()[0]; let h = q.shape()[1]; let m = q.shape()[2]; let d = q.shape()[3];
        let n = k.shape()[2];
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32, n: m as i32, k: d as i32,
                        alpha: f16::from_f32(1.0),
                        lda: d as i32, ldb: d as i32,
                        beta: f16::from_f32(0.0), ldc: n as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * d) as i64,
                    stride_c: (m * n) as i64,
                },
                &k.data, &q.data, &mut s.data,
            )?;
        }
        Ok(())
    }

    /// out = attn @ V  (attn: [b,h,m,n], V: [b,h,n,d] → [b,h,m,d])
    pub fn attention_av(&self, attn: &GpuTensor, v: &GpuTensor) -> Result<GpuTensor> {
        let b = attn.shape()[0]; let h = attn.shape()[1];
        let m = attn.shape()[2];
        let d = v.shape()[3];
        let o = self.alloc_uninit_f16(b * h * m * d)?;
        let mut o_t = GpuTensor::new(o, vec![b, h, m, d]);
        self.attention_av_into(attn, v, &mut o_t)?;
        Ok(o_t)
    }

    /// Grouped GQA prefill: replaces repeat_kv + attention_qk + softmax + attention_av.
    ///
    /// Instead of materialising the full repeated K/V tensors (nqh = nkvh × n_rep),
    /// loops over nkvh groups of n_rep Q heads each, doing a smaller strided-batched
    /// GEMM per group.  Eliminates:
    ///   • 2 × repeat_kv kernel launches per layer
    ///   • 2 × (n_rep − 1) × nkvh × s × d elements of dead memory copies
    ///
    /// For this model (nqh=16, nkvh=8, n_rep=2, s≈4567, d=128):
    ///   saves ~47 MB of useless HBM copies per layer × 28 layers ≈ 1.3 GB total.
    pub fn grouped_gqa_prefill(
        &self,
        q: &GpuTensor, k_cache: &CudaSlice<f16>, v_cache: &CudaSlice<f16>,
        nkvh: usize, max_seq: usize,
        scale: f32, causal: bool,
    ) -> Result<GpuTensor> {
        let b = q.shape()[0];
        let nqh = q.shape()[1];
        let m   = q.shape()[2];
        let d   = q.shape()[3];
        let n   = m; // kv length == q length for prefill
        let n_rep = nqh / nkvh;
        let batch_per_group = n_rep as i32;
        let score_sz = batch_per_group as usize * m * n;

        let mut scores_buf   = self.alloc_uninit_f16(b * nqh * m * n)?;
        let attn_out_buf     = self.alloc_uninit_f16(b * nqh * m * n)?; // softmax output
        let mut out_buf      = self.alloc_uninit_f16(b * nqh * m * d)?; // final AV output

        // Pass 1: compute all QK scores (nkvh batched GEMM calls, stride_a=0 for K reuse)
        for g in 0..nkvh {
            let q_off  = g * n_rep * m * d;
            let kv_off = g * max_seq * d;
            let s_off  = g * n_rep * m * n;
            unsafe {
                self.blas.gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_T,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: n as i32, n: m as i32, k: d as i32,
                            alpha: f16::from_f32(1.0),
                            lda: d as i32, ldb: d as i32,
                            beta: f16::from_f32(0.0), ldc: n as i32,
                        },
                        batch_size: batch_per_group,
                        stride_a: 0, // single K head shared by n_rep Q heads
                        stride_b: (m * d) as i64,
                        stride_c: (m * n) as i64,
                    },
                    &k_cache.slice(kv_off..kv_off + max_seq * d),
                    &q.data().slice(q_off..q_off + n_rep * m * d),
                    &mut scores_buf.slice_mut(s_off..s_off + score_sz),
                )?;
            }
        }

        // Softmax over all [b, nqh, m, n] — writes attention weights to attn_out_buf
        let scores_t = GpuTensor::new(scores_buf, vec![b, nqh, m, n]);
        let mut attn_out_t = GpuTensor::new(attn_out_buf, vec![b, nqh, m, n]);
        self.softmax_scaled_causal_into(&scores_t, scale, causal, &mut attn_out_t)?;

        // Pass 2: AV products — stride_a=0 shares V head across n_rep Q heads
        for g in 0..nkvh {
            let kv_off = g * max_seq * d;
            let a_off  = g * n_rep * m * n;
            let o_off  = g * n_rep * m * d;
            unsafe {
                self.blas.gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: d as i32, n: m as i32, k: n as i32,
                            alpha: f16::from_f32(1.0),
                            lda: d as i32, ldb: n as i32,
                            beta: f16::from_f32(0.0), ldc: d as i32,
                        },
                        batch_size: batch_per_group,
                        stride_a: 0, // single V head shared
                        stride_b: (m * n) as i64,
                        stride_c: (m * d) as i64,
                    },
                    &v_cache.slice(kv_off..kv_off + max_seq * d),
                    &attn_out_t.data().slice(a_off..a_off + score_sz),
                    &mut out_buf.slice_mut(o_off..o_off + n_rep * m * d),
                )?;
            }
        }

        Ok(GpuTensor::new(out_buf, vec![b, nqh, m, d]))
    }

    /// Same as `attention_av` but writes into a pre-allocated output.
    pub fn attention_av_into(&self, attn: &GpuTensor, v: &GpuTensor, o: &mut GpuTensor) -> Result<()> {
        let b = attn.shape()[0]; let h = attn.shape()[1];
        let m = attn.shape()[2]; let n = attn.shape()[3];
        let d = v.shape()[3];
        let batch = (b * h) as i32;
        unsafe {
            self.blas.gemm_strided_batched(
                StridedBatchedConfig {
                    gemm: GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: d as i32, n: m as i32, k: n as i32,
                        alpha: f16::from_f32(1.0),
                        lda: d as i32, ldb: n as i32,
                        beta: f16::from_f32(0.0), ldc: d as i32,
                    },
                    batch_size: batch,
                    stride_a: (n * d) as i64,
                    stride_b: (m * n) as i64,
                    stride_c: (m * d) as i64,
                },
                &v.data, &attn.data, &mut o.data,
            )?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  GPU element-wise kernel wrappers
// ═══════════════════════════════════════════════════════════════════════

fn block_for_reduction(last: usize) -> u32 {
    // Power-of-two block size, max 1024, at least 32. The reduction needs bs to be power of 2.
    let mut bs: u32 = 32;
    let target = last as u32;
    while bs < target && bs < 1024 { bs *= 2; }
    bs.min(1024).max(32)
}

impl CudaState {
    pub fn rms_norm(&self, x: &GpuTensor, w: &CudaSlice<f16>, eps: f32) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let last_i = last as i32;
        let outer_i = outer as i32;
        let mut b = self.stream.launch_builder(&self.k.rms_norm);
        b.arg(&mut out); b.arg(&x.data); b.arg(w);
        b.arg(&last_i); b.arg(&outer_i); b.arg(&eps);
        unsafe { b.launch(cfg) }?;
        Ok(GpuTensor::new(out, x.shape().to_vec()))
    }

    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
        let n = a.numel();
        let mut out = self.alloc_uninit_f16(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add);
        bb.arg(&mut out); bb.arg(&a.data); bb.arg(&b.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, a.shape().to_vec()))
    }

    pub fn add_inplace(&self, a: &mut GpuTensor, b: &GpuTensor) -> Result<()> {
        let n = a.numel();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_inplace);
        bb.arg(&mut a.data); bb.arg(&b.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Fused: residual += add_in (in-place), then out = rms_norm(residual, w).
    /// Saves one kernel launch vs separate `add` + `rms_norm`.
    #[allow(dead_code)]  // ASR autoregressive decode path; aligner does pure prefill.
    pub fn add_residual_rms_norm(&self, residual: &mut GpuTensor, add_in: &GpuTensor, w: &CudaSlice<f16>, eps: f32) -> Result<GpuTensor> {
        let nd = residual.shape().len();
        let last = residual.shape()[nd - 1];
        let outer: usize = residual.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(residual.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let last_i = last as i32; let outer_i = outer as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_residual_rms_norm);
        bb.arg(&mut residual.data); bb.arg(&mut out); bb.arg(&add_in.data); bb.arg(w);
        bb.arg(&last_i); bb.arg(&outer_i); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, residual.shape().to_vec()))
    }

    pub fn silu_mul_split(&self, gu: &GpuTensor) -> Result<GpuTensor> {
        let nd = gu.shape().len();
        let two_inter = gu.shape()[nd - 1];
        let inter = two_inter / 2;
        let outer: usize = gu.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(outer * inter)?;
        let cfg = LaunchConfig::for_num_elems((outer * inter) as u32);
        let outer_i = outer as i32;
        let inter_i = inter as i32;
        let mut bb = self.stream.launch_builder(&self.k.silu_mul_split);
        bb.arg(&mut out); bb.arg(&gu.data);
        bb.arg(&outer_i); bb.arg(&inter_i);
        unsafe { bb.launch(cfg) }?;
        let mut out_shape = gu.shape().to_vec();
        out_shape[nd - 1] = inter;
        Ok(GpuTensor::new(out, out_shape))
    }

    /// scores [b,h,m,n] → softmax(scale * scores) with optional causal mask. Out of place.
    pub fn softmax_scaled_causal(&self, scores: &GpuTensor, scale: f32, causal: bool) -> Result<GpuTensor> {
        let s = scores.shape();
        // Both kernels (legacy and online) write every output position exactly
        // once, so the buffer can be left uninitialized.
        let out = self.alloc_uninit_f16(scores.numel())?;
        let mut out_t = GpuTensor::new(out, s.to_vec());
        self.softmax_scaled_causal_into(scores, scale, causal, &mut out_t)?;
        Ok(out_t)
    }

    /// Same as `softmax_scaled_causal` but writes into a pre-allocated output.
    /// Both the legacy and online kernels write every position of `out`
    /// (online: valid positions get softmax values, masked positions get 0
    /// in a dedicated tail loop), so the buffer does not need pre-zeroing.
    pub fn softmax_scaled_causal_into(&self, scores: &GpuTensor, scale: f32, causal: bool, out: &mut GpuTensor) -> Result<()> {
        let s = scores.shape();
        assert_eq!(s.len(), 4, "softmax expects [b,h,m,n]");
        let bh = s[0] * s[1];
        let m = s[2]; let n = s[3];
        let rows = bh * m;
        // Block size is the same for both the legacy 3-pass kernel and the
        // online (single-pass max+sum) kernel.  Keeping the same block size
        // preserves the partial-sum reduction order, which in turn preserves
        // bit-exact f16 outputs on the JA fixture's punctuation tokens —
        // those have a degenerate zero-duration argmax that is sensitive to
        // sub-ULP rounding differences.
        let bs = block_for_reduction(n);
        let m_i = m as i32;
        let n_i = n as i32;
        let causal_i: i32 = if causal { 1 } else { 0 };
        if causal {
            // Online (single-pass max+sum) kernel — needs 2*bs floats shared mem.
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (bs, 1, 1),
                shared_mem_bytes: bs * 4 * 2,
            };
            let mut bb = self.stream.launch_builder(&self.k.softmax_causal_online);
            bb.arg(&mut out.data); bb.arg(&scores.data);
            bb.arg(&m_i); bb.arg(&n_i); bb.arg(&scale); bb.arg(&causal_i);
            unsafe { bb.launch(cfg) }?;
        } else {
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (bs, 1, 1),
                shared_mem_bytes: bs * 4,
            };
            let mut bb = self.stream.launch_builder(&self.k.softmax_causal);
            bb.arg(&mut out.data); bb.arg(&scores.data);
            bb.arg(&m_i); bb.arg(&n_i); bb.arg(&scale); bb.arg(&causal_i);
            unsafe { bb.launch(cfg) }?;
        }
        Ok(())
    }

    /// Q/K rotary embedding. x [b, h, s, d], cos/sin [total_s, d] (full table).
    /// pos_offset is added to each `is` to index into cos/sin.
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn rotary_emb(&self, x: &GpuTensor, cos: &CudaSlice<f16>, sin: &CudaSlice<f16>, pos_offset: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let n = x.numel();
        let mut out = self.alloc_uninit_f16(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let s0 = s[0] as i32; let s1 = s[1] as i32; let s2 = s[2] as i32; let s3 = s[3] as i32;
        let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.rotary_emb);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(cos); bb.arg(sin);
        bb.arg(&s0); bb.arg(&s1); bb.arg(&s2); bb.arg(&s3); bb.arg(&po);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, s.to_vec()))
    }

    /// Fused per-head RMSNorm + rotary on Q or K. x [b, h, s, d].
    /// cos/sin: [total_s, d] (full table). pos_offset is added to each `is`.
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn rms_norm_rotary(&self, x: &GpuTensor, w: &CudaSlice<f16>,
                           cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
                           pos_offset: usize, eps: f32) -> Result<GpuTensor>
    {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, h, sl, d) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * h * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let h_i = h as i32; let sl_i = sl as i32; let d_i = d as i32;
        let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.rms_norm_rotary);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&po);
        bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, sl, d]))
    }

    /// Repeat-KV from a sparse KV cache, producing a dense [b, nqh, cur_len, d] view.
    pub fn repeat_kv_from_cache(&self, cache: &CudaSlice<f16>,
        b: usize, nkvh: usize, max_seq: usize, d: usize, n_rep: usize, cur_len: usize,
    ) -> Result<GpuTensor> {
        let nqh = nkvh * n_rep;
        let total = b * nqh * cur_len * d;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let nrep_i = n_rep as i32; let cur_i = cur_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.repeat_kv_from_cache);
        bb.arg(&mut out); bb.arg(cache);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&nrep_i); bb.arg(&cur_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, nqh, cur_len, d]))
    }

    pub fn embed_lookup(&self, table: &GpuWeight, ids_gpu: &CudaSlice<i64>) -> Result<GpuTensor> {
        let n = ids_gpu.len();
        let d = table.cols;
        let mut out = self.alloc_uninit_f16(n * d)?;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32; let d_i = d as i32;
        let mut bb = self.stream.launch_builder(&self.k.embed_lookup);
        bb.arg(&mut out); bb.arg(&table.data); bb.arg(ids_gpu);
        bb.arg(&n_i); bb.arg(&d_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![n, d]))
    }

    pub fn swap_dims_12(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (d0, d1, d2, d3) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let cfg = LaunchConfig::for_num_elems(x.numel() as u32);
        let d0_i = d0 as i32; let d1_i = d1 as i32; let d2_i = d2 as i32; let d3_i = d3 as i32;
        let mut bb = self.stream.launch_builder(&self.k.swap_dims_12);
        bb.arg(&mut out); bb.arg(&x.data);
        bb.arg(&d0_i); bb.arg(&d1_i); bb.arg(&d2_i); bb.arg(&d3_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![d0, d2, d1, d3]))
    }

    /// Extract one head-group from a fused QKV tensor in [b, h, s, d] layout.
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn qkv_split(&self, qkv: &GpuTensor, h: usize, d: usize, offset: usize) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total) = (s[0], s[1], s[2]);
        let mut out = self.alloc_uninit_f16(b * h * sl * d)?;
        let cfg = LaunchConfig::for_num_elems((b * h * sl * d) as u32);
        let b_i = b as i32; let sl_i = sl as i32; let h_i = h as i32; let d_i = d as i32;
        let tot_i = total as i32; let off_i = offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_split);
        bb.arg(&mut out); bb.arg(&qkv.data);
        bb.arg(&b_i); bb.arg(&sl_i); bb.arg(&h_i); bb.arg(&d_i);
        bb.arg(&tot_i); bb.arg(&off_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, sl, d]))
    }

    /// Fused: extract Q from fused QKV, apply RMSNorm + rotary in one launch.
    /// qkv: [b, s, q_dim + 2*kv_dim].  Returns Q: [b, nqh, s, d].
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn qkv_extract_q_norm_rotary(&self, qkv: &GpuTensor, qn_w: &CudaSlice<f16>,
        cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
        nqh: usize, d: usize, pos_offset: usize, eps: f32,
    ) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let mut q_out = self.alloc_uninit_f16(b * nqh * sl * d)?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * nqh * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let sl_i = sl as i32;
        let d_i = d as i32; let tot_i = total_cols as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_q_norm_rotary);
        bb.arg(&mut q_out); bb.arg(&qkv.data); bb.arg(qn_w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&sl_i); bb.arg(&d_i);
        bb.arg(&tot_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(q_out, vec![b, nqh, sl, d]))
    }

    /// Fused: extract K (with RMSNorm+rotary) and V (raw) from fused QKV, write both into KV cache.
    /// Replaces qkv_split×2 + rms_norm_rotary + kv_cache_write_pair (4 launches → 1).
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn qkv_extract_kv_norm_rotary_cache(&self,
        k_cache: &mut CudaSlice<f16>, v_cache: &mut CudaSlice<f16>,
        qkv: &GpuTensor, kn_w: &CudaSlice<f16>,
        cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
        nkvh: usize, d: usize, q_dim: usize, kv_dim: usize,
        max_seq: usize, start: usize, pos_offset: usize, eps: f32,
    ) -> Result<()> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * nkvh * sl) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nkvh_i = nkvh as i32; let sl_i = sl as i32;
        let d_i = d as i32; let tot_i = total_cols as i32;
        let q_i = q_dim as i32; let kv_i = kv_dim as i32;
        let max_i = max_seq as i32; let start_i = start as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_kv_norm_rotary_cache);
        bb.arg(k_cache); bb.arg(v_cache); bb.arg(&qkv.data); bb.arg(kn_w);
        bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&tot_i);
        bb.arg(&q_i); bb.arg(&kv_i);
        bb.arg(&max_i); bb.arg(&start_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Single-kernel fused QKV extraction: builds Q, K (with norm+rotary) and V (raw),
    /// writes Q to a fresh output and K/V into their respective caches.
    /// Replaces qkv_extract_q_norm_rotary + qkv_extract_kv_norm_rotary_cache (2 launches → 1).
    pub fn qkv_extract_qkv_norm_rotary_cache(&self,
        k_cache: &mut CudaSlice<f16>, v_cache: &mut CudaSlice<f16>,
        qkv: &GpuTensor, qn_w: &CudaSlice<f16>, kn_w: &CudaSlice<f16>,
        cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
        nqh: usize, nkvh: usize, d: usize, q_dim: usize, kv_dim: usize,
        max_seq: usize, start: usize, pos_offset: usize, eps: f32,
    ) -> Result<GpuTensor> {
        let s = qkv.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, total_cols) = (s[0], s[1], s[2]);
        let mut q_out = self.alloc_uninit_f16(b * nqh * sl * d)?;
        let bs = block_for_reduction(d);
        let cfg = LaunchConfig {
            grid_dim: ((b * sl) as u32, (nqh + nkvh) as u32, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let sl_i = sl as i32; let d_i = d as i32; let tot_i = total_cols as i32;
        let q_i = q_dim as i32; let kv_i = kv_dim as i32;
        let max_i = max_seq as i32; let start_i = start as i32; let po = pos_offset as i32;
        let mut bb = self.stream.launch_builder(&self.k.qkv_extract_qkv_norm_rotary_cache);
        bb.arg(&mut q_out); bb.arg(k_cache); bb.arg(v_cache); bb.arg(&qkv.data);
        bb.arg(qn_w); bb.arg(kn_w); bb.arg(cos); bb.arg(sin);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&sl_i); bb.arg(&d_i); bb.arg(&tot_i);
        bb.arg(&q_i); bb.arg(&kv_i);
        bb.arg(&max_i); bb.arg(&start_i); bb.arg(&po); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(q_out, vec![b, nqh, sl, d]))
    }

    /// Write a [b, nkvh, s_new, d] tensor into a [b, nkvh, max_seq, d] cache at offset `start`.
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn kv_cache_write(&self, cache: &mut CudaSlice<f16>, k_new: &GpuTensor,
        b: usize, nkvh: usize, max_seq: usize, d: usize, start: usize,
    ) -> Result<()> {
        let s_new = k_new.shape()[2];
        let total = b * nkvh * s_new * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let start_i = start as i32; let snew_i = s_new as i32;
        let mut bb = self.stream.launch_builder(&self.k.kv_cache_write);
        bb.arg(cache); bb.arg(&k_new.data);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&snew_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Fused: write K and V into their caches in one kernel.
    #[allow(dead_code)]  // ASR autoregressive decode path.
    pub fn kv_cache_write_pair(&self, k_cache: &mut CudaSlice<f16>, v_cache: &mut CudaSlice<f16>,
        k_new: &GpuTensor, v_new: &GpuTensor,
        b: usize, nkvh: usize, max_seq: usize, d: usize, start: usize,
    ) -> Result<()> {
        let s_new = k_new.shape()[2];
        let total = b * nkvh * s_new * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let nkvh_i = nkvh as i32; let max_i = max_seq as i32;
        let d_i = d as i32; let start_i = start as i32; let snew_i = s_new as i32;
        let mut bb = self.stream.launch_builder(&self.k.kv_cache_write_pair);
        bb.arg(k_cache); bb.arg(v_cache); bb.arg(&k_new.data); bb.arg(&v_new.data);
        bb.arg(&b_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&snew_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    pub fn gelu_inplace(&self, x: &mut GpuTensor) -> Result<()> {
        let n = x.numel();
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.gelu_inplace);
        bb.arg(&mut x.data); bb.arg(&n_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    pub fn layer_norm(&self, x: &GpuTensor, w: &CudaSlice<f16>, bias: &CudaSlice<f16>, eps: f32) -> Result<GpuTensor> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let mut out = self.alloc_uninit_f16(x.numel())?;
        let bs = block_for_reduction(last);
        let cfg = LaunchConfig {
            grid_dim: (outer as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4 * 2,  // sum + sum_sq
        };
        let last_i = last as i32; let outer_i = outer as i32;
        let mut bb = self.stream.launch_builder(&self.k.layer_norm);
        bb.arg(&mut out); bb.arg(&x.data); bb.arg(w); bb.arg(bias);
        bb.arg(&last_i); bb.arg(&outer_i); bb.arg(&eps);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, x.shape().to_vec()))
    }

    pub fn add_bias_inplace(&self, x: &mut GpuTensor, bias: &CudaSlice<f16>) -> Result<()> {
        let nd = x.shape().len();
        let last = x.shape()[nd - 1];
        let outer: usize = x.shape()[..nd - 1].iter().product();
        let cfg = LaunchConfig::for_num_elems((outer * last) as u32);
        let outer_i = outer as i32; let last_i = last as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_bias_inplace);
        bb.arg(&mut x.data); bb.arg(bias);
        bb.arg(&outer_i); bb.arg(&last_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// Slice [b, h, s, d] → [b, h, len, d] taking rows `start..start+len`.
    pub fn slice_dim2(&self, x: &GpuTensor, start: usize, len: usize) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, h, sl, d) = (s[0], s[1], s[2], s[3]);
        assert!(start + len <= sl);
        let total = b * h * len * d;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let h_i = h as i32; let sl_i = sl as i32;
        let d_i = d as i32; let start_i = start as i32; let len_i = len as i32;
        let mut bb = self.stream.launch_builder(&self.k.slice_dim2);
        bb.arg(&mut out); bb.arg(&x.data);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&sl_i); bb.arg(&d_i);
        bb.arg(&start_i); bb.arg(&len_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, h, len, d]))
    }

    /// Write a [b, h, len, d] chunk into a pre-allocated [b, h, s, d] buffer at `dst_offset`.
    pub fn concat_dim2_write(&self, dst: &mut CudaSlice<f16>, src: &GpuTensor,
        b: usize, h: usize, s: usize, d: usize, dst_offset: usize,
    ) -> Result<()> {
        let len = src.shape()[2];
        let total = b * h * len * d;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let b_i = b as i32; let h_i = h as i32; let s_i = s as i32;
        let d_i = d as i32; let off_i = dst_offset as i32; let len_i = len as i32;
        let mut bb = self.stream.launch_builder(&self.k.concat_dim2_write);
        bb.arg(dst); bb.arg(&src.data);
        bb.arg(&b_i); bb.arg(&h_i); bb.arg(&s_i); bb.arg(&d_i);
        bb.arg(&off_i); bb.arg(&len_i);
        unsafe { bb.launch(cfg) }?;
        Ok(())
    }

    /// 3×3 conv2d with stride=2, padding=1 — via im2col + cuBLAS GEMM + fused bias/GELU.
    /// input: [b, c_in, h, w]; weight: [c_out, c_in, 3, 3]; bias: [c_out].
    /// Returns [b, c_out, h_out, w_out] with GELU applied.
    pub fn conv2d_3x3_s2p1_gelu(&self, x: &GpuTensor, weight: &CudaSlice<f16>,
        c_out: usize, c_in: usize, bias: &CudaSlice<f16>,
    ) -> Result<GpuTensor> {
        let s = x.shape();
        assert_eq!(s.len(), 4);
        let (b, c_in_chk, h, w) = (s[0], s[1], s[2], s[3]);
        assert_eq!(c_in_chk, c_in);
        let h_out = (h + 2 - 3) / 2 + 1;
        let w_out = (w + 2 - 3) / 2 + 1;

        // im2col: [b*h_out*w_out, c_in*9] (row-major)
        let m = b * h_out * w_out;
        let k = c_in * 9;
        let mut col = self.alloc_uninit_f16(m * k)?;
        let cfg = LaunchConfig::for_num_elems((m * k) as u32);
        let b_i = b as i32; let cin_i = c_in as i32;
        let h_i = h as i32; let w_i = w as i32;
        let ho_i = h_out as i32; let wo_i = w_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.im2col_3x3);
        bb.arg(&mut col); bb.arg(&x.data);
        bb.arg(&b_i); bb.arg(&cin_i); bb.arg(&h_i); bb.arg(&w_i);
        bb.arg(&ho_i); bb.arg(&wo_i);
        unsafe { bb.launch(cfg) }?;

        // GEMM: gemm_out[m, c_out] = col[m, k] @ weight^T[k, c_out]
        // We need weight reinterpreted as [c_out, c_in*9] (already that layout since
        // weight is stored as [c_out, c_in, 3, 3] row-major = [c_out, c_in*9]).
        let n = c_out;
        let mut gemm_out = self.alloc_uninit_f16(m * n)?;
        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,  // weight: K-major in row-major = T in col-major
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32, n: m as i32, k: k as i32,
                    alpha: f16::from_f32(1.0),
                    lda: k as i32, ldb: k as i32,
                    beta: f16::from_f32(0.0), ldc: n as i32,
                },
                weight, &col, &mut gemm_out,
            )?;
        }
        drop(col);

        // Post: [b*h_out*w_out, c_out] → [b, c_out, h_out, w_out] with bias+GELU
        let total = b * c_out * h_out * w_out;
        let mut out = self.alloc_uninit_f16(total)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let cout_i = c_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.conv_postprocess);
        bb.arg(&mut out); bb.arg(&gemm_out); bb.arg(bias);
        bb.arg(&b_i); bb.arg(&cout_i); bb.arg(&ho_i); bb.arg(&wo_i);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, c_out, h_out, w_out]))
    }

    /// Fused GQA attention for decode (s_q = 1).  Replaces repeat_kv + attention_qk + softmax + attention_av
    /// (4 launches + 2 big allocs) with a single kernel that reads K/V directly from the cache.
    /// Q: [b, nqh, 1, d].  Returns out: [b, nqh, 1, d].
    pub fn fused_gqa_decode(&self, q: &GpuTensor,
        k_cache: &CudaSlice<f16>, v_cache: &CudaSlice<f16>,
        nkvh: usize, max_seq: usize, cur_len: usize, scale: f32,
    ) -> Result<GpuTensor> {
        let s = q.shape();
        assert_eq!(s.len(), 4);
        assert_eq!(s[2], 1, "fused_gqa_decode requires s_q = 1");
        let (b, nqh, _, d) = (s[0], s[1], s[2], s[3]);
        let mut out = self.alloc_uninit_f16(b * nqh * d)?;
        // Adaptive block size: scale parallelism with cur_len so each thread does roughly the
        // same amount of work whether we have 200 or 2000 tokens of context.  bs must be a
        // multiple of d (so the stage-4 t-chunks layout works) and a power of 2 (for reductions).
        let bs: u32 = if cur_len > 1024 { 1024 }
                      else if cur_len > 512 { 512 }
                      else { 256 };
        let t_chunks = (bs as usize / d).max(1);
        // shared: scores[cur_len] + partial_out[d * t_chunks], both f32.
        let smem_bytes = (cur_len + d * t_chunks) * 4;
        let cfg = LaunchConfig {
            grid_dim: ((b * nqh) as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };
        let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
        let max_i = max_seq as i32; let d_i = d as i32; let cur_i = cur_len as i32;
        let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode);
        bb.arg(&mut out); bb.arg(&q.data); bb.arg(k_cache); bb.arg(v_cache);
        bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&max_i);
        bb.arg(&d_i); bb.arg(&cur_i); bb.arg(&scale);
        unsafe { bb.launch(cfg) }?;
        Ok(GpuTensor::new(out, vec![b, nqh, 1, d]))
    }

    /// Split-K variant of `fused_gqa_decode`: divides the cur_len axis into N chunks across
    /// independent blocks, then merges with an online-softmax correction kernel.  Use when
    /// cur_len is large enough that the single-block kernel underutilizes the SMs.
    pub fn fused_gqa_decode_split(&self, q: &GpuTensor,
        k_cache: &CudaSlice<f16>, v_cache: &CudaSlice<f16>,
        nkvh: usize, max_seq: usize, cur_len: usize, scale: f32,
        chunk_size: usize,
    ) -> Result<GpuTensor> {
        let s = q.shape();
        let (b, nqh, _, d) = (s[0], s[1], s[2], s[3]);
        let n_chunks = (cur_len + chunk_size - 1) / chunk_size;

        // Buffers: partial out [b, nqh, n_chunks, d] f32, max/sum [b, nqh, n_chunks] f32.
        let part_out = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks * d)?;
        let part_max = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks)?;
        let part_sum = self.stream.alloc_zeros::<f32>(b * nqh * n_chunks)?;

        // Phase 1: per-chunk partial computation.
        let bs: u32 = 256;
        let t_split = (bs as usize / d).max(1);
        let smem_bytes = (chunk_size + d * t_split) * 4;
        let mut part_out_buf = part_out;
        let mut part_max_buf = part_max;
        let mut part_sum_buf = part_sum;
        {
            let cfg = LaunchConfig {
                grid_dim: ((b * nqh) as u32, n_chunks as u32, 1),
                block_dim: (bs, 1, 1),
                shared_mem_bytes: smem_bytes as u32,
            };
            let b_i = b as i32; let nqh_i = nqh as i32; let nkvh_i = nkvh as i32;
            let max_i = max_seq as i32; let d_i = d as i32;
            let cur_i = cur_len as i32; let cs_i = chunk_size as i32; let nc_i = n_chunks as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p1);
            bb.arg(&mut part_out_buf); bb.arg(&mut part_max_buf); bb.arg(&mut part_sum_buf);
            bb.arg(&q.data); bb.arg(k_cache); bb.arg(v_cache);
            bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nkvh_i); bb.arg(&max_i); bb.arg(&d_i);
            bb.arg(&cur_i); bb.arg(&cs_i); bb.arg(&nc_i); bb.arg(&scale);
            unsafe { bb.launch(cfg) }?;
        }

        // Phase 2: merge chunks → final output.
        let mut out = self.alloc_uninit_f16(b * nqh * d)?;
        {
            let cfg = LaunchConfig {
                grid_dim: ((b * nqh) as u32, 1, 1),
                block_dim: (d as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let b_i = b as i32; let nqh_i = nqh as i32; let nc_i = n_chunks as i32; let d_i = d as i32;
            let mut bb = self.stream.launch_builder(&self.k.fused_gqa_decode_split_p2);
            bb.arg(&mut out); bb.arg(&part_out_buf); bb.arg(&part_max_buf); bb.arg(&part_sum_buf);
            bb.arg(&b_i); bb.arg(&nqh_i); bb.arg(&nc_i); bb.arg(&d_i);
            unsafe { bb.launch(cfg) }?;
        }
        Ok(GpuTensor::new(out, vec![b, nqh, 1, d]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  GpuWeight — 2-D weight matrix on GPU
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuWeight {
    pub data: CudaSlice<f16>,
    pub rows: usize,
    pub cols: usize,
}

// ═══════════════════════════════════════════════════════════════════════
//  KV cache (GPU-resident, pre-allocated)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuKvCache {
    pub k: Vec<CudaSlice<f16>>,   // per layer: [b, nkvh, max_seq, d]
    pub v: Vec<CudaSlice<f16>>,
    pub cur_len: usize,
    pub max_seq: usize,
}

impl GpuKvCache {
    pub fn new(cuda: &CudaState, num_layers: usize, b: usize, nkvh: usize, max_seq: usize, d: usize) -> Result<Self> {
        let mut k = Vec::with_capacity(num_layers);
        let mut v = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            k.push(cuda.alloc_zeros_f16(b * nkvh * max_seq * d)?);
            v.push(cuda.alloc_zeros_f16(b * nkvh * max_seq * d)?);
        }
        Ok(Self { k, v, cur_len: 0, max_seq })
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
//  Weight loading helpers (re-exported for other modules)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) mod gpu_helpers {
    use super::*;
    pub(crate) fn load_gpu_weight(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<GpuWeight> {
        super::load_gpu_weight(cuda, weights, name)
    }
    pub(crate) fn load_gpu_vec(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<CudaSlice<f16>> {
        super::load_gpu_vec(cuda, weights, name)
    }
}

fn get_weight_f16(weights: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<f16>, Vec<usize>)> {
    let td = weights.get(name).ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))?;
    td.as_f16().map_err(|e| anyhow::anyhow!("weight {} dtype error: {}", name, e))
}

fn load_gpu_weight(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<GpuWeight> {
    let (data_f16, shape) = get_weight_f16(weights, name)?;
    assert_eq!(shape.len(), 2, "weight {} should be 2D", name);
    let dev = cuda.upload_f16(&data_f16)?;
    Ok(GpuWeight { data: dev, rows: shape[0], cols: shape[1] })
}

fn load_gpu_vec(cuda: &CudaState, weights: &HashMap<String, RawTensor>, name: &str) -> Result<CudaSlice<f16>> {
    let (data_f16, _shape) = get_weight_f16(weights, name)?;
    cuda.upload_f16(&data_f16)
}

fn load_fused_qkv_weight(
    weights: &HashMap<String, RawTensor>, prefix: &str, cuda: &CudaState,
) -> Result<(GpuWeight, usize, usize)> {
    let (qw, qs) = get_weight_f16(weights, &format!("{}.q_proj.weight", prefix))?;
    let (kw, ks) = get_weight_f16(weights, &format!("{}.k_proj.weight", prefix))?;
    let (vw, vs) = get_weight_f16(weights, &format!("{}.v_proj.weight", prefix))?;
    let q_dim = qs[0]; let kv_dim = ks[0]; let hidden = qs[1];
    assert_eq!(ks[1], hidden); assert_eq!(vs[1], hidden);
    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hidden);
    fused.extend_from_slice(&qw); fused.extend_from_slice(&kw); fused.extend_from_slice(&vw);
    let total_rows = q_dim + 2 * kv_dim;
    let dev = cuda.upload_f16(&fused)?;
    Ok((GpuWeight { data: dev, rows: total_rows, cols: hidden }, q_dim, kv_dim))
}

fn load_fused_gate_up_weight(
    weights: &HashMap<String, RawTensor>, prefix: &str, cuda: &CudaState,
) -> Result<(GpuWeight, usize)> {
    let (gw, gs) = get_weight_f16(weights, &format!("{}.gate_proj.weight", prefix))?;
    let (uw, us) = get_weight_f16(weights, &format!("{}.up_proj.weight", prefix))?;
    let intermediate = gs[0]; let hidden = gs[1];
    assert_eq!(us[0], intermediate); assert_eq!(us[1], hidden);
    let mut fused = Vec::with_capacity(2 * intermediate * hidden);
    fused.extend_from_slice(&gw); fused.extend_from_slice(&uw);
    let dev = cuda.upload_f16(&fused)?;
    Ok((GpuWeight { data: dev, rows: 2 * intermediate, cols: hidden }, intermediate))
}

// ═══════════════════════════════════════════════════════════════════════
//  Decoder Layer (GPU-resident)
// ═══════════════════════════════════════════════════════════════════════

struct GpuDecoderLayer {
    iln_w: CudaSlice<f16>,
    pln_w: CudaSlice<f16>,
    qn_w: CudaSlice<f16>,
    kn_w: CudaSlice<f16>,
    qkv_w: GpuWeight,
    o_w: GpuWeight,
    gu_w: GpuWeight,
    dp_w: GpuWeight,
    nqh: usize, nkvh: usize, hd: usize, hs: usize, eps: f32,
}

impl GpuDecoderLayer {
    fn load(w: &HashMap<String, RawTensor>, p: &str, cfg: &TextConfig, cuda: &CudaState) -> Result<Self> {
        Ok(Self {
            iln_w: load_gpu_vec(cuda, w, &format!("{}.input_layernorm.weight", p))?,
            pln_w: load_gpu_vec(cuda, w, &format!("{}.post_attention_layernorm.weight", p))?,
            qn_w: load_gpu_vec(cuda, w, &format!("{}.self_attn.q_norm.weight", p))?,
            kn_w: load_gpu_vec(cuda, w, &format!("{}.self_attn.k_norm.weight", p))?,
            qkv_w: load_fused_qkv_weight(w, &format!("{}.self_attn", p), cuda)?.0,
            o_w: load_gpu_weight(cuda, w, &format!("{}.self_attn.o_proj.weight", p))?,
            gu_w: load_fused_gate_up_weight(w, &format!("{}.mlp", p), cuda)?.0,
            dp_w: load_gpu_weight(cuda, w, &format!("{}.mlp.down_proj.weight", p))?,
            nqh: cfg.num_attention_heads,
            nkvh: cfg.num_key_value_heads,
            hd: cfg.head_dim,
            hs: cfg.hidden_size,
            eps: cfg.rms_norm_eps as f32,
        })
    }

    /// x: [b, s, hs]  (always GPU, consumed — reused as the residual stream so we
    /// don't allocate+memcpy a clone before the O-proj accumulation).
    /// cos/sin: [s, d]  device-resident slice for the positions of this call
    fn forward(&self, x: GpuTensor, cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
               kv: &mut GpuKvCache, layer_idx: usize, kv_start: usize, use_causal: bool,
               cuda: &CudaState) -> Result<GpuTensor>
    {
        let b = x.shape()[0]; let s = x.shape()[1];
        let sub_profile = std::env::var_os("QFA_SUB_PROFILE").is_some();
        let mut t_rmsn = std::time::Duration::ZERO;
        let mut t_qkv = std::time::Duration::ZERO;
        let mut t_qk = std::time::Duration::ZERO;
        let t_softmax = std::time::Duration::ZERO;
        let mut t_av = std::time::Duration::ZERO;
        let mut t_o = std::time::Duration::ZERO;
        let mut t_mlp = std::time::Duration::ZERO;
        let t0 = std::time::Instant::now();

        // 1. Input RMSNorm
        let normed = cuda.rms_norm(&x, &self.iln_w, self.eps)?;
        if sub_profile { cuda.synchronize().ok(); t_rmsn = t0.elapsed(); }
        let t1 = std::time::Instant::now();

        // 2. Fused QKV projection
        let qkv = cuda.linear_gpu(&normed, &self.qkv_w)?;
        if sub_profile { cuda.synchronize().ok(); t_qkv = t1.elapsed(); }
        let t2 = std::time::Instant::now();
        let q_dim = self.nqh * self.hd;
        let kv_dim = self.nkvh * self.hd;

        // 3+4. Fused QKV: extract Q (norm+rotary), K (norm+rotary→cache), V (raw→cache).
        // One launch replaces qkv_extract_q_norm_rotary + qkv_extract_kv_norm_rotary_cache.
        let q = cuda.qkv_extract_qkv_norm_rotary_cache(
            &mut kv.k[layer_idx], &mut kv.v[layer_idx],
            &qkv, &self.qn_w, &self.kn_w, cos, sin,
            self.nqh, self.nkvh, self.hd, q_dim, kv_dim,
            kv.max_seq, kv_start, kv_start, self.eps,
        )?;
        drop(qkv);
        let cur_len = kv_start + s;

        // 6+7. Attention.  Three paths:
        //   - Prefill (s > 1): grouped GQA — loops over nkvh KV-head groups,
        //     stride_a=0 batched GEMM for QK and AV (no repeat_kv needed).
        //   - Decode short (s == 1, cur_len ≤ 1024): single-block fused_gqa_decode.
        //   - Decode long  (s == 1, cur_len > 1024): split-K fused_gqa_decode.
        let attn_out = if s == 1 {
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            const SPLIT_THRESHOLD: usize = 1024;
            // Adaptive chunk: smaller chunks = more parallelism but more merge overhead.
            // Empirically (P104, sm_61): 256 sweet spot until ~2048, then 512 wins on long ctx.
            let chunk_size: usize = if cur_len >= 2048 { 512 } else { 256 };
            if cur_len > SPLIT_THRESHOLD {
                cuda.fused_gqa_decode_split(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                    self.nkvh, kv.max_seq, cur_len, scale, chunk_size)?
                    .reshape(vec![b, self.nqh, 1, self.hd])
            } else {
                cuda.fused_gqa_decode(&q, &kv.k[layer_idx], &kv.v[layer_idx],
                    self.nkvh, kv.max_seq, cur_len, scale)?
            }
        } else {
            let scale = 1.0f32 / (self.hd as f32).sqrt();
            let av = cuda.grouped_gqa_prefill(
                &q, &kv.k[layer_idx], &kv.v[layer_idx],
                self.nkvh, kv.max_seq, scale, use_causal && s > 1,
            )?;
            if sub_profile {
                cuda.synchronize().ok();
                // grouped_gqa_prefill fuses QK + softmax + AV; report as qk.
                t_qk = t2.elapsed();
            }
            av
        };
        if sub_profile && s == 1 {
            // decode path: fused_gqa_decode — record total attention time as t_av.
            cuda.synchronize().ok();
            t_av = t2.elapsed();
        }

        // 8. Reshape [b, h, s, d] → [b, s, h*d], then O projection with residual add (beta=1).
        //    For decode (s == 1), swap_dims_12 is a no-op (both layouts collapse to [b, h*d]),
        //    so we can skip the kernel and just reshape.
        let attn_flat = if s == 1 {
            attn_out.reshape(vec![b, 1, self.nqh * self.hd])
        } else {
            cuda.swap_dims_12(&attn_out)?.reshape(vec![b, s, self.nqh * self.hd])
        };
        // h = x.clone() then h += attn_flat @ O^T   (cuBLAS beta=1)
        let mut h = x;
        let t5 = std::time::Instant::now();
        cuda.linear_gpu_accum(&mut h, &attn_flat, &self.o_w)?;
        drop(attn_flat);
        if sub_profile { cuda.synchronize().ok(); t_o = t5.elapsed(); }

        // 10. Post-attention RMSNorm
        let normed2 = cuda.rms_norm(&h, &self.pln_w, self.eps)?;

        // 11. Fused gate-up → SiLU·up
        let t6 = std::time::Instant::now();
        let gu = cuda.linear_gpu(&normed2, &self.gu_w)?;
        let activated = cuda.silu_mul_split(&gu)?;
        drop(gu);
        // 12. Down projection with residual add (beta=1) — fused, no separate add_inplace.
        cuda.linear_gpu_accum(&mut h, &activated, &self.dp_w)?;
        if sub_profile { cuda.synchronize().ok(); t_mlp = t6.elapsed(); }

        if sub_profile {
            eprintln!("    rmsn={:.1} qkv={:.1} qk={:.1} softmax={:.1} av={:.1} o={:.1} mlp={:.1} ms",
                t_rmsn.as_secs_f64()*1000.0, t_qkv.as_secs_f64()*1000.0,
                t_qk.as_secs_f64()*1000.0, t_softmax.as_secs_f64()*1000.0,
                t_av.as_secs_f64()*1000.0, t_o.as_secs_f64()*1000.0,
                t_mlp.as_secs_f64()*1000.0);
        }

        let _ = self.hs;  // silence
        Ok(h)
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Text Decoder
// ═══════════════════════════════════════════════════════════════════════

pub(crate) struct GpuTextDecoder {
    pub embed_table: GpuWeight,        // [vocab, hidden]
    pub lm_head: GpuWeight,            // [classify_num, hidden]  (independent, NOT shared with embed)
    layers: Vec<GpuDecoderLayer>,
    pub norm_w: CudaSlice<f16>,            // [hidden]
    pub eps: f32,
    pub cuda: Arc<CudaState>,
}

impl GpuTextDecoder {
    pub fn load_with(cuda: Arc<CudaState>, weights: &HashMap<String, RawTensor>, prefix: &str, config: &TextConfig) -> Result<Self> {
        let (embed_f16, embed_shape) = get_weight_f16(weights, &format!("{}.embed_tokens.weight", prefix))?;
        let embed_dev = cuda.upload_f16(&embed_f16)?;
        let embed_table = GpuWeight { data: embed_dev, rows: embed_shape[0], cols: embed_shape[1] };

        // lm_head is INDEPENDENT for aligner (thinker.lm_head.weight is a separate tensor,
        // not shared with embed_tokens). Prefix is "thinker.lm_head.weight" not "thinker.model.lm_head.weight".
        let lm_head = load_gpu_weight(&cuda, weights, "thinker.lm_head.weight")?;

        let norm_w = load_gpu_vec(&cuda, weights, &format!("{}.norm.weight", prefix))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(GpuDecoderLayer::load(weights, &format!("{}.layers.{}", prefix, i), config, &cuda)?);
        }

        Ok(Self { embed_table, lm_head, layers, norm_w, eps: config.rms_norm_eps as f32, cuda })
    }

    /// Forward pass.
    /// hs: [1, sl, hidden] (GPU). cos/sin: [sl, head_dim] (GPU).
    /// kv_start: how many positions are already in the cache (0 for prefill).
    /// Returns logits as a GpuTensor of shape [1, out_sl, vocab] (out_sl = 1 if llo).
    pub fn forward(&self, hs: GpuTensor, cos: &CudaSlice<f16>, sin: &CudaSlice<f16>,
                   kv: &mut GpuKvCache, kv_start: usize, use_causal: bool, llo: bool) -> Result<GpuTensor>
    {
        let sl = hs.shape()[1];
        let mut h = hs;
        let layer_profile = std::env::var_os("QFA_LAYER_PROFILE").is_some();
        let mut layer_times: Vec<std::time::Duration> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let t0 = std::time::Instant::now();
            h = layer.forward(h, cos, sin, kv, i, kv_start, use_causal, &self.cuda)?;
            if layer_profile {
                self.cuda.synchronize()?;
                let dt = t0.elapsed();
                layer_times.push(dt);
            }
        }
        if layer_profile {
            for (i, dt) in layer_times.iter().enumerate() {
                eprintln!("  [layer {i:2}] {:.3} ms", dt.as_secs_f64() * 1000.0);
            }
            let total_ms: f64 = layer_times.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
            eprintln!("  [28 layers total] {total_ms:.1} ms");
        }
        kv.cur_len = kv_start + sl;

        // Final RMSNorm
        let h = self.cuda.rms_norm(&h, &self.norm_w, self.eps)?;

        // Low-latency optimization: slice last token for prefill
        let h = if llo && sl > 1 {
            self.slice_last_token(&h)?
        } else {
            h
        };

        // LM head — INDEPENDENT weight (thinker.lm_head.weight), not shared with embed_tokens
        self.cuda.linear_gpu(&h, &self.lm_head)
    }

    fn slice_last_token(&self, h: &GpuTensor) -> Result<GpuTensor> {
        let s = h.shape();
        assert_eq!(s.len(), 3);
        let (b, sl, hidden) = (s[0], s[1], s[2]);
        // For [1, sl, hidden] the last token's contiguous row sits at offset (sl-1)*hidden.
        // We allocate a fresh buffer and ask the stream to copy device→device.
        let mut out = self.cuda.alloc_uninit_f16(b * hidden)?;
        let src_offset = (sl - 1) * hidden;
        let src_view = h.data.slice(src_offset..src_offset + b * hidden);
        self.cuda.stream.memcpy_dtod(&src_view, &mut out)?;
        Ok(GpuTensor::new(out, vec![b, 1, hidden]))
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  MRoPE cos/sin precompute (CPU side, then upload to GPU)
// ═══════════════════════════════════════════════════════════════════════

pub(crate) fn compute_mrope_cos_sin(
    pos: &[Vec<i64>; 3], hd: usize, rt: f64, ms: &[usize], il: bool,
) -> (CpuTensor, CpuTensor) {
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
    let cos = CpuTensor::new(cv.iter().map(|&v| f16::from_f32(v)).collect(), vec![sl, hd]);
    let sin = CpuTensor::new(sv.iter().map(|&v| f16::from_f32(v)).collect(), vec![sl, hd]);
    (cos, sin)
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
    }
    m
}
