// GPU element-wise kernels for the Qwen3-ASR text decoder + audio encoder.
// All arithmetic accumulates in f32 but storage is f16.
// Targets sm_61+ (no requirement for tensor cores or f16 atomics).

#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ─── RMS norm: x [outer, last] * weight[last] → out, with f32 accumulation ──
// One block per row; block_size threads cooperate over `last`.
// Shared mem: block_size * sizeof(float).
extern "C" __global__ void rms_norm_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float sdata[];

    float local = 0.0f;
    for (int j = tid; j < last; j += bs) {
        float v = __half2float(x[row * last + j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();

    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    for (int j = tid; j < last; j += bs) {
        float v = __half2float(x[row * last + j]) * inv_rms * __half2float(w[j]);
        out[row * last + j] = __float2half(v);
    }
}

// ─── Fused: residual_inplace = residual + add_in; out = rms_norm(residual_inplace, w) ──
// Writes `residual_inplace` with the residual sum AND `out` with the normed result.
// Shared mem: bs * sizeof(float).
extern "C" __global__ void add_residual_rms_norm_f16(
    __half* __restrict__ residual_inplace,   // r' = r + a
    __half* __restrict__ out,                // out = rms_norm(r', w)
    const __half* __restrict__ add_in,
    const __half* __restrict__ w,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float sdata[];

    // Pass 1: residual sum + sum of squares
    float local = 0.0f;
    for (int j = tid; j < last; j += bs) {
        float r = __half2float(residual_inplace[row * last + j]);
        float a = __half2float(add_in[row * last + j]);
        float v = r + a;
        residual_inplace[row * last + j] = __float2half(v);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)last + eps);

    // Pass 2: normalize
    for (int j = tid; j < last; j += bs) {
        float v = __half2float(residual_inplace[row * last + j]) * inv_rms * __half2float(w[j]);
        out[row * last + j] = __float2half(v);
    }
}

// ─── Element-wise add: out = a + b ─────────────────────────────────
extern "C" __global__ void add_f16(
    __half* __restrict__ out,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── In-place add: a += b ──────────────────────────────────────────
extern "C" __global__ void add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

// ─── SiLU(gate) * up (SwiGLU activation) ───────────────────────────
extern "C" __global__ void silu_mul_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2half(g * sig * __half2float(up[i]));
}

// ─── Fused gate-up split + SiLU(gate)*up ──────────────────────────
// gu: [outer, 2*inter] in row-major (gate first half of last dim, up second half).
// Writes activated [outer, inter].
extern "C" __global__ void silu_mul_split_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gu,
    int outer,
    int inter
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer * inter) return;
    int o = i / inter;
    int c = i % inter;
    int base = o * 2 * inter;
    float g = __half2float(gu[base + c]);
    float u = __half2float(gu[base + inter + c]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2half(g * sig * u);
}

// ─── Softmax with scale + optional causal mask ─────────────────────
// x shape (logical [bh, m, n]) launched as grid = (bh * m,). One block per row.
// is_causal != 0 keeps positions [0..row_in_m+1) only.
extern "C" __global__ void softmax_scaled_causal_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int m,
    int n,
    float scale,
    int is_causal
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int m_idx = row % m;
    int valid_n = (is_causal != 0) ? (m_idx + 1) : n;

    extern __shared__ float sdata[];

    // Max
    float local_max = -INFINITY;
    for (int j = tid; j < valid_n; j += bs) {
        float v = __half2float(x[row * n + j]) * scale;
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Sum
    float local_sum = 0.0f;
    for (int j = tid; j < valid_n; j += bs) {
        local_sum += expf(__half2float(x[row * n + j]) * scale - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int j = tid; j < n; j += bs) {
        if (j < valid_n) {
            float v = __half2float(x[row * n + j]) * scale - row_max;
            out[row * n + j] = __float2half(expf(v) * inv_sum);
        } else {
            out[row * n + j] = __float2half(0.0f);
        }
    }
}

// ─── Online (single-pass max+sum) softmax with causal mask ────────────
// Fuses the max and sum reductions into one pass via running rescale.
// Caller MUST zero the output buffer before launching (we skip writing
// masked-tail positions so AV-matmul reads valid zeros there).
// Shared memory layout: [bs floats max][bs floats sum] = bs*8 bytes.
extern "C" __global__ void softmax_scaled_causal_online_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int m,
    int n,
    float scale,
    int is_causal
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int m_idx = row % m;
    int valid_n = (is_causal != 0) ? (m_idx + 1) : n;

    extern __shared__ float sdata[];
    float* smax = sdata;        // [bs]
    float* ssum = sdata + bs;   // [bs]

    // Single-pass online softmax statistics.
    float my_max = -INFINITY;
    float my_sum = 0.0f;
    for (int j = tid; j < valid_n; j += bs) {
        float v = __half2float(x[row * n + j]) * scale;
        if (v > my_max) {
            my_sum = my_sum * __expf(my_max - v) + 1.0f;
            my_max = v;
        } else {
            my_sum += __expf(v - my_max);
        }
    }
    smax[tid] = my_max;
    ssum[tid] = my_sum;
    __syncthreads();

    // Tree reduction over (max, sum) pairs.
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float m1 = smax[tid],     s1 = ssum[tid];
            float m2 = smax[tid + s], s2 = ssum[tid + s];
            float mm = fmaxf(m1, m2);
            float ss = (mm == -INFINITY)
                ? 0.0f
                : s1 * __expf(m1 - mm) + s2 * __expf(m2 - mm);
            smax[tid] = mm;
            ssum[tid] = ss;
        }
        __syncthreads();
    }
    float row_max = smax[0];
    float row_sum = ssum[0];
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Write only valid positions; masked tail is left at the pre-zeroed value.
    for (int j = tid; j < valid_n; j += bs) {
        float v = __half2float(x[row * n + j]) * scale - row_max;
        out[row * n + j] = __float2half(__expf(v) * inv_sum);
    }
    // Zero the masked tail in a separate loop so the hot path has no branch.
    for (int j = valid_n + tid; j < n; j += bs) {
        out[row * n + j] = __float2half(0.0f);
    }
}

// ─── Rotary embedding ──────────────────────────────────────────────
// x [b, h, s, d], cos/sin [total_s, d] (broadcast over b, h, indexed at pos_offset+is).
// rotate_half: for i<half → -x[i+half], for i>=half → x[i-half].
extern "C" __global__ void rotary_emb_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int h, int s, int d, int pos_offset
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * s * d;
    if (tot >= total) return;

    int id = tot % d;
    int is = (tot / d) % s;
    int half_ = d >> 1;

    float x_val = __half2float(x[tot]);
    float pair_val = (id < half_)
        ? -__half2float(x[tot + half_])
        :  __half2float(x[tot - half_]);

    int row = pos_offset + is;
    float c = __half2float(cos[row * d + id]);
    float si = __half2float(sin[row * d + id]);
    out[tot] = __float2half(x_val * c + pair_val * si);
}

// ─── Fused RMSNorm + rotary embedding (Q/K path) ───────────────────
// Per-row norm over head_dim, then rotary on the same row.
// x [b, h, s, d], weight [d], cos/sin [total_s, d] indexed at pos_offset+is.
// Grid = (b*h*s,); one block per (head, seq) row.
extern "C" __global__ void rms_norm_rotary_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int h, int s, int d, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*h*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * h * s;
    if (row >= total_rows) return;
    int is = row % s;
    int cs_row = pos_offset + is;

    extern __shared__ float sdata[];

    // Compute sum of squares
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(x[row * d + j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    // x_normed[j] computed in registers per j; need x_normed at pj for rotary.
    // Cheap workaround: load x[pj], scale by inv_rms * w[pj].
    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float x_val_j  = __half2float(x[row * d + j])  * inv_rms * __half2float(w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float x_pair   = __half2float(x[row * d + pj]) * inv_rms * __half2float(w[pj]);
        float pair_val = (j < half_) ? -x_pair : x_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        out[row * d + j] = __float2half(x_val_j * c + pair_val * si);
    }
}

// ─── Repeat KV from sparse cache ───────────────────────────────────
// cache layout (per-layer): [b, nkvh, max_seq, d]; valid rows [0..cur_len).
// Output dst [b, nqh, cur_len, d] contiguous (cur_len-major within head).
// ─── Embedding lookup ─────────────────────────────────────────────
extern "C" __global__ void embed_lookup_f16(
    __half* __restrict__ out,
    const __half* __restrict__ table,
    const long long* __restrict__ ids,
    int n, int d
) {
    int i = blockIdx.x;
    if (i >= n) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long id = ids[i];
    for (int j = tid; j < d; j += bs) {
        out[i * d + j] = table[id * d + j];
    }
}

// ─── Single-token embed lookup, id read from GPU i32 buffer ──────
// Used by the decode hot loop to chain argmax (writes ids[slot]) → embed (reads ids[slot])
// without an htod round-trip.
extern "C" __global__ void embed_lookup_single_i32_f16(
    __half* __restrict__ out,
    const __half* __restrict__ table,
    const int* __restrict__ ids,
    int slot,
    int d
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    long long id = (long long)ids[slot];
    for (int j = tid; j < d; j += bs) {
        out[j] = table[id * (long long)d + j];
    }
}

// ─── Fused LM-head GEMV + argmax (decode-only) ─────────────────────
// hidden: [1, 1, hs]   embed_table: [vocab, hs]  (acts as lm_head weight; y = hidden @ embed_table^T)
// Computes logits = hidden @ embed_table^T (length vocab), tracks argmax inline.
// Output: a single i32 (the argmax index).
//
// Grid = (BLOCKS,) blocks; each block computes a slab of `vocab / BLOCKS` rows,
// reduces local max within the block, then writes (max, idx) to a temp [BLOCKS] pair.
// A small reduction kernel picks the global argmax.
//
// For simplicity we go single-pass: launch ONE block with grid=1 and lots of threads,
// each thread covers a stripe of vocab rows. cur GPU (P104, sm_61) can handle this fine
// because vocab~151936 / 1024 threads = ~148 dot-products per thread, each 1024 multiplies.
// Total ~152M ops per token, ~5ms on f16 — acceptable, saves alloc + launch overhead.
extern "C" __global__ void lm_head_gemv_argmax_f16(
    int* __restrict__ out_idx,
    const __half* __restrict__ hidden,      // [hs]
    const __half* __restrict__ embed_table, // [vocab, hs]   row-major
    int vocab, int hs
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;

    float local_max = -INFINITY;
    int local_idx = 0;
    for (int v = tid; v < vocab; v += bs) {
        float dot = 0.0f;
        const __half* row = embed_table + v * hs;
        for (int j = 0; j < hs; j++) {
            dot += __half2float(hidden[j]) * __half2float(row[j]);
        }
        if (dot > local_max) { local_max = dot; local_idx = v; }
    }

    __shared__ float smax[1024];
    __shared__ int sidx[1024];
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sidx[0];
}

// ─── Argmax over a vector of length n (single block) ───────────────
extern "C" __global__ void argmax_f16(
    int* __restrict__ out_idx,
    const __half* __restrict__ x,
    int n
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ float smax[1024];
    __shared__ int sidx[1024];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sidx[0];
}

// Same as argmax_f16 but writes into out_idx[slot] instead of out_idx[0].
// Lets the decode loop reuse a single i32 buffer across all steps and chain
// the result into embed_lookup_single_i32_f16 without an htod round-trip.
extern "C" __global__ void argmax_into_slot_f16(
    int* __restrict__ out_idx,
    const __half* __restrict__ x,
    int n,
    int slot
) {
    int tid = threadIdx.x;
    int bs = blockDim.x;
    __shared__ float smax[1024];
    __shared__ int sidx[1024];

    float local_max = -INFINITY;
    int local_idx = 0;
    for (int i = tid; i < n; i += bs) {
        float v = __half2float(x[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax[tid + s] > smax[tid]) {
                smax[tid] = smax[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out_idx[slot] = sidx[0];
}

// ─── Swap dims 1 and 2 of a 4-D tensor ─────────────────────────────
// src [d0, d1, d2, d3] (contig) → dst [d0, d2, d1, d3] (contig)
extern "C" __global__ void swap_dims_12_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int d0, int d1, int d2, int d3
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (tot >= total) return;
    int i3 = tot % d3;
    int i1 = (tot / d3) % d1;
    int i2 = (tot / (d3 * d1)) % d2;
    int i0 = tot / (d3 * d1 * d2);
    int src_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
    dst[tot] = src[src_idx];
}

// ─── Permute [b, c, f, t] → [b, t, c, f] (audio conv-stem post-conv layout) ─
// Replaces a download→CPU 4-loop→upload roundtrip in gpu_audio_encoder.rs.
// Output is contiguous in [b, t, c, f] order, matching the conv_out Linear's
// expected [b*t, c*f] row-major input.
extern "C" __global__ void permute_bcft_to_btcf_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int c, int f, int t
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * c * f * t;
    if (tot >= total) return;
    // dst index decomposition: dst[b][t][c][f] → (ib, it, ic, if)
    int if_ = tot % f;
    int ic  = (tot / f) % c;
    int it  = (tot / (f * c)) % t;
    int ib  = tot / (f * c * t);
    // src layout [b, c, f, t]: src[((ib*c + ic)*f + if_)*t + it]
    int src_idx = ((ib * c + ic) * f + if_) * t + it;
    dst[tot] = src[src_idx];
}

// ─── Add positional encoding with broadcast + f16-rounded sum ──────────────
// out[b, t, dm] = f16_round( f32(out[b,t,j]) + f32(pe[t,j]) )
// Matches the CPU path's f16::from_f32(f32(a)+f32(b)) exactly (the f16 add is
// done by widening both operands to f32, adding, then rounding back to f16).
// This preserves the bit-pattern that downstream f16 ops expect.
extern "C" __global__ void add_pe_broadcast_f16(
    __half* __restrict__ io,    // [b, t, dm] in-place
    const __half* __restrict__ pe,  // [t, dm]
    int b, int t, int dm
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * t * dm;
    if (tot >= total) return;
    int j  = tot % dm;
    int it = (tot / dm) % t;
    // pe index ignores b (broadcast): pe[it*dm + j]
    float a = __half2float(io[tot]);
    float pv = __half2float(pe[it * dm + j]);
    io[tot] = __float2half(a + pv);
}

// ─── Split fused QKV into a single head group ──────────────────────
// qkv [b, s, total_cols], offset selects start column, h*d contiguous columns.
// dst [b, h, s, d] (transposed layout from src).
extern "C" __global__ void qkv_split_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ qkv,
    int b, int s, int h, int d, int total_cols, int offset
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_total = b * h * s * d;
    if (tot >= dst_total) return;
    int id = tot % d;
    int is = (tot / d) % s;
    int ih = (tot / (d * s)) % h;
    int ib = tot / (d * s * h);
    int col = offset + ih * d + id;
    int src_idx = (ib * s + is) * total_cols + col;
    dst[tot] = qkv[src_idx];
}

// ─── Fused Q path: extract Q from QKV, RMSNorm per head row, apply rotary ──
// qkv [b, s, total_cols]; Q occupies columns [0..q_dim) with q_dim = nqh*d.
// Output q_out [b, nqh, s, d]; weight [d]; cos/sin [total_s, d] indexed at pos_offset+is.
// Grid = (b*nqh*s,) one block per (head, seq) row, threads cooperate over d.
extern "C" __global__ void qkv_extract_q_norm_rotary_f16(
    __half* __restrict__ q_out,
    const __half* __restrict__ qkv,
    const __half* __restrict__ w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nqh, int s, int d, int total_cols, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*nqh*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * nqh * s;
    if (row >= total_rows) return;
    int is = row % s;
    int ih = (row / s) % nqh;
    int ib = row / (s * nqh);
    int cs_row = pos_offset + is;

    // qkv row pointer: Q is at columns [ih*d .. ih*d + d) in row (ib, is).
    const __half* src = qkv + (ib * s + is) * total_cols + ih * d;

    extern __shared__ float sdata[];

    // Sum of squares (load Q values inline; we read them twice but cheap vs DRAM trip).
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(src[j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float x_val_j  = __half2float(src[j])  * inv_rms * __half2float(w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float x_pair   = __half2float(src[pj]) * inv_rms * __half2float(w[pj]);
        float pair_val = (j < half_) ? -x_pair : x_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        q_out[row * d + j] = __float2half(x_val_j * c + pair_val * si);
    }
}

// ─── Single-kernel QKV: extract Q, K, V from fused projection; norm+rotary on Q and K;
//     write Q to q_out, K to k_cache, V to v_cache.  One launch replaces two.
// Grid: x = b*s,  y = nqh + nkvh  (head slot)
// Block: d threads cooperate over one row of head_dim.
extern "C" __global__ void qkv_extract_qkv_norm_rotary_cache_f16(
    __half* __restrict__ q_out,
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ qkv,
    const __half* __restrict__ qn_w,
    const __half* __restrict__ kn_w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nqh, int nkvh, int s, int d, int total_cols,
    int q_dim, int kv_dim, int max_seq, int start, int pos_offset, float eps
) {
    int bx = blockIdx.x;
    int hy = blockIdx.y;
    int ib = bx / s;
    int is = bx - ib * s;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int cs_row = pos_offset + is;
    int half_ = d >> 1;

    extern __shared__ float sdata[];

    if (hy < nqh) {
        // Q path
        int ih = hy;
        const __half* src = qkv + (ib * s + is) * total_cols + ih * d;
        __half* dst = q_out + ((ib * nqh + ih) * s + is) * d;
        float local = 0.0f;
        for (int j = tid; j < d; j += bs) {
            float v = __half2float(src[j]);
            local += v * v;
        }
        sdata[tid] = local;
        __syncthreads();
        for (int sh = bs >> 1; sh > 0; sh >>= 1) {
            if (tid < sh) sdata[tid] += sdata[tid + sh];
            __syncthreads();
        }
        float inv_rms = rsqrtf(sdata[0] / (float)d + eps);
        for (int j = tid; j < d; j += bs) {
            float x_val_j = __half2float(src[j]) * inv_rms * __half2float(qn_w[j]);
            int pj = (j < half_) ? (j + half_) : (j - half_);
            float x_pair = __half2float(src[pj]) * inv_rms * __half2float(qn_w[pj]);
            float pair_val = (j < half_) ? -x_pair : x_pair;
            float c = __half2float(cos[cs_row * d + j]);
            float si = __half2float(sin[cs_row * d + j]);
            dst[j] = __float2half(x_val_j * c + pair_val * si);
        }
    } else {
        // K+V path
        int ih = hy - nqh;
        if (ih >= nkvh) return;
        const __half* k_src = qkv + (ib * s + is) * total_cols + q_dim + ih * d;
        const __half* v_src = qkv + (ib * s + is) * total_cols + q_dim + kv_dim + ih * d;
        int cache_idx = ((ib * nkvh + ih) * max_seq + (start + is)) * d;
        float local = 0.0f;
        for (int j = tid; j < d; j += bs) {
            float v = __half2float(k_src[j]);
            local += v * v;
        }
        sdata[tid] = local;
        __syncthreads();
        for (int sh = bs >> 1; sh > 0; sh >>= 1) {
            if (tid < sh) sdata[tid] += sdata[tid + sh];
            __syncthreads();
        }
        float inv_rms = rsqrtf(sdata[0] / (float)d + eps);
        for (int j = tid; j < d; j += bs) {
            float k_val_j = __half2float(k_src[j]) * inv_rms * __half2float(kn_w[j]);
            int pj = (j < half_) ? (j + half_) : (j - half_);
            float k_pair = __half2float(k_src[pj]) * inv_rms * __half2float(kn_w[pj]);
            float pair_val = (j < half_) ? -k_pair : k_pair;
            float c = __half2float(cos[cs_row * d + j]);
            float si = __half2float(sin[cs_row * d + j]);
            k_cache[cache_idx + j] = __float2half(k_val_j * c + pair_val * si);
            v_cache[cache_idx + j] = v_src[j];
        }
    }
}

// ─── Fused KV path: extract K (with norm+rotary) and V from QKV, write both into KV cache ──
// qkv [b, s, total_cols]; K at cols [q_dim..q_dim+kv_dim), V at [q_dim+kv_dim..q_dim+2*kv_dim).
// Writes k_cache [b, nkvh, max_seq, d] and v_cache [b, nkvh, max_seq, d] at rows [start..start+s).
// Grid = (b*nkvh*s,) one block per (kv_head, seq) row.
extern "C" __global__ void qkv_extract_kv_norm_rotary_cache_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ qkv,
    const __half* __restrict__ kn_w,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    int b, int nkvh, int s, int d, int total_cols,
    int q_dim, int kv_dim, int max_seq, int start, int pos_offset, float eps
) {
    int row = blockIdx.x;          // global row index over b*nkvh*s
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int total_rows = b * nkvh * s;
    if (row >= total_rows) return;
    int is = row % s;
    int ih = (row / s) % nkvh;
    int ib = row / (s * nkvh);
    int cs_row = pos_offset + is;

    const __half* k_src = qkv + (ib * s + is) * total_cols + q_dim + ih * d;
    const __half* v_src = qkv + (ib * s + is) * total_cols + q_dim + kv_dim + ih * d;
    int cache_idx = ((ib * nkvh + ih) * max_seq + (start + is)) * d;

    extern __shared__ float sdata[];

    // K: sum of squares (norm), then rotary + cache write
    float local = 0.0f;
    for (int j = tid; j < d; j += bs) {
        float v = __half2float(k_src[j]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int sh = bs >> 1; sh > 0; sh >>= 1) {
        if (tid < sh) sdata[tid] += sdata[tid + sh];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / (float)d + eps);

    int half_ = d >> 1;
    for (int j = tid; j < d; j += bs) {
        float k_val_j  = __half2float(k_src[j])  * inv_rms * __half2float(kn_w[j]);
        int pj = (j < half_) ? (j + half_) : (j - half_);
        float k_pair   = __half2float(k_src[pj]) * inv_rms * __half2float(kn_w[pj]);
        float pair_val = (j < half_) ? -k_pair : k_pair;
        float c = __half2float(cos[cs_row * d + j]);
        float si = __half2float(sin[cs_row * d + j]);
        k_cache[cache_idx + j] = __float2half(k_val_j * c + pair_val * si);
        // V: direct copy (no norm, no rotary).
        v_cache[cache_idx + j] = v_src[j];
    }
}

// ─── KV cache write ───────────────────────────────────────────────
// k_new [b, nkvh, s_new, d] (contig) → cache [b, nkvh, max_seq, d] at rows [start..start+s_new).
extern "C" __global__ void kv_cache_write_f16(
    __half* __restrict__ cache,
    const __half* __restrict__ k_new,
    int b, int nkvh, int max_seq, int d,
    int start, int s_new
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * nkvh * s_new * d;
    if (tot >= total) return;
    int id = tot % d;
    int isn = (tot / d) % s_new;
    int ih = (tot / (d * s_new)) % nkvh;
    int ib = tot / (d * s_new * nkvh);
    int dst_idx = ((ib * nkvh + ih) * max_seq + (start + isn)) * d + id;
    cache[dst_idx] = k_new[tot];
}

// ─── Fused KV cache write (K + V in one launch) ───────────────────
extern "C" __global__ void kv_cache_write_pair_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k_new,
    const __half* __restrict__ v_new,
    int b, int nkvh, int max_seq, int d,
    int start, int s_new
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * nkvh * s_new * d;
    if (tot >= total) return;
    int id = tot % d;
    int isn = (tot / d) % s_new;
    int ih = (tot / (d * s_new)) % nkvh;
    int ib = tot / (d * s_new * nkvh);
    int dst_idx = ((ib * nkvh + ih) * max_seq + (start + isn)) * d + id;
    k_cache[dst_idx] = k_new[tot];
    v_cache[dst_idx] = v_new[tot];
}

// ─── Element-wise GELU (audio encoder activation) ─────────────────
// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
extern "C" __global__ void gelu_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[i] = __float2half(g);
}

// ─── In-place GELU ─────────────────────────────────────────────────
extern "C" __global__ void gelu_inplace_f16(
    __half* __restrict__ x,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    x[i] = __float2half(g);
}

// ─── LayerNorm (with bias) ─────────────────────────────────────────
// x [outer, last] * weight[last] + bias[last]; mean/var per row.
extern "C" __global__ void layer_norm_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ bias,
    int last,
    int outer,
    float eps
) {
    int row = blockIdx.x;
    if (row >= outer) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    extern __shared__ float sdata[];

    // Sum and sum of squares
    float l_sum = 0.0f, l_sq = 0.0f;
    for (int j = tid; j < last; j += bs) {
        float v = __half2float(x[row * last + j]);
        l_sum += v;
        l_sq  += v * v;
    }
    sdata[tid] = l_sum;
    sdata[tid + bs] = l_sq;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]     += sdata[tid + s];
            sdata[tid + bs] += sdata[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)last;
    float var  = sdata[bs] / (float)last - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < last; j += bs) {
        float v = (__half2float(x[row * last + j]) - mean) * inv_std;
        out[row * last + j] = __float2half(v * __half2float(w[j]) + __half2float(bias[j]));
    }
}

// ─── Add bias broadcast (for Linear with bias) ─────────────────────
// x [outer, last] += bias[last]
extern "C" __global__ void add_bias_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int outer, int last
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= outer * last) return;
    int j = tot % last;
    x[tot] = __float2half(__half2float(x[tot]) + __half2float(bias[j]));
}

// ─── Fused GQA attention (decode-only: s_q = 1) ────────────────────
// Q  [b, nqh, 1, d]  (current query token)
// Kc [b, nkvh, max_seq, d]  (KV cache, valid rows [0..cur_len))
// Vc [b, nkvh, max_seq, d]
// out [b, nqh, 1, d]
// Computes scores = Q · K^T * scale → softmax → attn · V, all in one kernel.
// Each block handles one (b, q_head, head_dim) — but we collapse to one block per
// (b, q_head); threads cooperate over head_dim and (then) over cur_len.
// Uses two reduction passes (max, sum) in shared mem.
//
// Required shared mem: cur_len * sizeof(float) + d * sizeof(float).
#define MAX_SEQ_FUSED 4096
extern "C" __global__ void fused_gqa_decode_f16(
    __half* __restrict__ out,           // [b, nqh, 1, d]
    const __half* __restrict__ q,       // [b, nqh, 1, d]
    const __half* __restrict__ k_cache, // [b, nkvh, max_seq, d]
    const __half* __restrict__ v_cache, // [b, nkvh, max_seq, d]
    int b, int nqh, int nkvh, int max_seq, int d, int cur_len, float scale
) {
    int qh_global = blockIdx.x;
    int ib = qh_global / nqh;
    int qh = qh_global % nqh;
    int kh = qh / (nqh / nkvh);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    extern __shared__ float smem[];   // scores[cur_len], then partial_out[d * t_chunks]
    float* partial = smem + cur_len;

    const __half* q_row  = q       + (ib * nqh  + qh) * d;
    const __half* k_base = k_cache + (ib * nkvh + kh) * max_seq * d;
    const __half* v_base = v_cache + (ib * nkvh + kh) * max_seq * d;
    __half* out_row = out + (ib * nqh + qh) * d;

    // --- Stage 1: scores[t] = (Q · K[t]) * scale ---
    for (int t = tid; t < cur_len; t += bs) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) {
            dot += __half2float(q_row[j]) * __half2float(k_base[t * d + j]);
        }
        smem[t] = dot * scale;
    }
    __syncthreads();

    // --- Stage 2: row max (reduction over cur_len) ---
    float local_max = -INFINITY;
    for (int t = tid; t < cur_len; t += bs) {
        if (smem[t] > local_max) local_max = smem[t];
    }
    __shared__ float reduce_buf[1024];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float row_max = reduce_buf[0];

    // --- Stage 3: smem[t] = exp(smem[t] - row_max); sum-reduce ---
    float local_sum = 0.0f;
    for (int t = tid; t < cur_len; t += bs) {
        float v = expf(smem[t] - row_max);
        smem[t] = v;
        local_sum += v;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / reduce_buf[0];

    // --- Stage 4: out[j] = sum_t attn[t] * V[t, j]
    // Reuse the original t-split layout: tid = t_idx * d + j_idx (2 t-chunks for bs=256, d=128).
    // Each thread sums a t-stride for one j, then we reduce in shared mem.
    int t_chunks = bs / d;
    if (t_chunks < 1) t_chunks = 1;
    int j_idx = tid % d;
    int t_idx = tid / d;
    if (j_idx < d && t_idx < t_chunks) {
        float acc = 0.0f;
        for (int t = t_idx; t < cur_len; t += t_chunks) {
            acc += smem[t] * __half2float(v_base[t * d + j_idx]);
        }
        partial[t_idx * d + j_idx] = acc;
    }
    __syncthreads();
    if (tid < d) {
        float acc = 0.0f;
        for (int ti = 0; ti < t_chunks; ti++) {
            acc += partial[ti * d + tid];
        }
        out_row[tid] = __float2half(acc * inv_sum);
    }
}

// ─── Split-K fused GQA decode (for long context) ───────────────────
// 2-kernel flash-attention style.  Used when cur_len is large enough that single-block
// fused_gqa_decode_f16 becomes the bottleneck.
//
// Phase 1: each block handles one (b, nqh, t_chunk) — computes partial scores for its
// chunk, then partial numerator (sum of exp(scores) * V) and partial max + sum_exp.
// Writes to per-chunk buffers in global memory.
//
// Phase 2: each block handles one (b, nqh) — reads N chunks of (max, sum, partial_out),
// merges with online-softmax correction, writes final out.

extern "C" __global__ void fused_gqa_decode_split_p1_f16(
    float* __restrict__ part_out,       // [b, nqh, NCHUNKS, d]   (float for accumulation)
    float* __restrict__ part_max,       // [b, nqh, NCHUNKS]
    float* __restrict__ part_sum,       // [b, nqh, NCHUNKS]
    const __half* __restrict__ q,       // [b, nqh, d]
    const __half* __restrict__ k_cache, // [b, nkvh, max_seq, d]
    const __half* __restrict__ v_cache, // [b, nkvh, max_seq, d]
    int b, int nqh, int nkvh, int max_seq, int d,
    int cur_len, int chunk_size, int n_chunks, float scale
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;        // chunk index in [0, n_chunks)
    int ib = bx / nqh;
    int qh = bx % nqh;
    int kh = qh / (nqh / nkvh);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    int t_start = by * chunk_size;
    int t_end = min(t_start + chunk_size, cur_len);
    if (t_start >= cur_len) {
        if (tid == 0) {
            part_max[(ib * nqh + qh) * n_chunks + by] = -INFINITY;
            part_sum[(ib * nqh + qh) * n_chunks + by] = 0.0f;
        }
        if (tid < d) part_out[((ib * nqh + qh) * n_chunks + by) * d + tid] = 0.0f;
        return;
    }
    int chunk_len = t_end - t_start;

    extern __shared__ float smem[];   // scores[chunk_size]

    const __half* q_row  = q       + (ib * nqh  + qh) * d;
    const __half* k_base = k_cache + (ib * nkvh + kh) * max_seq * d + t_start * d;
    const __half* v_base = v_cache + (ib * nkvh + kh) * max_seq * d + t_start * d;

    // Stage 1: scores
    for (int t = tid; t < chunk_len; t += bs) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) {
            dot += __half2float(q_row[j]) * __half2float(k_base[t * d + j]);
        }
        smem[t] = dot * scale;
    }
    __syncthreads();

    // Stage 2: chunk max
    float local_max = -INFINITY;
    for (int t = tid; t < chunk_len; t += bs) {
        if (smem[t] > local_max) local_max = smem[t];
    }
    __shared__ float reduce_buf[1024];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float chunk_max = reduce_buf[0];

    // Stage 3: exp + sum
    float local_sum = 0.0f;
    for (int t = tid; t < chunk_len; t += bs) {
        float v = expf(smem[t] - chunk_max);
        smem[t] = v;
        local_sum += v;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float chunk_sum = reduce_buf[0];

    // Stage 4: partial out (NOT normalized — we let phase 2 do that with merged sum)
    // Use t-chunks layout for d=128, bs=256 → t_chunks=2.
    int t_split = bs / d;
    if (t_split < 1) t_split = 1;
    int j_idx = tid % d;
    int t_idx = tid / d;
    // Shared partial buffer at smem[chunk_size..].
    float* partial = smem + chunk_size;
    if (j_idx < d && t_idx < t_split) {
        float acc = 0.0f;
        for (int t = t_idx; t < chunk_len; t += t_split) {
            acc += smem[t] * __half2float(v_base[t * d + j_idx]);
        }
        partial[t_idx * d + j_idx] = acc;
    }
    __syncthreads();

    int meta_idx = (ib * nqh + qh) * n_chunks + by;
    if (tid == 0) {
        part_max[meta_idx] = chunk_max;
        part_sum[meta_idx] = chunk_sum;
    }
    if (tid < d) {
        float acc = 0.0f;
        for (int ti = 0; ti < t_split; ti++) {
            acc += partial[ti * d + tid];
        }
        part_out[meta_idx * d + tid] = acc;
    }
}

// Phase 2: merge chunks via online softmax correction.
// Grid = (b * nqh,), block_dim = d.
extern "C" __global__ void fused_gqa_decode_split_p2_f16(
    __half* __restrict__ out,           // [b, nqh, d]
    const float* __restrict__ part_out, // [b, nqh, n_chunks, d]
    const float* __restrict__ part_max, // [b, nqh, n_chunks]
    const float* __restrict__ part_sum, // [b, nqh, n_chunks]
    int b, int nqh, int n_chunks, int d
) {
    int qh_global = blockIdx.x;
    int ib = qh_global / nqh;
    int qh = qh_global % nqh;
    int tid = threadIdx.x;

    // Find global max across chunks (one thread reads all maxes since n_chunks is small).
    const float* maxes = part_max + (ib * nqh + qh) * n_chunks;
    const float* sums  = part_sum + (ib * nqh + qh) * n_chunks;
    float g_max = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        if (maxes[c] > g_max) g_max = maxes[c];
    }
    // Renormalized total sum.
    float g_sum = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        if (maxes[c] > -INFINITY) {
            g_sum += sums[c] * expf(maxes[c] - g_max);
        }
    }
    float inv_g_sum = 1.0f / g_sum;

    // Accumulate per-chunk partial outputs scaled by exp(chunk_max - g_max) * inv_g_sum.
    if (tid < d) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            if (maxes[c] > -INFINITY) {
                float w = expf(maxes[c] - g_max) * inv_g_sum;
                acc += w * part_out[((ib * nqh + qh) * n_chunks + c) * d + tid];
            }
        }
        out[(ib * nqh + qh) * d + tid] = __float2half(acc);
    }
}

// ─── Slice along dim 2 of [b, h, s, d] ─────────────────────────────
// out [b, h, len, d] = src[..., start:start+len, ...]
extern "C" __global__ void slice_dim2_f16(
    __half* __restrict__ out,
    const __half* __restrict__ src,
    int b, int h, int s, int d, int start, int len
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * len * d;
    if (tot >= total) return;
    int id = tot % d;
    int isl = (tot / d) % len;
    int ih = (tot / (d * len)) % h;
    int ib = tot / (d * len * h);
    int src_idx = ((ib * h + ih) * s + (start + isl)) * d + id;
    out[tot] = src[src_idx];
}

// ─── Concat along dim 2: write a chunk [b, h, len, d] into dst [b, h, s, d] at offset ──
extern "C" __global__ void concat_dim2_write_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    int b, int h, int s, int d, int dst_offset, int len
) {
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * h * len * d;
    if (tot >= total) return;
    int id = tot % d;
    int isl = (tot / d) % len;
    int ih = (tot / (d * len)) % h;
    int ib = tot / (d * len * h);
    int dst_idx = ((ib * h + ih) * s + (dst_offset + isl)) * d + id;
    dst[dst_idx] = src[tot];
}

// ─── im2col for conv2d with stride=2, pad=1, kernel=3×3 ────────────
// input  [b, c_in, h, w]
// out    [b * h_out * w_out, c_in * 3 * 3]   row-major
//   where h_out = (h + 2 - 3) / 2 + 1, w_out similarly
// One thread per output element across the full unfolded matrix.
extern "C" __global__ void im2col_3x3_s2p1_f16(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    int b, int c_in, int h, int w, int h_out, int w_out
) {
    int total = b * h_out * w_out * c_in * 9;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    // tot layout: [b * h_out * w_out] [c_in * 9]
    int kk = tot % 9;
    int ci = (tot / 9) % c_in;
    int spatial = tot / (9 * c_in);
    int iw = spatial % w_out;
    int ih = (spatial / w_out) % h_out;
    int ib = spatial / (w_out * h_out);

    int ky = kk / 3;
    int kx = kk % 3;
    int in_y = ih * 2 + ky - 1;
    int in_x = iw * 2 + kx - 1;
    __half v;
    if (in_y < 0 || in_y >= h || in_x < 0 || in_x >= w) {
        v = __float2half(0.0f);
    } else {
        int in_idx = ((ib * c_in + ci) * h + in_y) * w + in_x;
        v = in[in_idx];
    }
    out[tot] = v;
}

// ─── Fused GELU + bias add + reshape from GEMM output to [b, c_out, h_out, w_out] ──
// gemm_out [b * h_out * w_out, c_out] (row-major, no bias) → conv_out [b, c_out, h_out, w_out]
// with bias[c_out] added and GELU applied.
extern "C" __global__ void conv_postprocess_f16(
    __half* __restrict__ out,
    const __half* __restrict__ gemm_out,
    const __half* __restrict__ bias,
    int b, int c_out, int h_out, int w_out
) {
    int total = b * c_out * h_out * w_out;
    int tot = blockIdx.x * blockDim.x + threadIdx.x;
    if (tot >= total) return;
    int iw = tot % w_out;
    int ih = (tot / w_out) % h_out;
    int co = (tot / (w_out * h_out)) % c_out;
    int ib = tot / (w_out * h_out * c_out);
    int gemm_idx = ((ib * h_out + ih) * w_out + iw) * c_out + co;
    float v = __half2float(gemm_out[gemm_idx]) + __half2float(bias[co]);
    // GELU
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    out[tot] = __float2half(g);
}


// ─── Scatter audio features into embed table at specified token positions ───
// Used in aligner (Step 4) to replace audio_token positions in inputs_embeds
// with audio encoder outputs, without a CPU round trip.
//
// embeds  [seq_len, hidden] — to be modified in place
// audio   [n_audio, hidden]   — source rows
// positions [n_audio] i32     — destination row indices in embeds
// Each block handles one audio row, threads cooperate over hidden.
extern "C" __global__ void scatter_audio_rows_f16(
    __half* __restrict__ embeds,
    const __half* __restrict__ audio,
    const int* __restrict__ positions,
    int n_audio, int hidden
) {
    int row = blockIdx.x;
    if (row >= n_audio) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int dst = positions[row];
    const __half* src = audio + row * hidden;
    __half* dst_ptr = embeds + dst * hidden;
    for (int j = tid; j < hidden; j += bs) {
        dst_ptr[j] = src[j];
    }
}


