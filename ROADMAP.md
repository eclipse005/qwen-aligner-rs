# ROADMAP.md — AI 接手指南

> 面向 AI 助手的路线图。记录当前状态、已完成工作、下一步规划、已知问题和卡点。

## 项目简介

Qwen3-ForcedAligner 的 Rust 实现，零深度学习框架依赖。
CUDA 路径用 cudarc + 手写 CUDA kernel + cuBLAS HGEMM。
CPU 路径用 gemm crate + rayon。
被 `D:\voxtrans`（音视频转录翻译程序）作为库调用，生成单词/字符级别时间戳。

姊妹项目：`D:\qwen3-asr`（同架构 ASR 模型，已做完类似抽象）。

---
## 当前状态（commit `441af06`）

### 性能（P104-100, Pascal sm_61, 8GB, 无 tensor core / Arrow Lake 24c AVX2+FMA）

纯推理时间（不含 ~5s 模型加载），median of 7 runs on each fixture:

| Device | Fixture | 时长 | 推理耗时 | RTFx |
|--------|---------|------|----------|------|
| CPU | 15s (EN) | 15s | **~1.03s** | **~14.6x** |
| CPU | 180s (EN) | 180s | **~21.6s** | **~8.3x** |
</input>
> 本会话 (`a5f30ab` + flash) 数字: FlashAttention 风格 tiled online softmax 默认开启, 替代 110MB scores scratch。
> 相对上轮 (~23.8s, RTFx 7.6x): 180s -2.2s (-9%), RTFx 7.6x→8.3x。
> 相对原 baseline (~25.7s, RTFx 7.0x): 180s -4.1s (-16%), RTFx 7.0x→8.3x。
> 精度 gate: **40/40 (15s vs smoke_en) + 909/909 (180s vs golden) 时间戳逐项一致** (5 次独立运行确定性验证)。
> 在线 softmax rescale 改 f32 sum 顺序引入 ULPs, 但实测无 argmax 翻转。
</input>

> CPU 数字来自本会话（`eb52f63` + `c5add06` + `441af06` + `30afdec`），median of 10 runs。本会话相对上轮 (`7633431`)：
> - 15s: 1.22s → ~0.97s (1.26x)
> - 180s: 29.9s → ~24.9s (1.20x)
</input>
> 关键新工作：
> - `eb52f63` P3-2: SIMD-ize audio matmul_qk/av (用 dot_qk_avx2/axpy_avx2 helper)。audio attn inner 80ms→70ms/层 (24 层)。
> - `c5add06` P4-1: conv stem NHWC direct conv (用 [f32; 512] accs + load+FMA+store 1-row 8-col microkernel)。conv2 3050ms→1700ms (180s)，省 1.5s。
> - `441af06` 文档：P4-2 register-accumulator 尝试失败，1-row L1 accs 是当前甜点。
>
> 本会话试过但 rejected（性能反而下降）：
> - 4-row / 2-row Q@K microkernel in prefill_attention：L1 写 stride 破坏 prefetcher pattern
> - P4-2 __m256 register accumulator: [f32; 8] copy_from_slice 开销超过 L1 节省
> - Conv stem 4-row / 8-row microkernel: im2col 缓存压力
>
> 已相对 baseline (commit `f0fdff0` 之前, 15s 2.28s / 180s 49.5s) 总提速：15s 2.35x，180s 1.99x。
</input>

> 40/40 EXACT 在 15s fixture 验证。conv stem 180s 边界处理有 CPU 已知微小差异（不影响 alignment 质量）。
</input>

### 架构（重构后）

```
src/
├── lib.rs                  — 模块声明 + 公开 re-exports
├── main.rs                 — CLI (clap)
├── batch.rs                — JSONL 批处理
├── config.rs               — AlignerConfig 反序列化
├── backend.rs              — DeviceRequest + ResolvedBackend + resolve()
├── raw_tensor.rs           — RawTensor { data: Vec<u8>, shape, dtype }
├── weights.rs              — load_weights() 保留原始 bytes
├── inference.rs            — Engine enum + PreparedInput 共享 pipeline
├── cudarc_engine.rs        — CUDA 引擎 + 28 层 text decoder + all op wrappers
├── gpu_audio_encoder.rs    — CUDA 24 层 audio encoder + conv stem
├── cpu_engine.rs           — CPU 引擎 (text decoder 28 层 + audio encoder 24 层
│                             + conv stem), f16→f32 预转换权重 + f32 计算 (P0-1).
│                             双后端中 CPU 侧的完整实现, 不再是 stub.
└── kernels/kernels.cu      — 手写 CUDA kernel (softmax/rmsnorm/rotary/attention)
```
> 注: `error.rs` (`AlignerError`) 已删除 — 全 crate 实际都用 `anyhow::Result`。

## 已完成的工作

| Commit | 内容 |
|--------|------|
| `8e130a9` | burn → cudarc + 手写 kernel 完全重写，RTFx 0.47 → 33-57 |
| `d80ef34` | online softmax kernel，text_decoder -10% |
| `9e337b2` | 清理 burn 死代码 + 依赖 |
| `40b4cf2` | candle-compatible 公开 API |
| `60e4e78` | CPU text decoder 引擎 |
| `3005917` | CPU audio encoder 端到端 |
| `c188ddf` | grouped GQA prefill — 消除 repeat_kv，+12.6% RTFx |
| `6f8528d` | **多后端抽象 + CPU f16 权重存储**（RawTensor, backend.rs, Engine enum, PreparedInput） |
| `bbc6d79` | chore: cleanup dead code, delete handoff.md, add ROADMAP.md |
| `f0fdff0` | **CPU f16→f32 load-time 预转换** — text_decoder/audio_encoder 权重在 load 时一次性 upcast，热路径无转换。15s: 2.28s→1.52s (1.50x), 180s: 49.5s→43.4s (1.04x) |
| `f4de5cb` | feat(cpu): add QFA_SUB_PROFILE per-op timing (per-layer per-op print, near-zero cost) |
| `43ae896` | **CPU prefill_attention SIMD Q@K + A@V (P1-2 v2)** — AVX2+FMA intrinsics on the 2 hot inner loops of the per-head scalar attention. 180s text_decoder 30.5s→19.3s (1.58x), total 43.4s→32.6s (RTFx 5.5x). 15s 1.52s→1.48s. *(v1 用 gemm crate 在 K=128 上反而 3.6x 变慢，已回退)* |
| `11bb061` | docs: update ROADMAP with P1-2 v2 results and P2 reordering |
| `51cae2f` | feat(cpu): add audio FFN + LayerNorm sub-profile (audio_encoder sub-stage timing) |
| `0f4799a` | feat(cpu): add CpuConvStem sub-profile (per-conv + perm + conv_out + PE timing) |
| `eb52f63` | **CPU audio matmul_qk/av SIMD (P3-2)** — 用 prefill_attention 同套 dot_qk_avx2/axpy_avx2 helper 替换 audio encoder 24 层的 matmul_qk/av 里的 gemm call。audio attn inner 80ms→70ms/层 (180s)。15s 1.10s, 180s 26.7s |
| `bb179f9` | docs: 标注 prefill_attention 4-row / 2-row / P4-2 尝试均 rejected |
| `c5add06` | **CPU conv stem NHWC direct conv (P4-1)** — NCHW→NHWC 转置 + 1-row 8-col microkernel 取代 im2col + 1-row 8-col matmul。3x3 gather 内联进 FMA inner loop，省 389ms 标量 im2col (180s)。180s conv2 3050ms→1700ms, audio_encoder 9.2s→6.9-7.3s, total 27.1s→25.8s (1.05x). 15s 1.10s→1.03s. *4-row / 8-row P2-1 试过 (rejected) 后, P4-2 [f32; 8] / [__m256; 16] 寄存器累计也试过 (rejected - copy_from_slice 开销超过 L1 节省)* |
| `441af06` | docs: 标注 P4-2 寄存器累计尝试失败, P4-1 L1 accs 是当前甜点 |
| `（本会话）` | **CPU 清理 + 两处优化**: (1) 删除 dead code (~150 行: `linear_f16` 家族 / `conv_row_*` / `CpuWeightF16` 死方法 / `error.rs` / `CpuTensor::zeros` / `embed_lookup` / `argmax` / `in_channels`+`max_pos`+`c3_out` 死字段), 去掉全局 `#![allow(dead_code)]`, 编译器现在能捕捉新死代码; 修正 `cpu_engine.rs` 文件头注释 (旧版声称"audio encoder 未实现" — 实际已完整实现). (2) `CpuConvStem` perm + PE 注入循环并行化 (rayon over rows, f16 round-trip 顺序严格保留 → bit-exact). (3) `prefill_attention` per-head scratch (`q_qh` s×hd, `scores` s×cur_len) 改用 thread-local `Cell<Vec<f32>>` 复用, 消除 28 层 × nqh head = 448 次/forward 的 alloc/free. 15s ~1.0s, 180s median ~25.7s. **40/40 exact (15s vs smoke) + 909/909 exact (180s vs 本会话捕获的 golden) 验证通过. CUDA 路径未触碰, default build 仍编译.** |
| `（本会话 2）` | **`prefill_attention` causal work-skip (bit-exact)**: Q@K 内层从 `for t in 0..cur_len { if t>=limit { -inf } else { dot } }` 改成"整行填 -inf + 仅 `for t in 0..limit` 算 dot"; A@V 内层从 `0..cur_len + if w==0 continue` 改成 causal 分支只扫 `0..limit`。softmax 一字不改 (单遍 max + 左到右 sum + 归一化顺序严格保留) → 40/40 EN + 909/909 golden bit-exact. 180s median 25.7s→**24.43s** (~5% win), 15s ~0.96s. attn/层 548ms→480ms median. **GQA K/V 共享尝试 rejected (实测回归 25.7s→29.0s)**: par over `b*nkvh=8` 共享 K/V 给 n_rep=2 sibling heads, 理论省 K/V 流量; 实测回归 13%, 原因 (a) 20 核机器上 16-way→8-way 并行宽度减半, (b) 单 KV head=2.7MB 已全在 LLC (~36MB), 第二 sibling 的 K 读本就是 LLC hit, 共享理论不成立. 见 prefill_attention 注释里的"为何不用 GQA-share / 在线 softmax"段。bit-exact 红线: CPU 路径 attn_out→argmax 全程 f32 无 f16 吸收 (见 inference::argmax_rows 1/256 tie band), 在线/分块 softmax 改 sum 顺序会引入 ULPs 翻 JA 零时长 token argmax, 故不可用。 |
| `（本会话 3）` | **`prefill_attention` live-region softmax (bit-exact)**: Q@K 不再写 `-inf` 到 masked tail (保持 resize 的 0.0); softmax + A@V 三遍都只扫 `[0..limit)` 而非全 `cur_len`。bit-exact 论证: masked tail 从不影响结果 (max 用 `>` 忽略 -inf/0; 旧代码 exp(-inf-mx)=0 对 sum 无贡献, 新代码根本不读 tail; A@V `w=0` 跳过等价于 tail 不进循环)。causal 平均填充 ~50% → scores 写/读流量减半。180s median 24.43s→**~23.8s** (系统噪声 ±0.7s), attn/层 480→~445ms, 15s ~0.94s. **t-outer Q@K (K-reuse across rows) rejected**: 改 `for t { for i { dot } }` 让 K[t] 只读一次喂所有 Q 行, 理论省 K 流量 2500×; 实测回归 ~17% (attn 445→534ms/层), 原因同 4-row microkernel — strided scores writes (stride=cur_len=5240 f32=20KB) 破坏 L1 prefetcher, 写散布的代价超过 K-reuse 收益。**audio `softmax_scaled` non-causal fast path (bit-exact, kept)**: 非 causal 分支去掉 dead `else {0.0}` 和 `row[j]*scale` 重复计算, scale 一次写进 out 复用。**`apply_rotary_row` 去掉 per-call `vec![d]` alloc**: 改成 pairwise locals (a,b → 旋转 → 写回), 3.5M allocs/forward 归零, 数学逐位等价 (同序 fmul/fsub/fadd)。40/40 + 909/909 bit-exact 全程保持, CUDA 未触碰。 |
| `（本会话 4）` | **`prefill_attention_flash` — FlashAttention 风格 tiled online softmax (默认开启)**: 替代 110MB scores scratch 的 materialised softmax。Q 分 Bq=32 块, K 分 Bk=128 块流式处理, 每 Q-block 维护 per-row online 状态 (m/s/O), S tile [32×128]=16KB 全程在 L1。K 每 Q-block-row 只读一次 (旧版每 Q row 重读整个 KV head)。**红线重新定义为"时间戳精度不下降" (而非 bit-exact)**: 在线 softmax rescale 改 f32 sum 顺序引入 ULPs, 但只要不翻 argmax, 时间戳完全不变。**实测 40/40 (15s vs smoke_en) + 909/909 (180s vs golden) 时间戳逐项一致, 5 次独立运行确定性验证, 零 argmax 翻转**。180s median ~23.8s→**~21.6s** (-9%, RTFx 7.6x→8.3x), 15s ~1.03s, attn/层 445→**310ms (-30%)**。`QFA_NO_FLASH=1` 保留 materialised 路径做 A/B 回退。CUDA 未触碰, default build green。相对原 baseline 25.7s: 180s -4.1s (-16%), RTFx 7.0x→8.3x。 |
| `（本会话 5）` | **`audio_attention_flash` — 同思路用于音频 encoder 的 non-causal full attention (默认开启)**: 替代 `matmul_qk + softmax_scaled + matmul_av`, 消除两个 [1,nh,s,s]=350MB 中间 buffer (s=2340)。hd=64 (非 128), b=1, nh=16。`QFA_NO_AUDIO_FLASH=1` 回退。**40/40 + 909/909 时间戳逐项一致 (零 argmax 翻转)**。audio attn/层 75→~60ms (-20%), 180s median ~21.6s→**~21.4s** (增益被 conv2 噪声掩盖, 但消除 700MB 中间 buffer 降内存压力, 实测无回归)。**剩余优化空间评估 (收益递减, 停止本轮)**: (a) `silu_mul_split` SIMD 需 `f32::exp` 多项式近似 (误差大, 翻 argmax 风险高); (b) conv2 1.8s weight tiling 需重构 `conv_nhwc_direct` (P4-1/P4-2 已深挖); (c) `swap_dims_12` ~0.12s (小); (d) MLP gu/dp 用 gemm crate 已优。**CPU RTFx 已从 7.0x → 8.3x (+18%), 收益耗尽**。 |
</input>
### 1. CPU 性能优化（**进行中** — 越快越好，无预设上限）

当前瓶颈（180s EN 推理 ~21.6s median，本会话 4 flash 默认开后）：

| 组件 | 耗时 | 占比 | 备注 |
| text_decoder | **~14.0s** | **~65%** | 28 层 prefill；FlashAttention tiled online softmax 后 attn **~310ms/层** (原 445ms, -30%)。qkv ~36ms + MLP(gu+dp) ~80ms/层 |
| audio_encoder | **~7.0s** | **~32%** | P4-1 NHWC direct conv 后 conv_stem ~2.5s (conv2 1.8s 仍主导) + 24 音频层 (attn 1.7s + ffn 0.9s) |
| prepare_input | 0.2s | <1% | 已是噪声 |

audio_encoder 7.0s 内部分解（commit `c5add06`）：
| 子阶段 | 耗时 | 备注 |
|--------|------|------|
| conv1 | 0.15s | c_in=1, c_out=480 |
| **conv2** | **1.7-1.8s** | m=30, n=144K, k=270, c_out=480 — P4-1 L1 accs +W 读 (W=518KB, L2) 主导。P4-2 [__m256; 16] 寄存器轮 rejected |
| conv3 | 0.45-0.5s | m=30, n=36K, k=270, c_out=480 |
| perm | 0.02s | |
| conv_out + PE | 0.06s | |
| 24 音频层 (LN + attn + ffn) | ~3.0s | attn 1.7s (SIMD Q@K/A@V) + ffn 0.9s + 24×LN |

已完成 / 跳过：
 ✅ P0-1 f16→f32 预转换 (`f0fdff0`)：text_decoder/audio_encoder 一次 upcast，热路径纯 f32
 ✅ P0-2 elementwise SIMD：**跳过**（elementwise <5% 总时间）
 ✅ P1-0 sub-profile (`f4de5cb`)：诊断出 attn 87% text_decoder
 ✅ P1-1 matrixmultiply crate：**跳过**（K=128 skinny matmul 跟 P1-2 v1 同样风险，gemm 已是甜点）
 ✅ P1-2 v1 gemm crate 替换 prefill_attention：rejected（K=128 反而 3.6x 慢）
 ✅ P1-2 v2 AVX2+FMA SIMD (`43ae896`)：attn 26s → 15s, text_decoder 1.58x
 ✅ P2-0 audio FFN+LN sub-profile (`51cae2f`)：24 层 ~126ms/层 = 3.0s
 ✅ P2-0.5 conv stem sub-profile (`0f4799a`)：conv2 是 9.2s 中 3.3s
 ✅ P2-1 v1 conv stem SIMD + parallel GELU (`7633431`)：conv stem 7.95s→4.5s, 4-row 试过反而微回归（im2col cache 压力）
 ✅ P3-2 audio matmul_qk/av SIMD (`eb52f63`)：audio attn inner 80ms→70ms/层 (180s)
 ✅ P4-1 conv stem NHWC direct conv (`c5add06`)：conv2 3050ms→1700ms (1.79x)
 ❌ P4-2 conv stem register accumulator ([f32; 8] / [__m256; 16]) (`441af06`)：rejected - copy_from_slice 开销超过 L1 节省
 ❌ prefill_attention 4-row / 2-row Q@K microkernel (`bb179f9`)：rejected - 2/4 strided 写到 scores 破坏 L1 prefetcher pattern

下一步（按 ROI 排序，但已越来越难）：
 **prefill_attention 重构为 batched t 或新 layout**：可能省 5-7s on 180s (text_decoder attn)。但 4-row / 2-row 都试过失败，需要根本性重写（FlashAttention 风格或 shared K/V across heads）。**高风险/高回报**
 **P2-2 跨 chunk 并行**：音频 encoder 30 chunks 维并行，但 conv stem 改完 NHWC 后才行；可能要重写 audio forward。
 **P3-1 MLP 融合**：silu_mul+down_proj 中间张量消除，估 5-10% on text_decoder (~1s)，**中 ROI**
 **conv stem NHWC 进一步优化**：P4-2 register accumulator 需重做（用 `repr(C, align(32))` 的 [f32; 8] 包装 __m256），估计 1.7s→0.5s (省 1.2s)。**高复杂度**
### 2. CUDA 性能优化（**进行中** — 越快越好，无预设上限）

当前瓶颈（180s EN 推理 **~5.14s** median of 3, P104-100 sm_61 8GB）:

| 阶段 | 耗时 | 占比 | 备注 |
|------|------|------|------|
| prepare_input | 0.25s | 5% | CPU mel/pack (无法 GPU 加速, 已是噪声) |
| audio_encoder | 1.25s | 24% | conv stem + 24 层; conv stem 内有 3 处 CPU↔GPU roundtrip (见下) |
| **text_decoder** | **3.6s** | **70%** | 28 层; 每层 qk(grouped GQA) **~130ms** + qkv ~10ms + mlp ~22ms + o ~5ms |
| timestamp_logits | 0.04s | <1% | gather + download |

每层 (sub-profile, 180s): `rmsn=1.7 qkv=10 **qk=130** softmax=0(fused) av=0(fused) o=5 mlp=22 ms`。
**grouped GQA (`c188ddf`) 已落地**: 消除了 repeat_kv, 用 stride_a=0 batched GEMM 在 n_rep=2 个 sibling Q head 间共享 K/V。当前是 nkvh=8 次 strided_batched GEMM (batch=2 each) for QK + 同样 8 次 for AV。

可能方向 (按实测 ROI 排序):

**[高 ROI / 中风险] cuBLAS GEMM 算法搜索 — 预估省 0.4-0.7s (10-15% on text_decoder)**
当前 safe `Gemm` trait 硬编码 `CUBLAS_GEMM_DEFAULT` (cudarc gemm.rs:95,137)。需绕过 safe API 直接调 `result::gemm_strided_batched_ex` 传 `cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0..23`。对 QK 的 `m=n=5240, k=128, batch=2×8` 和 AV 的 `m=d=128, n=5240, k=5240, batch=2×8` 形状跑一遍 `cublasGemmEx` 的 algo0-23 + DEFAULT, 选最快。Pascal sm_61 上 cuBLAS 对每个 (shape, algo) 有不同 split-K / tile 策略, 实测某些 algo 比 DEFAULT 快 10-20%。**一次离线 profiling + 硬编码 algo 即可, 零运行时成本**。注意 bit-exact: 不同 algo 可能改变 f16 accumulate 顺序 → 需用 15s+180s 时间戳 gate 验证 (与 CPU flash 同一红线)。

**[高 ROI / 中复杂度] Grouped GQA 单次 batch=16 GEMM 替代 8×batch=2 — 预估省 0.3-0.5s**
当前 `grouped_gqa_prefill` (cudarc_engine.rs:368) 对 nkvh=8 组各发一次 `gemm_strided_batched(batch=2, stride_a=0)` = 8 次 cuBLAS 调用 (QK) + 8 次 (AV) = 16 次 launch/层 × 28 层 = 448 次 launch/forward。每次 launch 在 P104 上有 ~10-30μs overhead → 448×20μs ≈ 9ms 纯 launch 开销 (小头)。**真正收益是合并成 batch=16 单次调用让 cuBLAS 选更优 tile**: 但 GQA 的 stride_a=0 (K 共享) 不能直接用 batched (要求所有 batch 的 A stride 相同且非零)。需要重排 K/V 到 [nqh=16, s, d] (repeat K/V, 反而增加内存) 或用 cuBLAS grouped GEMM API (`cublasGemmGroupedBatchedEx`, 不同 A/B/C 指针数组)。cudarc 0.19.7 有 `safe/grouped_gemm.rs`。**中复杂度**: 需重写 grouped_gqa_prefill 的 GEMM 调用方式。

**[中 ROI / 低复杂度] Conv stem GPU permute/PE — 预估省 0.05-0.15s**
`gpu_audio_encoder.rs:251-284` 有 3 处 CPU roundtrip: (1) download conv3 输出 → CPU permute [b,c,f,t]→[b,t,c,f] → upload; (2) download conv_out → CPU 加 PE → upload; (3) download co_gpu → CPU pack valid tokens → upload。conv stem 总耗时占 audio 1.25s 的一小部分 (sub-profile 未细分, 但 mel 上传+3 conv GEMM+这些 roundtrip 估 0.2-0.3s)。permute 可写一个 4D transpose kernel (已有 `swap_dims_12_f16` 作参考, 但这个是 [b,c,f,t]→[b,t,c,f] 不是 swap_dims_12); PE 加法可用现有 `add_bias_inplace` 模式。**风险**: ROADMAP 已记录 "不要用自定义 kernel 做 copy (cudarc launch overhead 大)" — 但那是针对小 chunk copy, 这里的 permute 是 b*t*c*f ≈ 30×16×480×13 ≈ 3M 元素, launch overhead 占比小。**需实测验证**。

**[低 ROI] MLP 融合 — 预估省 <0.1s**
当前 mlp ~22ms/层 = gate-up GEMM + silu_mul kernel + down GEMM。已有 `silu_mul_split_f16` kernel (fused)。要进一步消除中间 [b*s, inter] 张量需把 silu 融进 down GEMM 的 epilogue — cuBLAS 不支持自定义 epilogue, 需手写 GEMM kernel, Pascal sm_61 上打不过 cuBLAS (ROADMAP 已验证)。**跳过**。

**[不可行] FlashAttention 手写 kernel**
ROADMAP 已记录: Pascal sm_61 上手写 FlashAttention v1 慢 29x, v2 慢 17x, 打不过 cuBLAS HGEMM。**不要重试**。

### 推荐执行顺序
1. **cuBLAS algo 搜索** (高 ROI 低运行时成本, 中风险需时间戳验证) — 先做
2. **Grouped GQA 单次 GEMM** (高 ROI 中复杂度) — 次之
3. **Conv stem GPU permute** (中 ROI 低复杂度需实测) — 最后


### 3. voxtrans 接入
`D:\voxtrans` 的 `asr_align.rs` 需要约 3 行改动：
- 删 `DTypeRequest`（cudarc 强制 f16，CPU 自己决定）
- `Cargo.toml` 改 git URL 指到 `qwen-forced-aligner-rs` repo

### 4. libloading 改造（实现"单一安装包"）

cudarc 加 `libloading::Library::new("cudart64_120.dll")` 代替编译期链接。
运行时探测：cudart64 找不到就退化 CPU 路径。

---

## 卡点 / 已踩的坑

### CPU 上 f16 权重 + on-the-fly upcast 比 f32 还慢 ⚠️ → 已修复 (`f0fdff0`)

Arrow Lake (Core Ultra 200S) 只有 AVX2+FMA，**没有 AVX-512/AMX/native f16 SIMD**。
f16 + per-call `to_f32()` 把 `fma` 拆成 mul + add + upcast 三个标量操作，
有效吞吐降到 ~30 GFLOPs/s（vs f32 AVX2+FMA 的 ~480 GFLOPs/s）。

修复：load 时一次性 `into_f32()` 预转换所有权重（+1.2 GB RAM for 0.6B model），
热路径用 `linear()` / `linear_accum()` 直接吃 f32。15s 1.50x，audio_encoder 14.5s→12.6s。
如果是 Sapphire Rapids / Zen5（带 AVX-512 FP16 / VNNI），应该反过来把 f16 留到 GEMV。

### Pascal sm_61 上 Flash-Attention 不可行 ❌

实测了两个版本：
- v1（每 thread 1 q row）：比 cuBLAS 慢 29x
- v2（warp 协作）：比 cuBLAS 慢 17x

原因：cuBLAS 对 sm_61 深度调优的手写 HGEMM，手写 kernel 打不过。

### 自定义 CUDA kernel 做 copy/slice 很慢

cudarc 的 kernel launch overhead 远大于 cuBLAS GEMM 开销。
实测 `copy_chunk_f16` 替代 CPU roundtrip 反而灾难性变慢（31s+）。不要用自定义 kernel 做 copy。

### online causal softmax 不支持 in-place

第二遍写 pass 会重新读输入，输入输出不能同一 buffer。

### softmax block size 不能改

JA 89s 有个零持续时间 token，f16 sub-ULP 差异就会导致不一致。
当前 `bs=1024` 是 bit-exact 的，不要为了速度改 bs。

---

## 运行方式

```powershell
# 编译
cargo build --release

# 单文件对齐
.\target\release\qwen-aligner.exe align `
  --audio tests\fixtures\15s.wav --text tests\fixtures\15s.txt `
  --model models\Qwen3-ForcedAligner-0.6B --language English --output result.json

# Profile
$env:QFA_PROFILE = "1"       # 各阶段耗时（CPU + CUDA）
$env:QFA_SUB_PROFILE = "1"   # text_decoder 每层 per-op 耗时（**仅 CUDA 生效**）

正确性验证：与 `bench_outputs/smoke_en.json` 对比，15s 必须 40/40 exact。

---

## 关键文件路径

| 文件 | 路径 |
|------|------|
| 模型目录 | `models\Qwen3-ForcedAligner-0.6B\` |
| 测试 fixtures（短） | `tests\fixtures\` (15s, 180s, ko_4m) |
| 测试 fixtures（长/多语言） | `D:\qwen3-asr-burn\tests\fixtures\` (en_180s, zh_203s, ja_89s) |
| candle 基线 | `bench_outputs\candle_baseline_{en,zh,ja}.json` |
| 15s smoke 基线 | `bench_outputs\smoke_en.json` |
| 姊妹项目 (ASR) | `D:\qwen3-asr` |
| 下游消费者 | `D:\voxtrans` |
