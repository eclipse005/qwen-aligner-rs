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

纯推理时间（不含 ~5s 模型加载），median of 10 runs on each fixture:

| Device | Fixture | 时长 | 推理耗时 | RTFx |
|--------|---------|------|----------|------|
| CPU | 15s (EN) | 15s | **~0.97s** | **~15.5x** |
| CPU | 180s (EN) | 180s | **~24.9s** | **~7.2x** |
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
├── error.rs                — AlignerError (scaffolding, 未接入公开 API)
├── inference.rs            — Engine enum + PreparedInput 共享 pipeline
├── cudarc_engine.rs        — CUDA 引擎 + 28 层 text decoder + all op wrappers
├── gpu_audio_encoder.rs    — CUDA 24 层 audio encoder + conv stem
├── cpu_engine.rs           — CPU 引擎: f16→f32 预转换权重 + f32 计算 (P0-1)
```

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
</input>
### 1. CPU 性能优化（**进行中** — 越快越好，无预设上限）

当前瓶颈（180s EN 推理 ~25.8s median of 10，commit `441af06`）：

| 组件 | 耗时 | 占比 | 备注 |
| text_decoder | **~17.6s** | **~68%** | 28 层 prefill；P1-2 v2 + 4 GEMM + RMSN/elementwise。attn inner ~480-510ms/层（K reads 主导） |
| audio_encoder | **~7.0s** | **~27%** | P4-1 NHWC direct conv 后 conv_stem ~2.4-2.5s (conv2 1.7-1.8s 仍主导) + 24 音频层 (attn 1.7s + ffn 0.9s) + conv_out+PE ~0.07s |
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

当前瓶颈（180s EN 推理 4.69s）：

可能方向：

- **Conv stem GPU permute**：消除 download→permute→upload 的 CPU roundtrip（~0.05-0.1s）。注意不要用自定义 kernel 做 copy（实测 cudarc launch overhead 很大）
- **Grouped GQA 单次 GEMM**：当前 8×batch=2 小 GEMM，探索用单次 batch=16 + stride 实现
- **cuBLAS GEMM 算法搜索**：对常用形状跑 algo0-6 选最优（预估 5-15%）
- **MLP 融合**：silu_mul_split + down_proj 之间的中间张量消除

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
