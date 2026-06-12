# ROADMAP.md — AI 接手指南

> 面向 AI 助手的路线图。记录当前状态、已完成工作、下一步规划、已知问题和卡点。

## 项目简介

Qwen3-ForcedAligner 的 Rust 实现，零深度学习框架依赖。
CUDA 路径用 cudarc + 手写 CUDA kernel + cuBLAS HGEMM。
CPU 路径用 gemm crate + rayon。
被 `D:\voxtrans`（音视频转录翻译程序）作为库调用，生成单词/字符级别时间戳。

姊妹项目：`D:\qwen3-asr`（同架构 ASR 模型，已做完类似抽象）。

---

## 当前状态（commit `43ae896`）

### 性能（P104-100, Pascal sm_61, 8GB, 无 tensor core）

纯推理时间（不含 ~5s 模型加载）：

| Device | Fixture | 时长 | 推理耗时 | RTFx |
|--------|---------|------|----------|------|
| CUDA | 15s (EN) | 15s | 0.21s | ~71x |
| CUDA | 180s (EN) | 180s | 4.69s | **38.4x** |
| CUDA | ko_4m (KO) | 267s | 4.98s | **53.5x** |
| CPU | 15s (EN) | 15s | **1.48s** | **10.1x** |
| CPU | 180s (EN) | 180s | **32.6s** | **5.5x** |
| CPU | ko_4m (KO) | 267s | ~51s | ~5.2x |

> CPU 数字来自 commit `43ae896`（P0-1 f16→f32 预转换 + P1-2 v2 SIMD prefill_attention）。
> 180s text_decoder 30.5s → 19.3s (1.58x)；attn 是 180s 上 87% 的 text_decoder
> 时间，所以 fused/SIMD attention 是最大单点。ko_4m 未在此 commit 重测。

### 正确性

- CUDA: 所有 fixture 与重构前 candle baseline 完全一致（15s 40/40, 180s 909/909, ko_4m 189/189）
- CPU: 15s 40/40 与 CUDA 一致；180s/ko_4m 有 CPU 音频编码器边界处理的已知微小差异

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
| `a0e51bc` | docs: update ROADMAP with P0-1 results and CPU optimization plan |
| `f4de5cb` | feat(cpu): add QFA_SUB_PROFILE per-op timing (per-layer per-op print, near-zero cost) |
| `43ae896` | **CPU prefill_attention SIMD Q@K + A@V (P1-2 v2)** — AVX2+FMA intrinsics on the 2 hot inner loops of the per-head scalar attention. 180s text_decoder 30.5s→19.3s (1.58x), total 43.4s→32.6s (RTFx 5.5x). 15s 1.52s→1.48s. *(v1 用 gemm crate 在 K=128 上反而 3.6x 变慢，已回退)* |

---

### 1. CPU 性能优化（**进行中** — 已完成 P0-1, P1-0, P1-2 v2，下一个目标是 P2 audio encoder）

当前瓶颈（180s EN 推理 32.6s）：

| 组件 | 耗时 | 占比 | 备注 |
| text_decoder | **19.3s** | **59%** | 28 层 prefill，P1-2 v2 后 attn 从 ~26s 降到 ~15s |
| audio_encoder | 12.6s | 39% | 24 层 + conv stem (3 convs) — **P2 下一个目标** |
| prepare_input | 0.2s | 1% | 已是噪声 |

text_decoder 19.3s 中：4 个 GEMM ~3.5s，attn ~15s，elementwise ~0.8s。

已完成 / 跳过：
- ✅ P1-0 sub-profile (`f4de5cb`)：诊断出 attn 在 180s 上占 87% text_decoder
- ✅ P1-2 v1 gemm crate 替换：rejected（K=128 skinny matmul 反而 3.6x 变慢）
- ✅ P1-2 v2 AVX2+FMA SIMD (`43ae896`)：attn 26s → 15s, text_decoder 1.58x

下一步：
- **P2-1: Conv stem SIMD + 并行化**（12.6s 中的 3 个 conv + GELU + permute 1-2s，估 1.5-2x on audio_encoder）
- **P2-2: Audio encoder 跨层并行**（24 层串行，估 1.3-1.5x on audio_encoder）
- **P1-1: matrixmultiply crate 直替 gemm crate**（4 个 GEMM ~3.5s，估 1.2-1.5x on text_decoder）
- **P3-1: MLP 融合**（silu_mul+down_proj 中间张量消除，估 5-10% on text_decoder）
| 组件 | 耗时 | 占比 |
|------|------|------|
| text_decoder | 3.19s | 68% |
| audio_encoder | 1.25s | 27% |
| prepare_input | 0.19s | 4% |

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
