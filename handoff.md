# Qwen3-ForcedAligner — Handoff 文档（cudarc 自研引擎版本）

> **接力说明**：这份文档面向接手 RTFx 优化工作的下一个 AI/工程师。先看 §1-§4 了解当下状态，然后从 §5 接手。

---

## 1. 项目概况

| 项目 | 路径 | 推理框架 | 用途 |
|------|------|---------|------|
| 原始 candle 版 | `D:\qwen-aligner` | candle 0.10 | Qwen3-ForcedAligner 词级时间戳对齐 |
| ASR 参考项目 | `D:\qwen3-asr-burn` | **cudarc + 手写 kernel**（burn 是 fallback） | 同架构 ASR 模型，是当前项目的优化范本 |
| **本项目** | `D:\qwen-aligner-rs` | **cudarc 手写 CUDA + gemm+rayon 手写 CPU** | aligner 重构；CUDA / CPU 双后端，零 DL 框架 |

**目标**：把 RTFx 推到极致，正确性与 candle 版逐字段一致。

---

## 2. 当前状态（commit `9e337b2` 之后）

### 2.1 性能（P104-100, Pascal sm_61, 8GB, **无 tensor core**）

3 语言基准（best of 3 稳态运行，去掉模型加载 ~5s）：

| Fixture | 时长 | 词数 | 总耗时 | RTFx | vs candle 加速比 |
|---|---|---|---|---|---|
| `en_180s` | 180s | 597 | **4.44s** | **40.6** | **2.04x** (candle 9.05s) |
| `zh_203s` | 203s | 889 | **5.28s** | **38.4** | **2.36x** (candle 12.45s) |
| `ja_89s`  | 89s  | 253 | **1.47s** | **60.8** | **1.91x** (candle 2.80s) |
| `ko_4m`   | 267s | 189 | 5.42s   | 49.2  | (无 candle baseline) |

15s 冒烟测试: `tests/fixtures/15s.wav` (40 词)，耗时 0.22s，RTFx 67。

### 2.2 正确性

3 语言 × candle baseline 逐字段对比：

| Fixture | 词数 mismatches | 平均时间戳差 | 最大时间戳差 |
|---|---|---|---|
| en | 0 / 597 | 0.67ms | 400ms（单 token 边界偏移，candle 自身也有） |
| zh | 0 / 889 | 0.18ms | 80ms |
| ja | 0 / 253 | **0ms** | **0ms**（bit-exact） |

15s 冒烟：与 `bench_outputs/smoke_en.json` **bit-exact 一致**。

### 2.3 编译状态

- `cargo build --release` → **0 warning、0 error**
- 依赖只有 cudarc + 通用 crate（无 burn/candle/cubecl/wgpu）
- `.cargo\config.toml` 配 USTC 镜像，需要 `NO_PROXY=*` 绕过本地代理

---

## 3. 架构概览（必读）

### 3.1 推理链路

```
AlignerInference::align()
  ↓
  ├─ Mel 提取 (CPU, rustfft)
  ├─ 文本 tokenize (CPU, tokenizers + lindera)
  ↓
  └─ cudarc_engine.rs + kernels/kernels.cu  ← 真正的 CUDA 推理
       │
       ├─ GpuAudioEncoder (gpu_audio_encoder.rs)
       │    conv stem (im2col + cuBLAS) → 24 层 transformer → proj
       │
       └─ GpuTextDecoder (cudarc_engine.rs)
            embed_lookup → MRoPE → 28 层 GQA decoder → norm → lm_head → argmax
```

### 3.2 文件清单（清理后）

```
src/
├── lib.rs                  (22 行)  - 模块声明
├── main.rs                 (63 行)  - CLI
├── batch.rs                (76 行)  - JSONL 批处理
├── config.rs               (139 行) - AlignerConfig 反序列化
├── audio_io.rs             (105 行) - WAV 加载 + 重采样
├── text_io.rs              (13 行)  - 文本读取
├── mel.rs                  (175 行) - Log-Mel 特征
├── text.rs                 (238 行) - 多语言分词
├── tokenizer.rs            (121 行) - BPE
├── prompt.rs               (46 行)  - audio pad 展开
├── timestamp.rs            (170 行) - LIS 时间戳修复
├── inference.rs            (400 行) - 主推理流水线（cudarc-only）
├── cudarc_engine.rs        (1424 行) - ★★★ 核心：CUDA 引擎 + 28 层 decoder + 所有 op wrapper
├── gpu_audio_encoder.rs    (356 行) - 24 层音频编码器
└── kernels/kernels.cu      (1265 行) - ★★★ 核心：所有 fused CUDA kernel
```

### 3.3 关键设计点

1. **零深度学习框架** —— 直接调 cuBLAS HGEMM + NVRTC 编译的手写 kernel。`cudarc` 只是 CUDA driver API 的 Rust 绑定（类似 PyCUDA 之于 Python）。
2. **全部 f16** —— 权重加载时一次性 `f32 → f16`，之后所有计算 f16，f32 仅用于 reduction 累加（RMS、softmax）。
3. **KV cache 预分配** —— `GpuKvCache::new(max_seq = seq_len + 64)`，一次性分配 28 × 8 × 128 × max_seq × 2 × 2 bytes，整个推理过程不再分配 KV 显存。
4. **正确性核心点**：
   - `argmax_rows` 用 `F16_TIMESTAMP_ARGMAX_TIE_EPS = 1/256` 做 f16 tie-breaking，保留最小 index（与 candle 一致）
   - softmax/RMS 在 f32 累加再转 f16
   - audio embedding 的 f16→f32→f16 round-trip 保留（与 candle 数值兼容）

---

## 4. 已完成的优化历史

| Commit | 改动 | 收益 |
|---|---|---|
| `8e130a9` | burn → cudarc + 手写 kernel 完全重写 | RTFx 0.47 → 33-57（70-120x） |
| `d80ef34` | softmax 改为 online 单 pass（Flash-Attention v1 风格） | text_decoder -10%，softmax 63ms → 55ms |
| `9e337b2` | 清理 burn 死代码 + 依赖 | 性能持平，warning 40→0，代码 6951 删 / 1561 改 |

---

## 5. 接下来的 RTFx 优化方向（按优先级排序）

### 5.1 当前 text_decoder per-layer profile

跑 `QFA_PROFILE=1 QFA_SUB_PROFILE=1 ./target/release/qwen-aligner.exe align ...` 看到的每层耗时（EN 180s，seq_len=4567）：

| Op | per-layer | × 28 layers | 占 text_decoder 比例 |
|---|---|---|---|
| `rmsn` (RMSNorm) | 1.2ms | 34ms | 0.7% |
| `qkv` (fused QKV linear) | 13.7ms | 384ms | 7.5% |
| **`qk` (Q@K^T matmul)** | **54ms** | **1.51s** | **30%** |
| **`softmax` (online causal)** | **55ms** | **1.54s** | **30%** |
| `av` (AV matmul) | 21.5ms | 602ms | 12% |
| `o` (O proj + residual fuse) | 11ms | 308ms | 6% |
| `mlp` (gate-up + silu_mul + down) | 27.4ms | 767ms | 15% |
| **总计** | **184ms** | **5.15s** | — |

实测 text_decoder 总耗时 ~5.07s，与上面 5.15s 一致（差额是 embed_lookup + 最终 norm + lm_head）。

### 5.2 Task #1: Flash-Attention 融合 kernel（最大单点收益）⭐⭐⭐

**目标**：把 `attention_qk + softmax_scaled_causal + attention_av` 三个步骤融成单个 kernel，**消除 666MB 的 `scores` 中间张量**。

**当前代价**：
- 每层分配 `scores [1, 16, 4567, 4567] f16 = 666MB`，一次写一次读 = 1.33GB HBM 流量
- 28 层 × (1.51s qk + 1.54s softmax + 0.6s av) = 总共 **~3.65s**

**Flash-Attention 思路**：
- 把 Q 切成 `[B_r=64, hd=128]` 的 tile（沿 seq 轴）
- 把 K/V 切成 `[B_c=64, hd=128]` 的 tile（沿 kv-seq 轴）
- 每个 `(b, h, q_tile)` block 在 shared memory 里维护 `(m, l, o)` 三元组（max, sum, output），逐 K-tile 滚动更新（online softmax）
- 完全不分配 scores 中间张量

**实施细节**：

文件：`src/kernels/kernels.cu` 新增 `fused_prefill_attn_f16` kernel
```cuda
// 参考 Flash-Attention v1 paper Algorithm 1
// 输入: Q [b, h, m, d], K [b, h, n, d], V [b, h, n, d], scale, causal
// 输出: O [b, h, m, d]
// Grid:  (b * h, ceil(m / Br), 1)
// Block: (128 threads, 1, 1)  // 每 thread 处理 128/Br 个 q rows
extern "C" __global__ void fused_prefill_attn_f16(
    __half* __restrict__ O,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    int b, int h, int m, int n, int d,
    float scale, int is_causal
) {
    constexpr int Br = 64;   // q-tile
    constexpr int Bc = 64;   // kv-tile
    // shared mem: q_tile[Br][d] + k_tile[Bc][d] + v_tile[Bc][d] + s_tile[Br][Bc]
    // + m_state[Br] + l_state[Br] = ~ (64*128 + 64*128 + 64*128 + 64*64 + 64 + 64) * 2bytes ≈ 56KB
    // ↑ Pascal sm_61 shared mem per block = 48KB（够紧，可能需要 Br=Bc=32）

    // 算法:
    //   for j in 0..ceil(n/Bc):
    //     load K_j, V_j into smem
    //     S_ij = Q_i @ K_j^T * scale          (Br x Bc)
    //     apply causal mask (if is_causal)
    //     m_new = max(m_old, rowmax(S_ij))
    //     P_ij = exp(S_ij - m_new)
    //     l_new = exp(m_old - m_new) * l_old + rowsum(P_ij)
    //     O_i = (l_old * exp(m_old - m_new) / l_new) * O_i + (1 / l_new) * P_ij @ V_j
    //     m_old = m_new; l_old = l_new
}
```

文件：`src/cudarc_engine.rs` 新增 wrapper
```rust
pub fn fused_prefill_attn(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor,
                         scale: f32, causal: bool) -> Result<GpuTensor> {
    let s = q.shape();
    let (b, h, m, d) = (s[0], s[1], s[2], s[3]);
    let n = k.shape()[2];
    let mut out = self.alloc_uninit_f16(b * h * m * d)?;
    // grid = (b*h, ceil_div(m, Br), 1), block = (128, 1, 1)
    // shared mem计算...
    // launch...
    Ok(GpuTensor::new(out, vec![b, h, m, d]))
}
```

修改 `GpuDecoderLayer::forward`（cudarc_engine.rs 大约 1255-1282 行）：
```rust
// 替换:
let scores = cuda.attention_qk(&q, &k_rep)?;
let attn = cuda.softmax_scaled_causal(&scores, scale, use_causal && s > 1)?;
drop(scores);
let av = cuda.attention_av(&attn, &v_rep)?;
// 改为:
let av = cuda.fused_prefill_attn(&q, &k_rep, &v_rep, scale, use_causal && s > 1)?;
```

**Pascal sm_61 关键约束**：
- Shared mem per block: 48KB（必须严格 fit）
- 无 tensor core：matmul 用 FMA 手写（不是 `mma.sync`）
- Warp size 32，block 用 128 thread (4 warps) 比较稳

**预期收益**：3.65s → 1.5-2s（节省 1.5-2s）；EN RTFx **40 → 60-70**

**风险**：
- Pascal 数值精度跟 cuBLAS 不完全一致 → 可能引入 1-2 个边界 token 偏移
- shared mem 紧张，Br/Bc 取值需要实验
- 复杂度高，~250 行 CUDA C++

**验证清单**：
1. 15s smoke test 必须通过（与 `bench_outputs/smoke_en.json` 对比，允许 1-2 个 token 偏移）
2. 3 语言跑过，词数 mismatch = 0
3. JA 89s 必须 bit-exact 或 ≤ 1 个 token 偏移
4. 记录新的 per-layer profile

### 5.3 Task #2: 减少 cudaMalloc 开销 ⭐⭐

**问题**：每层 prefill 都 `alloc_uninit_f16(b * h * m * n)` 分配 scores buffer（666MB）+ alloc av output（19MB）。28 层 × 2 次 alloc = 56 次 cudaMalloc。

**注意**：之前尝试过 scratch buffer 复用（commit history 里有），**会破坏正确性**（前 3 个词的 start_time = 0.0），rolled back 了。原因可能是 cudarc 的内部 pool 在并发请求下的 aliasing 问题。

**两条可行路径**：
1. **配合 Flash-Attention（Task #1）做完后，scores buffer 自然消失**——这是首选
2. 直接预分配 1 个 `attn_out` scratch buffer（19MB × 1，所有层共享），不动 scores —— 比较安全

**预期收益**：50-100ms（小，但 free if Task #1 done）

### 5.4 Task #3: cuBLAS GEMM 算法搜索 ⭐⭐

`linear_gpu` / `attention_qk` / `attention_av` 走的是 `cublasGemmStridedBatchedEx` 的默认算法。Pascal 上 cuBLAS 的算法选择是次优的。

**实施**：
```rust
// src/cudarc_engine.rs::CudaState
// 在 new_with_ctx 里加一个 warm-up 阶段，对每个常用 (m, n, k) 形状跑
// cublasGemmEx 的 7 种算法（CUBLAS_GEMM_ALGO0..6），选最快的存到 HashMap
```

cudarc 0.19 应该已经暴露了 `gemm_ex` 接口（带 algo 参数）。

**预期收益**：5-15%（每个 matmul 1-3ms × 28 层 × 4 matmul = 100-300ms）

**风险**：cudarc API 可能不直接支持 algo 选择，需要 fork 或 raw FFI

### 5.5 Task #4: MLP 融合（gate_up + silu + mul + down）⭐

**当前**：
```
gu = linear(x, gate_up_weight)         // 1 GEMM, output [b, s, 2*inter]
activated = silu_mul_split(gu)         // 1 kernel, output [b, s, inter]
linear_gpu_accum(h, activated, down)   // 1 GEMM (累加到 residual)
```
27.4ms / layer × 28 = 767ms

**优化思路**：silu_mul_split + down_proj 之间的中间张量 `activated` 是 `[1, 4567, 1536] f16 = 14MB`，可以避免存储——但融合 GEMM + activation 在 Pascal 上需要写 mma-style kernel，复杂度类似 Flash-Attention。

**预期收益**：5-10%

**优先级**：Task #1 之后再看，可能不必要

### 5.6 Task #5: Audio encoder Flash-Attention 化 ⭐

audio encoder 24 层 self-attention 用的是 full attention（aligner 不做 window），目前直接 `attention_qk + softmax + attention_av`。同 text decoder 一样的问题，可以应用同一个 fused kernel。

**当前耗时**：1.45s（EN 180s）

**预期收益**：1.45s → 0.5-0.8s

---

## 6. 关键已知问题 & 陷阱

### 6.1 子 profile 时间归属（已修复但要小心）

`cudarc_engine.rs::GpuDecoderLayer::forward` 里的 `QFA_SUB_PROFILE=1` 输出是**正确的**（已修复）：
- `qk` = 只测 `attention_qk` 的时间
- `softmax` = 只测 `softmax_scaled_causal` 的时间
- `av` = 只测 `attention_av` 的时间

**修改这段代码时不要把它们的 timer 起点搞混**（之前犯过这个错，softmax 显示 91ms 实际是 qk+softmax 合计）。

### 6.2 数值噪声敏感的 token

**JA 89s 的 `Ů` token**（idx 82）是个零持续时间 token (`start = end`)，f16 sub-ULP 差异就会导致 burn vs candle 不一致。

- 当前 softmax 用 `bs=1024` (block size for reduction) → JA bit-exact 0ms
- 改成 `bs=512` 或 `bs=256` → JA 出现 1-4 个 token 偏移（其他语言不受影响）
- **不要为了 5% 的速度去改 bs**（用户明确要求与 candle 一致）

### 6.3 Flash-Attention 的 causal mask 边界

`is_causal=true` 时，row i 只能看到 col ≤ i 的位置。但对于 row 0 来说 `valid_n=1`，整个 K-tile 几乎都被 mask 掉。Online softmax 必须处理 `valid_n=0` 的极端情况（虽然实际不会发生，因为 row ≥ 0）。

### 6.4 cudarc 的内存池

`alloc_uninit_f16` 实际命中 cudarc 的 stream-local memory pool，**释放不是立刻发生的**。`drop(scores)` 后 buffer 在 stream 完成下一次 sync 之前还活着。这是 scratch buffer 复用方案失败的可能原因——并发请求复用了同一块物理 buffer。

---

## 7. 运行方式

### 7.1 编译

```powershell
$env:HTTPS_PROXY = ""; $env:HTTP_PROXY = ""; $env:NO_PROXY = "*"
cargo build --release
```

### 7.2 单文件对齐

```powershell
# 关键环境变量
$env:QFA_PROFILE = "1"        # 各阶段耗时
$env:QFA_SUB_PROFILE = "1"    # text_decoder 每层 per-op 耗时

.\target\release\qwen-aligner.exe align `
  --audio tests\fixtures\15s.wav `
  --text  tests\fixtures\15s.txt `
  --model models\Qwen3-ForcedAligner-0.6B `
  --language English `
  --output result.json
```

### 7.3 3 语言 benchmark

```bash
for lang in en zh ja; do
  case $lang in en) f=en_180s; dur=180;; zh) f=zh_203s; dur=203;; ja) f=ja_89s; dur=89;; esac
  for i in 1 2 3; do
    log=$(QFA_PROFILE=1 ./target/release/qwen-aligner.exe align \
      --audio D:/qwen3-asr-burn/tests/fixtures/$f.wav \
      --text  D:/qwen3-asr-burn/tests/fixtures/$f.txt \
      --model models/Qwen3-ForcedAligner-0.6B \
      --language $lang \
      --output bench_outputs/$lang.json 2>&1)
    tot=$(echo "$log" | grep "^profile total" | sed 's/.*total=//' | sed 's/s$//')
    rtfx=$(python -c "print(f'{$dur/$tot:.1f}')")
    echo "$lang run$i: ${tot}s RTFx=$rtfx"
  done
done
```

### 7.4 正确性验证（Python）

```python
import json
for lang, dur in [('en', 180), ('zh', 203), ('ja', 89)]:
    a = json.load(open(f'bench_outputs/candle_baseline_{lang}.json', encoding='utf-8'))
    b = json.load(open(f'bench_outputs/{lang}.json', encoding='utf-8'))
    mt = sum(1 for x,y in zip(a,b) if x['text'] != y['text'])
    td = sum(abs(x['start_time']-y['start_time'])+abs(x['end_time']-y['end_time']) for x,y in zip(a,b))
    md = max(abs(x['start_time']-y['start_time'])+abs(x['end_time']-y['end_time']) for x,y in zip(a,b))
    print(f'{lang}: words={len(b)} mism={mt}, avg={td/len(b)*1000:.2f}ms, max={md*1000:.0f}ms')
```

---

## 8. 性能上限估计

| 优化阶段 | 预期 EN 180s | RTFx | 累计收益 |
|---|---|---|---|
| 当前 | 4.44s | 40.6 | baseline |
| + Task #1 (Flash-Attention text_decoder) | 2.5-3.0s | 60-72 | -35% |
| + Task #5 (Flash-Attention audio_encoder) | 1.7-2.2s | 82-105 | -50% |
| + Task #3 (cuBLAS algo search) | 1.5-2.0s | 90-120 | -55% |

**Pascal sm_61 上的硬件上限**：~RTFx 100-150（无 tensor core，6 TFLOPs FP16 算力上限）。

要再往上只能换卡：
- RTX 3060：~RTFx 150-200（有 tensor core，25 TFLOPs）
- RTX 4090：~RTFx 500-800（165 TFLOPs）
- H100：~RTFx 2000+（1500 TFLOPs）

---

## 9. 模型架构（速查）

- **Audio Encoder**：3 层 stride-2 Conv2d stem → Sinusoidal PE → 24 层 Transformer（全注意力，16 头，d_model=1024）→ LayerNorm → proj1(GELU) → proj2
- **Text Decoder**：Embedding → 28 层 GQA Decoder（16 Q heads / 8 KV heads，RoPE theta=1M，MRoPE sections=[24,20,20] interleaved，SwiGLU MLP）→ RMSNorm → lm_head[5000, 1024]
- **推理方式**：单次前向传播（非自回归），不需要 KV Cache 自回归（但要 KV cache 存中间值）
- **classify_num**：5000（lm_head 输出维度）
- **timestamp_token_id**：151705
- **audio_token_id**：151654

权重前缀（safetensors）：
| 模块 | 前缀 |
|------|------|
| Audio Encoder | `thinker.audio_tower.*` |
| Text Decoder layers | `thinker.model.layers.*` |
| Final norm | `thinker.model.norm.*` |
| Embedding | `thinker.model.embed_tokens.*` |
| LM Head | `thinker.lm_head.*` |

---

## 10. 相关文件路径

| 文件 | 路径 |
|------|------|
| 本项目二进制 | `D:\qwen-aligner-rs\target\release\qwen-aligner.exe` |
| Candle baseline 二进制 | `D:\qwen-aligner\target\release\qwen-aligner.exe` |
| 模型目录 | `D:\qwen-aligner-rs\models\Qwen3-ForcedAligner-0.6B\` |
| 测试 fixtures（短） | `D:\qwen-aligner-rs\tests\fixtures\` (15s.wav/txt, ko_4m.wav/txt) |
| 测试 fixtures（长） | `D:\qwen3-asr-burn\tests\fixtures\` (en_180s, zh_203s, ja_89s) |
| candle 基线对齐输出 | `D:\qwen-aligner-rs\bench_outputs\candle_baseline_{en,zh,ja}.json` |
| 15s smoke 基线 | `D:\qwen-aligner-rs\bench_outputs\smoke_en.json` |

> 注：`bench_outputs/` 和 `tests/` 已在 `.gitignore` 中，本地存在但不入库。

---

## 11. Git 历史关键 commits

```
9e337b2  refactor: drop burn entirely — cudarc + hand-written kernels only
73416fa  chore: gitignore tests/ — local-only fixtures
4dedab0  chore: gitignore bench_outputs/ — local-only benchmark output
d80ef34  perf(softmax): online single-pass softmax kernel — text_decoder ~10% faster
8e130a9  feat(perf): rewrite with cudarc — 1.87x over candle, RTFx 33-57
```

---

## 12. 接手第一步建议

1. **跑一遍 §7.3 的 3 语言 benchmark + §7.4 正确性验证**，确认你的环境跟我这里数字一致（±5%）
2. **跑 `QFA_SUB_PROFILE=1` 看 per-layer profile**，确认 softmax + qk 仍然是大头
3. **从 Task #1 (Flash-Attention) 开始**——参考 `D:\qwen3-asr-burn\src\kernels\kernels.cu` 看 ASR 项目有没有现成的 fused attention kernel 可以借鉴
4. **每改一步**：
   - 15s smoke 必须 bit-exact 通过
   - 3 语言 word_mismatches = 0
   - JA 必须 bit-exact 或 ≤ 1 token 偏移
   - 记录新的 RTFx 数字到这份文档

---

## 13. CPU engine 状态（commit `4e337b2` 之后）

CUDA / CPU 双后端架构已落地。`DeviceRequest::{Cuda(n), Cpu, Auto}` 三种入口全打通。

### 13.1 已实现

| 组件 | CUDA 路径 | CPU 路径 | 文件 |
|---|---|---|---|
| 28 层 text decoder forward | ✅ 完整 | ✅ **完整** | `cudarc_engine.rs` / `cpu_engine.rs` |
| QKV fused linear + silu_mul_split | ✅ cudarc | ✅ gemm | `cudarc_engine.rs` / `cpu_engine.rs` |
| online causal softmax | ✅ 自写 kernel | ✅ rayoned 3-pass | `kernels.cu` / `cpu_engine.rs` |
| prefill attention（scores materialised） | ✅ cuBLAS batched GEMM | ✅ 手写 nested-parallel | `cudarc_engine.rs` / `cpu_engine.rs` |
| MRoPE cos/sin precompute | ✅ | ✅ | `cudarc_engine.rs` / `cpu_engine.rs` |
| lm_head（独立权重，不是 tied embed） | ✅ | ✅ linear | `cudarc_engine.rs` / `cpu_engine.rs` |
| 24 层 audio encoder + conv stem | ✅ 完整 | ❌ **stub，bail** | `gpu_audio_encoder.rs` |

CPU 路径里 m=1 lm_head GEMV 用 gemm crate（强制 `Parallelism::Rayon(0)`，避免 burn-flex 的 7M 阈值把 decode 留下单线程）。f32-only（现代 x86 缺 f16 SIMD，f32 实际比 f16-with-upcast 快）。

### 13.2 验证

| 测试 | 结果 |
|---|---|
| `cargo build --release`（default: cuda + cpu） | 0 warning、0 error |
| `cargo build --release --no-default-features --features cpu` | 0 warning、0 error |
| 15s 冒烟 `--device cuda` | bit-exact（与 candle baseline 一致） |
| `--device cpu` | text decoder 加载成功，audio encoder 步骤清晰 bail "not yet implemented" |
| `--device auto`（CUDA build） | 走 CUDA |
| `--device auto`（CPU-only build） | 走 CPU |
| `--device cuda`（CPU-only build） | 清晰 bail "CUDA backend not compiled" |

### 13.3 接入 voxtrans 的最小改动

voxtrans `asr_align.rs`：

```diff
- use qwen_forced_aligner_rs::{ AlignRequest, AudioInput, DTypeRequest, DeviceRequest, ForcedAlignItem, ForcedAlignResult,
-     ModelOptions, TextInput, load_model };
+ use qwen_forced_aligner_rs::{ AlignRequest, AudioInput, DeviceRequest, ForcedAlignItem, ForcedAlignResult,
+     ModelOptions, TextInput, load_model };
  ...
  ModelOptions { device: device.qwen_device,
-                dtype: DTypeRequest::F16 }
+               /* 删掉 dtype，cudarc 强制 f16，CPU 自己决定 */ }
```

`Cargo.toml` 改 git URL 指到 `qwen-forced-aligner-rs` repo。约 3 行 diff。

### 13.4 下一步（按优先级）

1. **Audio encoder CPU 实现**（剩余 CPU 引擎最后一块拼图）
   - conv stem 3 × stride-2 conv2d（im2col + gemm 已经在 `cpu_engine.rs` 里有 stub，但没接完整 per-chunk reshape）
   - 24 层 transformer (LayerNorm + GELU FFN + full attn)
   - 估算 600-800 行（参考 `gpu_audio_encoder.rs` 391 行）

2. **CUDA-side Flash-Attention 融合**（最大单点收益 -35%）
   - 见 §5.2
   - text_decoder 28 层 减 1.5s，EN RTFx 39→60+

3. **Flash-Attention 移植到 CPU**（同步把 §5.2 的算法搬到 CPU；f32 m=Q@K^T 比 CUDA 慢 ~50x，所以收益会小一些）

4. **libloading 改造**（实现"单一安装包"）
   - cudarc 加 `libloading::Library::new("cudart64_120.dll")` 代替编译期链接
   - 运行时探测：cudart64 找不到就退化 CPU 路径

5. **继续 RTFx 优化**（handoff §5）

---

## 14. Git 历史关键 commits

```
4e337b2  feat(cpu): CPU text decoder engine — DeviceRequest::Cpu/Auto wire-up
40b4cf2  feat(api): candle-compatible public API + cuda/cpu feature gating
ce94662  docs: rewrite handoff.md as cudarc-engine baton-pass document
9e337b2  refactor: drop burn entirely — cudarc + hand-written kernels only
73416fa  chore: gitignore tests/ — local-only fixtures
4dedab0  chore: gitignore bench_outputs/ — local-only benchmark output
d80ef34  perf(softmax): online single-pass softmax kernel — text_decoder ~10% faster
8e130a9  feat(perf): rewrite with cudarc — 1.87x over candle, RTFx 33-57
```
