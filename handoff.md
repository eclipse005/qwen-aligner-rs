# Qwen3-ForcedAligner Burn 重构 — Handoff 文档

## 1. 项目概况

| 项目 | 路径 | 框架 | 用途 |
|------|------|------|------|
| 原始 candle 版 | `D:\qwen-aligner` | candle | Qwen3-ForcedAligner 词级时间戳对齐 |
| ASR 参考项目 | `D:\qwen3-asr-burn` | burn 0.21 | 同架构 ASR 模型（已从 candle 迁移到 burn，RTFx 从 7+ 提升到 20+） |
| **本项目（burn 版）** | `D:\qwen-aligner-rs` | burn 0.21 | 将 aligner 从 candle 重构为 burn |

**目标**：用 burn 0.21 + CUDA + f16 重写 aligner，时间戳准确度与 candle 版一致，尽可能提高 RTFx。

## 2. 当前状态

### 2.1 功能正确性：已验证通过

使用 `tests\fixtures\15s.wav` + 英文文本测试，burn 版输出的 40 个词的时间戳与 candle 版 **逐字段完全一致**（与 `bench_outputs\smoke_en.json` 对比 `==`）。

### 2.2 编译状态

- `cargo build --release --features cuda` — 零 warning、零 error
- `.cargo\config.toml` 配置了 USTC 镜像源（因为 crates.io 直连不通，需 `NO_PROXY=*` 环境变量绕过本地代理）

### 2.3 性能数据

测试音频：`15s.wav`（15 秒，双声道 48kHz，aligner 内部会下混+重采样到 16kHz mono）

**Candle 版（CUDA f16）** — 二进制 `D:\qwen-aligner\target\release\qwen-aligner.exe`：
```
profile prepare_input:       stage=0.017s  total=0.017s
profile tokenize:            stage=0.001s  total=0.018s
profile audio_conv_prefix:   stage=0.067s  total=0.085s
profile audio_pos_flatten:   stage=0.001s  total=0.085s
profile audio_encoder:       stage=0.088s  total=0.173s
profile audio_projection:    stage=0.003s  total=0.176s
profile load_cpu_aux_weights:stage=0.000s  total=0.176s
profile merge_audio_features:stage=0.126s  total=0.302s
profile text_decoder:        stage=0.259s  total=0.561s
profile timestamp_logits:    stage=0.004s  total=0.565s
profile total:                             total=0.565s
Wall time: 5.5s（含模型加载约 5s）
RTFx = 15 / 0.565 ≈ 26.5
```

**Burn 版（CUDA f16）** — 二进制 `D:\qwen-aligner-rs\target\release\qwen-aligner.exe`：
```
profile prepare_input:       stage=0.021s  total=0.021s
profile tokenize:            stage=0.001s  total=0.022s
profile audio_encoder:       stage=16.6s   total=16.6s
profile merge_embeddings:    stage=0.069s  total=16.7s
profile rope_compute:        stage=0.000s  total=16.7s
profile text_decoder:        stage=14.1s   total=30.8s
profile timestamp_logits:    stage=1.27s   total=32.1s
profile total:                             total=32.1s
Wall time: ~38s（含模型加载约 6s）
RTFx = 15 / 32.1 ≈ 0.47
```

**逐阶段对比**：

| 阶段 | Candle | Burn | 倍差 |
|------|--------|------|------|
| prepare_input (Mel) | 0.017s | 0.021s | ~1x |
| tokenize | 0.001s | 0.001s | ~1x |
| audio encoder | 0.088s | 16.6s | ~189x |
| merge embeddings | 0.126s | 0.069s | ~0.5x |
| text decoder | 0.259s | 14.1s | ~54x |
| timestamp logits | 0.004s | 1.27s | ~318x |
| **总推理** | **0.565s** | **32.1s** | **~57x** |

性能差距集中在 audio encoder（24 层 Transformer）和 text decoder（28 层 GQA Transformer）的 GPU 计算上。

## 3. 项目结构

```
D:\qwen-aligner-rs\
├── .cargo\config.toml          # USTC 镜像源配置
├── Cargo.toml                  # 依赖配置
├── Cargo.lock
├── models\Qwen3-ForcedAligner-0.6B\   # 模型文件（从 D:\qwen-aligner 复制）
├── assets\korean_dict_jieba.dict      # 韩语词典
├── tests\fixtures\
│   ├── 15s.wav                 # 测试音频（15s）
│   ├── sample1.wav             # 测试音频
│   └── test_text.txt           # 测试文本
├── bench_outputs\smoke_en.json # candle 版参考输出
└── src\
    ├── lib.rs       (749B)   # Backend/Device 类型别名 + 模块声明
    ├── config.rs    (4.8KB)  # AlignerConfig 反序列化
    ├── encoder.rs   (16.6KB) # Audio Encoder: conv stem + 24 层 Transformer + projection
    ├── decoder.rs   (14.8KB) # Text Decoder: 28 层 GQA + RMSNorm + lm_head
    ├── inference.rs (14.6KB) # 主推理流水线 + 权重加载 + argmax + Profile
    ├── mel.rs       (5.3KB)  # Log-Mel 特征提取（aligner 原版实现）
    ├── main.rs      (2.9KB)  # CLI（Align + Batch 子命令）
    ├── batch.rs     (2.6KB)  # JSONL 批处理
    ├── audio_io.rs  (3.8KB)  # WAV 加载（hound + 下混 + 线性重采样）
    ├── text_io.rs   (383B)   # 文本文件读取
    ├── tokenizer.rs (4.0KB)  # BPE tokenizer
    ├── text.rs      (7.0KB)  # 多语言分词（英/中/日/韩）
    ├── prompt.rs    (1.6KB)  # audio pad 展开
    └── timestamp.rs (4.8KB)  # LIS 时间戳修复算法
```

## 4. 架构关键信息

### 4.1 模型架构（Qwen3-ForcedAligner-0.6B）

- **Audio Encoder**：3 层 stride-2 Conv2d stem → Sinusoidal PE → 24 层 Transformer（全注意力，16 头，d_model=1024）→ LayerNorm → proj1(GELU) → proj2
- **Text Decoder**：Embedding → 28 层 GQA Decoder（16 Q heads / 8 KV heads，RoPE theta=1M，MRoPE sections=[24,20,20] interleaved，SwiGLU MLP）→ RMSNorm → lm_head[5000, 1024]
- **推理方式**：单次前向传播（非自回归），不需要 KV Cache
- **classify_num**：5000（lm_head 输出维度）
- **timestamp_token_id**：151705

### 4.2 推理流程（inference.rs align_waveform_text）

1. `encode_timestamp(text, language)` → (words, aligner_input)
2. `extract_log_mel_features(waveform)` → LogMelFeatures
3. `feature_extract_output_len` → audio_pad_count
4. `expand_audio_pad_once` → 展开 `<|video_pad|>`
5. `encode_to_ids` + 记录 timestamp_positions
6. mel → `audio_encoder.forward()` → audio_embeds
7. Embedding 合并（embed_tokens 查表 + audio 位置替换为 audio_embeds，f16 截断）
8. `compute_mrope_cos_sin`（3 维位置相同）
9. `text_decoder.forward_hidden` → hidden_states
10. `extract_timestamp_logits` → timestamp 位置 → RMSNorm → lm_head
11. `argmax_rows`（f16 tie-breaking: `F16_TIMESTAMP_ARGMAX_TIE_EPS = 1.0/256.0`）
12. `timestamp_ids_to_run` → ForcedAlignResult

### 4.3 Burn 相关类型

```rust
// lib.rs
pub type Backend = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, half::f16, i32, u8>;
pub type Device = burn::backend::cuda::CudaDevice;
```

### 4.4 已做的优化

- encoder.rs: `FusedAudioQkv` — 将 Q/K/V 三个独立 matmul 合并为 1 个（24 层省 48 次 kernel launch）
- decoder.rs: 已使用 `FusedQkv`（合并 QKV）和 `FusedGateUp`（合并 gate/up）
- decoder.rs: `extract_timestamp_logits` 已改用 GPU-side `Tensor::select` 代替 GPU→CPU→GPU 往返

### 4.5 精度关键点

- Mel 特征提取必须用 aligner 原版 `mel.rs`（与 ASR 的 hann_window/reflect_pad/帧数计算有差异）
- `lm_head` 权重独立（`thinker.lm_head.weight` [5000, 1024]，不与 embed_tokens 共享）
- Audio feature 替换时 `to_f16_f32` 精度截断
- Final RMSNorm 中 f32 cast 计算 RMS
- Argmax tie-breaking 保留最小 index

## 5. 运行方式

```powershell
# 编译
$env:HTTPS_PROXY = ""; $env:HTTP_PROXY = ""; $env:NO_PROXY = "*"
cargo build --release --features cuda

# 单文件对齐
$env:QFA_PROFILE = "1"   # 可选，输出各阶段耗时
.\target\release\qwen-aligner.exe align --audio <WAV> --text <TXT> --model models\Qwen3-ForcedAligner-0.6B --language en --output result.json

# 批处理
.\target\release\qwen-aligner.exe batch --manifest <JSONL> --model models\Qwen3-ForcedAligner-0.6B --output-dir <DIR>
```

## 6. 权重前缀

| 模块 | safetensors 前缀 |
|------|-----------------|
| Audio Encoder | `thinker.audio_tower.*` |
| Text Decoder layers | `thinker.model.layers.*` |
| Text Decoder final norm | `thinker.model.norm.*` |
| Embedding | `thinker.model.embed_tokens.*` |
| LM Head | `thinker.lm_head.*` |

## 7. 网络环境

- crates.io 直连不通（通过 127.0.0.1 代理失败）
- 需设置 `$env:NO_PROXY = "*"` 绕过本地代理
- `.cargo\config.toml` 已配置 USTC 镜像源

## 8. 相关文件路径

| 文件 | 路径 |
|------|------|
| Burn 版二进制 | `D:\qwen-aligner-rs\target\release\qwen-aligner.exe` |
| Candle 版二进制 | `D:\qwen-aligner\target\release\qwen-aligner.exe` |
| 测试音频 | `D:\qwen-aligner-rs\tests\fixtures\15s.wav` |
| 测试文本 | `D:\qwen-aligner-rs\tests\fixtures\test_text.txt` |
| Candle 参考输出 | `D:\qwen-aligner-rs\bench_outputs\smoke_en.json` |
| Burn 测试输出 | `D:\qwen-aligner-rs\test_output.json` |
| Candle 测试输出 | `D:\qwen-aligner-rs\candle_output.json` |
| 模型目录 | `D:\qwen-aligner-rs\models\Qwen3-ForcedAligner-0.6B\` |
