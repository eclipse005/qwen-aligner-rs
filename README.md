# qwen-aligner-rs

[Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的 Rust 实现（官方项目见 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)）。为 ASR 转录文本生成单词/字符级时间戳，支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

基于 Qwen 开源的 Forced Aligner 架构与权重，将音频与转录文本对齐，输出每个单词/字符的精确起止时间。

## 分词器

不同语言走不同的分词路径，目标是与上游 Python `qwen_asr` 逐 token 对齐：

| 语言 | 实现 | 与 Python 一致性 |
|------|------|------------------|
| 日语 | [`nagisa-rs`](https://github.com/eclipse005/nagisa-rs)（纯 Rust BiLSTM-CRF，对应 Python `nagisa`） | 逐 token 一致 |
| 韩语 | 内置 LR 分词（对应 Python `soynlp.LTokenizer`） | 行为对齐 |
| 中/英/混合 | 空格 + CJK 单字切分 | 行为对齐 |

日语路径需要 nagisa 模型权重。模型文件随 Qwen 模型目录分发，放在 `<model_dir>/nagisa/`，包含 7 个文件（约 25 MB）：

```
<model_dir>/nagisa/
├── hp.json
├── weights.safetensors
├── uni2id.json
├── bi2id.json
├── word2id.json
├── pos2id.json
└── word2postags.json
```

加载时若未找到 `<model_dir>/nagisa/`，会记一条 `info` 日志，日语对齐不可用，但其它语言不受影响。模型文件本身随整个 `models/` 目录在运行时分发，不进 git。

## 安装

```toml
[dependencies]
qwen-forced-aligner-rs = { git = "https://github.com/eclipse005/qwen-aligner-rs.git" }
```

CPU-only 构建：

```toml
qwen-forced-aligner-rs = { git = "https://github.com/eclipse005/qwen-aligner-rs.git", default-features = false, features = ["cpu"] }
```

## 音频采样率建议

**最佳实践：输入 16 kHz 单声道 WAV。** 这是 Qwen3-ForcedAligner 模型的原生采样率，16 kHz 输入不经过重采样，与上游 Python `qwen_asr` 的输出完全一致。

其他采样率的处理：

| 输入采样率 | 重采样质量 | 与 Python 一致性 |
|-----------|-----------|-----------------|
| **16 kHz** | 无需重采样 | ✅ 完全一致 |
| 32 / 48 kHz | 整数比抽取（polyphase Kaiser sinc） | ✅ 高精度 |
| 24 kHz | 有理比抽取（L=2/M=3，Kaiser sinc） | ✅ 高精度 |
| 44.1 / 22.05 / 96 kHz | Kaiser sinc 插值 | ⚠️ 近似（通带内一致，过渡带有微小差异） |

如果源音频是 44.1 kHz 或其他非标准采样率，建议先用 ffmpeg 预转码到 16 kHz：

```bash
ffmpeg -i input.flac -ar 16000 -ac 1 -c:a pcm_f32le output.wav
```

这样能保证对齐效果与原版 Python 完全一致。

## 使用

### 作为库

```rust
use qwen_forced_aligner_rs::{Aligner, DeviceRequest};

let aligner = Aligner::load("path/to/model", DeviceRequest::Best)?;
let result = aligner.align("audio.wav", "transcript text")?;
for word in &result.words {
    println!("{:.3}-{:.3} {}", word.start, word.end, word.text);
}
```

### 命令行

```bash
cargo run --release -- --model path/to/model --audio audio.wav --text "transcript" --device cuda
```

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端，需要 CUDA 12.8+ |
| `cpu`（默认） | CPU 后端 |

## 性能

测试环境：NVIDIA P104-100 8 GiB，CUDA 12.8，模型 `models/Qwen3-ForcedAligner-0.6B`。

与上游 Python `qwen_asr` 强制对齐器对比（典型 15 s ~ 4 min 音频片段）：

| 时长 | tokens | Rust 时间 | Python 时间 | Rust 显存 | Python 显存 | 时间戳偏差 |
|------|--------|-----------|-------------|-----------|-------------|------------|
| 15 s | 40 | 4.7 s | 14.9 s | ~2.0 GB | ~2.1 GB | 16 ms |
| ~90 s | 200 | 6.2 s | 15.8 s | ~3.1 GB | ~2.9 GB | 16 ms |
| ~3 min | 600 | 10.9 s | 21.8 s | ~4.6 GB | ~5.8 GB | <1 ms |
| ~3 min | 900 | 11.8 s | 22.9 s | ~5.1 GB | ~6.8 GB | <1 ms |
| ~4 min | 190 | 11.5 s | 22.0 s | ~5.3 GB | ~6.6 GB | <1 ms |

说明：

- 测试数据为本地私有音频/文本片段，不在仓库中。
- Python 使用 `torch.bfloat16`，Rust 使用 f16 CUDA kernel + f32 累加；两者 token 数完全一致，时间戳偏差在 16 ms 以内。
- Python 版在单进程内连续跑多个长音频时显存会累积，8 GiB 显卡上后续任务会 OOM；Rust 版不存在此问题。
- Rust 版当前仅支持 WAV 输入，MP3 或 raw f32le 需先用 ffmpeg 等工具转成 WAV。

## 模型下载

从 HuggingFace 下载 safetensors 格式权重（版权归原作者）：

- [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

官方项目：

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（含 Forced Aligner 说明与 Python 推理）

## 致谢 / 原版出处

本仓库是 **独立的 Rust 推理实现**，用于加载并运行官方 Qwen3-ForcedAligner 权重；**不是** Alibaba / Qwen 官方发行版，与原作者无隶属关系。

| 组件 | 原版 | 链接 | 协议（以官方页面为准） |
|------|------|------|------------------------|
| 模型权重 | Qwen3-ForcedAligner-0.6B | [Hugging Face](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache-2.0 |
| 官方推理与文档 | Qwen3-ASR 仓库 | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 |
| 日语分词参考 | Python `nagisa`（本仓库通过 [nagisa-rs](https://github.com/eclipse005/nagisa-rs) 对齐） | — | 见各自上游 |

使用模型权重时请遵守原作者许可证；本仓库的 Rust 推理代码以本仓库 License 为准。

## License

MIT
