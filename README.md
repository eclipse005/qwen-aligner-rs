# qwen-aligner-rs

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 强制对齐器（Forced Aligner）的 Rust 实现。为 ASR 转录文本生成单词/字符级时间戳，支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

基于 Qwen3-ASR 模型架构，将音频与转录文本对齐，输出每个单词/字符的精确起止时间。

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

## License

MIT
