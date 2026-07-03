# qwen-aligner-rs

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 强制对齐器（Forced Aligner）的 Rust 实现。为 ASR 转录文本生成单词/字符级时间戳，支持 CUDA 和 CPU 双后端，零深度学习框架依赖。

基于 Qwen3-ASR 模型架构，将音频与转录文本对齐，输出每个单词/字符的精确起止时间。

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
