# qwen-aligner-rs

[Qwen3](https://github.com/QwenLM/Qwen3) 强制对齐器（Forced Aligner）的纯 Rust 实现，手写 CUDA + CPU 双后端，生成单词/字符级时间戳，零深度学习框架依赖。

## 特性

- **双后端**：CUDA（cuBLAS + NVRTC 手写 kernel）和 CPU（gemm + rayon + AVX2），运行时切换
- **零拷贝权重加载**：mmap safetensors
- **多语言支持**：嵌入 lindera（日文分词）+ jieba（中文分词）+ 韩文词典
- **确定性输出**：时间戳逐项一致

## 快速开始

```rust
use qwen_forced_aligner_rs::{Aligner, DeviceRequest};

let aligner = Aligner::load("path/to/model", DeviceRequest::Best)?;
let result = aligner.align("audio.wav", "transcript text")?;
for word in &result.words {
    println!("{:.3}-{:.3} {}", word.start, word.end, word.text);
}
```

命令行使用：

```bash
cargo run --release -- --model path/to/model --audio audio.wav --text "transcript" --device cuda
```

## Features

```toml
default = ["cuda", "cpu"]         # 双后端同时编译
cuda = ["dep:cudarc"]             # CUDA 后端
cpu = ["dep:gemm", "dep:rayon"]   # CPU 后端
```

CPU-only 构建：`cargo build --no-default-features --features cpu`

## License

MIT
