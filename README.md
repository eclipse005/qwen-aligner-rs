# qwen-aligner-rs

> Qwen3 强制对齐器（Forced Aligner）的纯 Rust 实现。**手写 CUDA + CPU 双后端**，生成单词/字符级时间戳，零深度学习框架依赖。

## 为什么不用 Candle / Burn？

原版基于 [`candle`](https://github.com/huggingface/candle) 实现，对齐器推理场景有几个痛点：

- **conv-stem 是瓶颈**：candle 的通用 conv2d 对 ASR 音频编码器的 1D conv 不友好，缺少 im2col + fused kernel 优化
- **Attention score 内存爆炸**：长音频的 Q@K 矩阵在 candle 下分配完整 (seq × seq) 内存，180s 音频需要 110MB 临时空间
- **CPU GEMV 单线程**：m=1 的 lm_head 在 candle 下走单线程
- **无法精细控制 fusion**：RMSNorm + Rotary + Attention 的 fusion 在 candle 里做不到

本项目直接用 `cudarc` + `cuBLAS` + `NVRTC` 手写所有 kernel，CPU 路径用 `gemm` + `rayon` + AVX2 SIMD。**热路径上没有任何深度学习框架**。

## 性能（vs Candle 原版）

| 指标 | Candle 原版 | 本项目 | 提升 |
|------|------------|--------|------|
| 180s 音频 CPU 对齐 | ~49.5s | ~21.6s | **2.3x** |
| 15s 音频 CPU 对齐 | ~2.3s | ~1.0s | **2.3x** |
| 180s 音频 GPU 对齐 (P104-100) | ~25s | ~10s | **2.5x** |
| Attention 临时内存 (180s) | 110MB | 0MB（tiled online softmax） | **∞** |
| 模型加载时间 | ~40s | ~5s | **8x**（mmap + 零拷贝） |

> 测试硬件：CPU = Intel Core Ultra 7 265K (Arrow Lake, 20c, AVX2)；GPU = P104-100 (Pascal sm_61, 8GB)。
> 精度：40/40 (15s) + 909/909 (180s) 时间戳逐项一致，5 次独立运行确定性验证。

## 特性

- **双后端，单一二进制**：CUDA + CPU 同时编译进同一个库，运行时通过 `DeviceRequest` 切换
- **CUDA 路径**：cuBLAS HGEMM + NVRTC 手写 kernel（fused RMSNorm、Rotary、FlashAttention 风格 tiled online softmax）
- **CPU 路径**：gemm + rayon + AVX2 SIMD（手写 dot_qk_avx2 / axpy_avx2 helper）
- **FlashAttention 风格 online softmax**：替代 110MB scores scratch，长音频内存友好
- **Conv-stem NHWC direct conv**：用 `[f32; 512]` 累加器 + 1-row 8-col microkernel，conv2 提速 1.8x
- **零拷贝权重加载**：mmap safetensors + `Bytes::from_owner`
- **多语言支持**：嵌入 lindera（日文分词）+ jieba（中文分词）+ 韩文词典

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
default = ["cuda", "cpu"]  # 双后端同时编译
cuda = ["dep:cudarc"]      # CUDA 后端
cpu = ["dep:gemm", "dep:rayon"]  # CPU 后端
```

CPU-only 构建：`cargo build --no-default-features --features cpu`

## 项目结构

```
src/
├── backend.rs            # Backend 枚举 + 调度
├── inference.rs          # 主对齐循环：mel → embed → CTC alignment
├── cpu_engine.rs         # 手写 CPU 引擎（gemm + rayon + AVX2 SIMD）
├── cudarc_engine.rs      # 手写 GPU 引擎（cuBLAS + NVRTC kernel）
├── gpu_audio_encoder.rs  # 手写 GPU 音频编码器（NHWC direct conv）
├── kernels/kernels.cu    # 所有 CUDA kernel（NVRTC 运行时编译）
├── weights.rs            # safetensors mmap 零拷贝权重加载
├── tokenizer.rs          # 多语言分词器（lindera + jieba + 韩文）
└── timestamp.rs          # CTC 解码 + 时间戳生成
```

## License

MIT
