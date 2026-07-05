# 对齐 Python 原版：状态交接文档

> 目标：让 `qwen-aligner-rs` 与官方 `qwen_asr`（旧版，非 `-hf`）的 forced aligner 输出
> **逐 token + 逐时间戳完全一致**。本文档记录已完成的修复、唯一未解决的根因，
> 以及接入 `libsoxr-rs`（纯 Rust 重采样器）后如何收尾验证。

参考实现（旧版，本项目对齐目标）：
- 仓库：`D:/asr/Qwen3-ASR`（`qwen_asr` 包）
- 模型：`Qwen3-ForcedAligner-0.6B`（`Qwen3ASRForConditionalGeneration` 架构）
- 关键文件：`qwen_asr/inference/qwen3_forced_aligner.py`
- **不要混淆** `-hf` 版本（`Qwen3ASRForTokenClassification`，完全不同的模型架构，
  24 层 encoder / hidden 1024 / conv_chunksize 500，本项目不对齐它）

---

## 已完成的修复（4 项，均已验证有效）

这些修复无论重采样怎么解决都成立，是净改进，**不要回退**。

### 1. 日语分词：lindera → nagisa-rs
- **问题**：原 Rust 用 `lindera`（IPADIC 词典），与 Python 原版的 `nagisa`（BiLSTM-CRF）
  分词边界不同，导致日语 token 数和文本都对不齐。
- **修复**：接入 `nagisa_rs`（git 依赖 `eclipse005/nagisa-rs`，branch `master`），
  这是上游 nagisa 的纯 Rust 移植，与 Python `nagisa` 逐 token 一致。
- **代码**：`src/text.rs` 的 `tokenize_japanese` 改用 `nagisa_rs::Tagger::tagging().words`；
  `encode_timestamp` 签名加了 `Option<&Tagger>` 参数；
  `Qwen3ForcedAligner` 持有 `Option<nagisa_rs::Tagger>`，从 `<model_dir>/nagisa/` 加载。
- **模型文件**：`models/Qwen3-ForcedAligner-0.6B/nagisa/` 下 7 个文件
 （`hp.json`、`weights.safetensors`、`uni2id.json`、`bi2id.json`、`word2id.json`、
  `pos2id.json`、`word2postags.json`，约 25MB，不进 git，随模型目录分发）。
- **验证**：日语 fixture 的 token 文本 0 分歧。

### 2. `fix_timestamp` 逐行对齐 Python
- **问题**：Rust 在 LIS 之前多了一段预处理（把每对 `start > end` 的 start 压平到 end），
  Python 原版 `qwen3_forced_aligner.py:147-234` **没有这段**。这改变了哪些 token 被判为异常，
  是 ko_4m / ja_1m / zh_180s 大块时间戳漂移的主因。
- **修复**：`src/timestamp.rs` 的 `fix_timestamp` 重写，删预处理段，逐行对应 Python。
  特别注意 `anomaly_count <= 2` 分支：Python 用 `(k - (i - 1))` 的有符号算术
  （`i==0` 时为 `k+1`），Rust 之前加了 `i > 0` 守卫导致边界行为不同，已用 `i64` 算术修正。
- **测试**：`timestamp::tests::fix_timestamp_matches_python_reference` 断言与 Python 输出一致。

### 3. argmax tie-break：epsilon → PyTorch 严格语义
- **问题**：Rust 的 `argmax_rows`（`src/inference.rs`）用了 epsilon tie-break
  （`tie_floor = best_val - 1/256`），把"接近最大值"的都算平局取第一个。
  Python `logits.argmax(dim=-1)` 是严格的 first-argmax，无 epsilon。
- **修复**：改为标准 first-argmax（`if v > best_val` 严格大于，遇相等不更新），
  删除 `F16_TIMESTAMP_ARGMAX_TIE_EPS` 常量。
- **注意**：GPU 路径也走这个 `argmax_rows`（logits 先 download 到 f32 再 CPU argmax），
  所以修复对 GPU 生效。

### 4. GPU dtype：f16 → bf16 全线重构
- **问题**：Rust GPU 用 `f16`（`__half`），Python 用 `torch.bfloat16`。两者尾数位数不同
  （f16: 10bit，bf16: 7bit），即便计算在 f32 做，存储舍入特性不同导致 argmax 边界翻向不同。
  更糟的是：**磁盘权重本身就是 BF16 存储**（708 tensors 全 BF16），Rust 之前做
  `BF16→F16` 有损转换，Python 是 `BF16→BF16` 零转换。
- **修复**：全线 `f16` → `bf16`：
  - `src/kernels/kernels.cu`：`__half`→`__nv_bfloat16`，`__half2float`→`__bfloat162float`，
    `__float2half`→`__float2bfloat16`，24 个 kernel 函数名 `_f16`→`_bf16`，加 `#include <cuda_bf16.h>`
  - `src/cudarc_engine.rs`：`use half::f16`→`bf16`，所有 `CudaSlice<f16>`→`<bf16>`，
    helper 函数改名（`upload_f16`→`upload_bf16` 等），GEMM alpha/beta，`load_function` 字符串
  - `src/gpu_audio_encoder.rs`、`src/inference.rs`：类型同步
  - `src/raw_tensor.rs`：新增 `to_bf16_vec`/`as_bf16`，BF16 dtype 走 **bit-exact reinterpret cast**
    （零舍入），保留 `to_f16_vec` 供 CPU 后端用
  - `src/cpu_engine.rs`：mel 接口（`run`/`forward` 的 `&[f16]`→`&[bf16]`），权重存储保持 f16 不变
- **验证**：kernels.cu 的 half 用法极规律（仅 `__half`类型 / `__half2float` / `__float2half`，
  全部 f32 中转无直接 half 算术），替换是纯机械操作。

---

## 唯一未解决的根因：音频重采样

### 根因
WAV 文件常以 48kHz 录制（如 `tests/fixtures/*.wav`），模型需要 16kHz mono。
- **Python**（`librosa.load(sr=16000)`）：默认 `res_type="soxr_hq"`，用 **libsoxr**
  （高质量 FFT 多相抗混叠滤波器）。
- **Rust**（`src/audio_io.rs::resample_linear`）：自写的线性插值 / 整数倍抽样，
  **无抗混叠滤波**，引入混叠失真。

两种重采样的 16kHz 样本值不同（实测 `audio[109400]`：Rust `-0.15924072` vs Python soxr `-0.15890685`，
差 8e-3），经 FFT → mel（log10 放大）→ conv2d → 24 层 encoder，在少数边界 token
（词尾）改变 argmax，约 7.5% 的 token 差 ±1 bin（±80ms）。

### 因果链已闭环验证（关键证据）
用 Python **精确复刻 Rust 的 `audio_io.rs` 音频加载链路**（hound 解码 → int16/32768 →
downmix → 线性插值重采样），跑 align，dump raw output_ids，与 Rust 二进制对比：
- **0 mismatch / 80**（之前是 6/80）

这证明：**重采样是唯一根因**。一旦 Rust 改用与 Python soxr 数值等价的重采样，
raw output_ids 将 100% 一致，加上前 4 项修复，时间戳严格一致。

### 为什么 rubato 不行
试过 `rubato = "0.16"`（纯 Rust FFT 重采样器），失败原因：
1. **group delay**：rubato 输出有 ~3918 样本（245ms）的滤波器前缀偏移，需要补偿
2. **滤波器数值不同**：即便对齐延迟，rubato 与 soxr 的抗混叠滤波器系数不同
   （实测对齐后 max diff 0.94，比线性插值的 0.85 还差）

rubato 与 soxr 是不同的 FFT 滤波器实现，**数值不兼容**，无法达到 100% 对齐。
代码已清理移除。

### 解决方案：libsoxr-rs（纯 Rust 移植 libsoxr）
由用户手写（参考 `nagisa-rs` 的移植方式），目标是与 `libsoxr` 数值 bit-exact。
关键参考：
- libsoxr 源码：https://sourceforge.net/p/soxr/code/ci/master/tree/
- Python `soxr` 包静态编译的 `soxr_ext.cp310-win_amd64.pyd` 即 libsoxr
- librosa 默认参数：`res_type="soxr_hq"`（对应 libsoxr 的 `SOXR_HQ` recipe）

---

## 接入 libsoxr-rs 的收尾步骤

当 `libsoxr-rs` 手写完成后（假设 git 仓库 `eclipse005/libsoxr-rs`）：

### 1. 加依赖
```toml
# Cargo.toml
libsoxr_rs = { git = "https://github.com/eclipse005/libsoxr-rs.git", branch = "master" }
```

### 2. 替换重采样
`src/audio_io.rs` 的 `load_wav_mono_16k` 中，把
```rust
Ok(resample_linear(&samples, spec.sample_rate, SAMPLE_RATE))
```
替换为 libsoxr-rs 的调用（参数对应 librosa 的 `soxr_hq`：input_rate, output_rate, 单声道）。
删除 `resample_linear` 函数（或保留为 fallback）。

### 3. 验证（en_15s，应一次通过）
```bash
# 生成 Rust raw
cd D:/qwen-aligner-rs
QWEN_ALIGNER_DUMP_RAW=1 ./target/release/qwen-aligner.exe align \
  --audio tests/fixtures/en_15s.wav --text tests/fixtures/en_15s.txt \
  --model models/Qwen3-ForcedAligner-0.6B --language English \
  --output /tmp/dummy.json --device cuda

# 生成 Python soxr raw（参考基准）
cd D:/asr && conda activate asr
PYTHONPATH=D:/asr/Qwen3-ASR python -c "
import json, torch, librosa
import qwen_asr.inference.qwen3_forced_aligner as fa
from qwen_asr.inference.utils import ensure_list
def align_raw(self, audio, text, language):
    texts = ensure_list(text); languages = ensure_list(language)
    wls, aits = [], []
    for t,lang in zip(texts,languages):
        wl,ait = self.aligner_processor.encode_timestamp(t,lang); wls.append(wl); aits.append(ait)
    inputs = self.processor(text=aits, audio=[audio], return_tensors='pt', padding=True)
    input_ids_cpu = inputs['input_ids'].clone()
    inputs = inputs.to(self.model.device).to(self.model.dtype)
    logits = self.model.thinker(**inputs).logits
    out_cpu = logits.argmax(dim=-1).to('cpu'); del logits
    raw=[]
    for iid, oid, wl in zip(input_ids_cpu, out_cpu, wls):
        masked = oid[iid == self.timestamp_token_id]
        raw.append(masked.numpy().astype(int).tolist())
    return raw, wls
fa.Qwen3ForcedAligner.align_raw = align_raw
from qwen_asr import Qwen3ForcedAligner
a = Qwen3ForcedAligner.from_pretrained(r'D:/voxtrans/target/debug/models/Qwen3-ForcedAligner-0.6B', dtype=torch.bfloat16, device_map='cuda:0')
audio, _ = librosa.load(r'D:/qwen-aligner-rs/tests/fixtures/en_15s.wav', sr=16000, mono=True)
text = open(r'D:/qwen-aligner-rs/tests/fixtures/en_15s.txt', encoding='utf-8').read().strip()
raw, wl = a.align_raw(audio, text, 'English')
json.dump({'output_ids': raw[0]}, open('/tmp/py_raw.json','w'))
"

# 对比（期望 0 mismatch）
python -c "
import json
py = json.load(open('/tmp/py_raw.json'))['output_ids']
rs = json.load(open('/tmp/dummy.json.raw.json'))['output_ids']
mm = [i for i,(a,b) in enumerate(zip(py,rs)) if a!=b]
print(f'mismatches: {len(mm)}/{len(py)}')
print('PASS' if not mm else f'FAIL: {mm[:10]}')
"
```

### 4. 全 fixture 复测
对 `tests/fixtures/` 下的 en_15s / en_3m / zh_180s / ko_4m / ja_1m（5 分钟以内的）
逐个跑 Rust + Python，对比 items JSON 的 text + start_time + end_time。
期望：token 文本 0 分歧，时间戳在浮点容差内严格一致。

---

## 当前代码状态（截至交接时）

- `cargo build --release` 通过，零 warning
- `cargo test --lib` 11/11 全绿
- 重采样仍是 `resample_linear`（线性插值），即 `audio_io.rs:49`
- 4 项修复全部生效，分词/token 文本已 100% 对齐
- raw output_ids 在 en_15s 上 92.5% 一致（6/80 差异，全来自重采样）

## 保留的调试设施
- `main.rs` 的 `QWEN_ALIGNER_DUMP_RAW=1` 环境变量：dump `<output>.raw.json`
  （含 output_ids / raw_timestamp_ms / fixed_timestamp_ms），供后续验证用。
  不影响正常运行，保留。

## 不在本项目范围
- `-hf` 新版模型（`Qwen3ASRForTokenClassification`）—— 用户后续另开项目重构
- ASR 模块 —— 本项目只对齐 forced aligner
