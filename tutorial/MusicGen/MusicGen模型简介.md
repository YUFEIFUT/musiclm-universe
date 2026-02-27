# MusicGen 模型简介

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      MusicGen Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Text Prompt ──────────────────────────────────┐            │
│  "80s synth pop with drums"                    │            │
│                                                ▼            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Text Encoder (T5/Flan-T5)                │  │
│  │              • Pre-trained, frozen                    │  │
│  │              • Output: text embeddings                │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              MusicGen Transformer Decoder             │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  • Codebook Embedding (4 codebooks)             │  │  │
│  │  │  • Delay Pattern (interleaving)                 │  │  │
│  │  │  • Transformer Blocks (12/24/32 layers)         │  │  │
│  │  │  • Output Projection (4 heads)                  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │              ↓ (autoregressive generation)            │  │
│  │         [Audio tokens: 4 parallel streams]            │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              EnCodec Audio Decoder                    │  │
│  │              • Pre-trained, frozen                    │  │
│  │              • Decodes tokens → audio waveform        │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  Audio Output (.wav) @ 32kHz                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 模型规格

| 组件        | 架构 | 参数数量     | 模型大小     | 输入     | 输出           |
|------------|--------------------------|----------|----------|--------|--------------|
| MusicGen   | Transfromer Decoder      | 0.3-3.3B | 1.2-12GB | 文本嵌入   | 音频token(4码本) |
| T5/Flan-T5 | Encoder-only Transformer | 0.2-3B   | 0.8-11GB | 文本     | 文本嵌入         |
| EnCodec    | CNN + RVQ                | -        | 50 MB    | 音频token | 32kHz 音频     |

---

## 核心组件

### 1. MusicGen主模型

**`audiocraft/models/musicgen.py`** - MusicGen 模型入口
- `class MusicGen`: MusicGen 主模型封装
- `get_pretrained(name)`: 加载预训练 MusicGen 模型
- `set_generation_params(...)`: 设置生成参数（duration、temperature、top_k、top_p、cfg_coef 等）
- `generate(descriptions, ...)`: 基于文本描述生成音乐
- `generate_with_chroma(descriptions, melody_wavs, ...)`: 基于文本 + 旋律条件生成音乐
- `_generate_tokens(...)`: 内部调用语言模型生成音频 token
- `_decode_audio(tokens)`: 使用 EnCodec 将 token 解码为音频波形

**`audiocraft/models/lm.py`** - 核心自回归语言模型（Transformer）
- `class LMModel`: 音频语言模型（Decoder-only Transformer）
- `generate(...)`: 自回归生成音频 token 的主方法
- `forward(...)`: Transformer 前向传播，输出 logits

**`audiocraft/models/encodec.py`** - 音频编解码模型
- `class CompressionModel`: EnCodec 模型封装
- `encode(wav)`: 音频波形 → 离散音频 token
- `decode(tokens)`: 离散音频 token → 音频波形

**`audiocraft/modules/conditioners/text.py`** - 文本条件编码
- `class T5Conditioner`: 基于 T5 的文本条件编码器
- `forward(texts)`: 文本描述 → 条件 embedding

**`audiocraft/modules/conditioners/chroma.py`** - 旋律条件编码
- `class ChromaConditioner`: 旋律条件编码器
- `forward(wavs)`: 旋律音频 → chroma 特征

### 2. 条件编码器（Conditioners）

**`audiocraft/modules/conditioners/base.py`** - 条件编码器基类
- `class Conditioner`: 所有条件编码器的抽象基类
- `forward(...)`: 条件输入 → 条件 embedding

**`audiocraft/modules/conditioners/text.py`** - 文本条件编码
- `class T5Conditioner`: 基于 T5 Encoder 的文本条件编码器
- `forward(texts)`: 文本描述 → 条件 embedding（用于 cross-attention）

**`audiocraft/modules/conditioners/chroma.py`** - 旋律条件编码
- `class ChromaConditioner`: 基于 chroma 的旋律条件编码器
- `forward(wavs)`: 旋律音频 → chroma 特征条件

### 3. 回归生成（Audio Token Generation）

**`audiocraft/models/lm.py`** - 音频语言模型
- `class LMModel`: Decoder-only Transformer 音频语言模型
- `forward(...)`: 前向传播，输出音频 token logits
- `generate(...)`: 自回归方式生成音频离散 token 序列

**`audiocraft/models/musicgen.py`** - 生成流程调度
- `_generate_tokens(...)`: 
  - 组织条件编码
  - 调用 `LMModel.generate`
  - 返回生成的音频 token 序列

