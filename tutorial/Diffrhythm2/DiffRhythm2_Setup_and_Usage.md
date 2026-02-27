# DiffRhythm2_minimal_Demo 使用指南

`minimal_demo.py` 是从 DiffRhythm2 完整框架中提取的核心功能演示脚本。**只保留生成音乐所必需的最少代码**，去掉所有复杂性，让你能快速理解和使用 AI 音乐生成功能。

与完整框架相比：

- **代码量**: 500行 vs 2000+行（多个文件）
- **文件数**: 1个 vs 50+个
- **依赖**: 相同（但理解更容易）
- **功能**: 保留核心功能，去掉批处理和高级选项



## 快速开始

### 1. 确保环境准备就绪

```bash
# 检查 Python 版本
python --version  # 需要 3.10+

# 检查 GPU 是否可用
cd diffrhythm2
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

**如果没有正确配置环境**，请参考 `DiffRhythm2_部署问题解决记录.md` 

[DiffRhythm2 部署问题解决记录](https://github.com/minimum-generated-pig/musiclm-universe/blob/main/tutorial/Diffrhythm2/Diffrhythm2%E9%83%A8%E7%BD%B2%E9%97%AE%E9%A2%98%E8%A7%A3%E5%86%B3%E8%AE%B0%E5%BD%95.md)

或运行：

```bash
# 安装核心依赖
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.1 numpy==1.26.0
pip install muq safetensors huggingface_hub pedalboard

# 安装音频处理依赖
pip install phonemizer jieba cn2an pypinyin

# Windows 用户还需要安装 eSpeak NG:
# 下载: https://github.com/espeak-ng/espeak-ng/releases
```

### 2. 运行最小化 Demo

```bash
cd diffrhythm2
python minimal_demo.py
```

**第一次运行会自动下载模型**（约 4.8GB，需要等待）。

### 3. 查看结果

生成的音乐保存在：

```
./results/demo/my_song.mp3
```

## 如何自定义生成？

编辑 `minimal_demo.py` 开头的配置区域：

### 修改歌词

找到 `LYRICS` 变量：

```python
LYRICS = """[start]
[intro]
[verse]
你的歌词第一行
你的歌词第二行
[chorus]
副歌部分的歌词
[outro]
"""
```

**歌词格式说明**：

- 每行一句歌词
- 结构标签（可选）：`[intro]`, `[verse]`, `[chorus]`, `[bridge]`, `[outro]`
- 空行会被自动忽略
- 支持英文和中文混输

### 修改音乐风格

找到 `STYLE_PROMPT` 变量：

```python
STYLE_PROMPT = "Pop, Cheerful, Upbeat, Beatles Style, Rock and Roll"
# 可以是任何文本描述

# 或者使用音频文件作为风格参考：
# STYLE_PROMPT = "./path/to/reference_audio.wav"
```

**风格提示示例**：

- `Pop, Happy, Energetic, Piano, Drums`
- `Rock, Electric Guitar, Heavy Bass, Aggressive`
- `Jazz, Smooth, Saxophone, Lounge Music`
- `Classical, Orchestra, Epic, Cinematic`
- `Electronic, Synthwave, Retro, 80s`

### 修改输出设置

```python
OUTPUT_DIR = "./results/demo"  # 输出目录
SONG_NAME = "my_song"          # 文件名（不需要 .mp3）
CFG_STRENGTH = 2.0             # 控制生成的一致性，1.0-3.0
SAMPLE_STEPS = 16              # 采样步数，16（快）或 32（质量更好）
MAX_DURATION = 210.0           # 最大时长（秒）
```

### 时间控制说明

#### 为什么生成的歌曲时间相似？

你生成的歌曲可能都是2分38秒左右，这是**正常现象**，原因如下：

1. **时间由歌词行数决定**

   - 每行歌词 ≈ 5秒
   - 30行歌词 ≈ 150秒（2分30秒）
   - 结构标签（[intro], [verse]等）也会占用时间

2. **MAX_DURATION 是上限，不是精确值**

   ```python
   MAX_DURATION = 210.0  # 最多生成3.5分钟
   ```

   这个参数限制了**最长时长**，实际时长由歌词内容决定

#### 理论时间控制公式

```
总时长 ≈ 歌词行数 × 5秒 + 结构标签 × 2-3秒
```

**示例**：

- 10行歌词 ≈ 50秒
- 20行歌词 ≈ 100秒（1分40秒）
- 30行歌词 ≈ 150秒（2分30秒）
- 40行歌词 ≈ 200秒（3分20秒）

#### 如何控制生成时长？

有3种方法：

**方法1：调整歌词行数（最简单）**

```python
# 生成短歌（约1分钟）
LYRICS = """[start]
[intro]
[verse]
第一行歌词
第二行歌词
[chorus]
副歌部分歌词
[outro]
"""
# 约5-6行 ≈ 30-40秒

# 生成长歌（约3分钟）
LYRICS = """[start]
[intro]
[verse]
歌词第1行
歌词第2行
...
[verse]
歌词第N行
[chorus]
副歌部分
[bridge]
桥段歌词
[outro]
"""
# 约40行 ≈ 3分钟
```

**方法2：使用 --max-secs 参数（批量生成时）**

如果使用 `inference.py` 批量生成，可以指定最大时长：

```bash
# 生成长达5分钟的歌
python inference.py --max-secs 300 ...

# 生成1分钟短歌
python inference.py --max-secs 60 ...
```

**方法3：调整 duration 系数（高级）**

编辑 `inference.py` 第173行：

```python
# 默认：duration=int(max_secs * 5)
# 改为：duration=int(max_secs * 3)  # 每句歌词更短
```

#### 实际应用示例

如果你想生成不同时长的歌曲：

```python
# 30秒铃声（简洁）
LYRICS = """[start]
[verse]
简短的歌词一行
再一行
[chorus]
铃声音符
[outro]
"""

# 5分钟完整歌曲（详细）
LYRICS = """[start]
[intro]
[verse1]
第1段主歌歌词...
...更多行...
[verse2]
第2段主歌歌词...
...更多行...
[pre-chorus]
[chorus]
副歌部分...
...重复段落...
[bridge]
桥段...
[outro]
"""
```

**总结**：

- 时间完全可控
- 行数越多 = 时间越长
- 每行 ≈ 5秒
- 结构标签也会占用时间

**技巧**：

- 测试用：5-8行歌词（30-40秒）
- 标准歌：15-20行歌词（1.5-2分钟）
- 完整歌：30-40行歌词（2.5-3.5分钟）

## 对比：完整框架 vs 最小化 Demo

| 功能         | 完整框架           | 最小化 Demo      |
| ------------ | ------------------ | ---------------- |
| 代码行数     | 2000+              | ~500             |
| 支持批量生成 | ✅                  | ❌                |
| 支持音频风格 | ✅                  | ✅ (简化)         |
| 支持文本风格 | ✅                  | ✅                |
| 中英双语     | ✅                  | ✅                |
| 多语言支持   | ✅ (日、韩、法、德) | ❌ (已注释掉)     |
| 命令行参数   | ✅ (复杂)           | ❌ (直接修改代码) |
| 进度条       | ✅                  | ✅                |
| 生成质量     | 相同               | 相同             |
| 学习难度     | 高                 | 低               |

## 技术要点（如果你想理解代码）

### 核心流程（5个步骤）

1. **加载模型** (~2分钟)
   - DiffRhythm2（扩散主模型，4.3GB）
   - MuLan（风格编码器）
   - BigVGAN（音频解码器）

2. **解析歌词** (< 1秒)
   - 分词并转换为 token ID
   - 处理结构标签 `[verse]`, `[chorus]` 等
   - 中英文分开处理

3. **编码风格** (< 1秒)
   - 文本描述 → 向量
   - 或 参考音频 → 向量

4. **生成音频** (2-3分钟)
   - 扩散模型采样
   - 从随机噪声逐步生成音频特征

5. **解码保存** (< 10秒)
   - 音频特征 → 波形
   - 保存为 MP3 文件

### 关键技术

- **扩散模型（Diffusion Model）**: 核心生成技术
- **CFG（Classifier-Free Guidance）**: 控制生成的一致性
- **MuLan**: 多模态音乐理解模型，用于风格编码
- **BigVGAN**: 高质量音频解码器

## 常见问题

### Q: 运行时报错 "Unknown language"

A: 确保歌词文件中没有空行，或检查 `parse_lyrics_simple` 函数是否有空行跳过逻辑

### Q: 生成时间太长

A: 降低 `SAMPLE_STEPS` 到 8-12，或者换更好的 GPU

### Q: 生成的音乐质量不好

A: 提高 `SAMPLE_STEPS` 到 32，或调整 `CFG_STRENGTH`（1.5-2.5 之间尝试）

### Q: 想生成中文歌曲

A: 修改 `LYRICS` 为中文歌词即可，代码已支持中英文混输

### Q: 可以用自己的风格音频吗？

A: 可以！将 `STYLE_PROMPT` 设置为音频文件路径：

```python
STYLE_PROMPT = "./my_style.wav"
```

## 如何进一步简化？

如果你想要**更简化**的版本（比如用于教学演示）：

1. **移除立体声处理**：去掉 `make_fake_stereo` 相关代码，输出单声道
2. **硬编码参数**：将 `CFG_STRENGTH` 等参数直接写死在函数里
3. **固定歌词**：在代码里写死一行歌词，不需要解析
4. **移除错误处理**：去掉所有 try-except 和检查

这样可以进一步减少到 **300行左右**。

## 如何扩展功能？

如果你想要**更完整**的功能，可以参考 `inference.py`：

1. **批量生成**：读取 JSONL 文件批量生成多首歌
2. **更多参数**：支持命令行参数调整所有配置
3. **更多语言**：启用日语、韩语、法语、德语支持
4. **更多音频格式**：支持 WAV, FLAC 等输出格式
5. **音频后处理**：添加混响、均衡器等效果

---

**记住**：这个 demo 的目的是让你**快速理解核心原理**，而不是替代完整框架。当你熟悉后，可以逐步探索完整框架的更多功能！
