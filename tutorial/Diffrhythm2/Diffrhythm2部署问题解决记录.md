# DiffRhythm2 部署问题及解决记录

## 项目信息

- 项目名称：DiffRhythm2 - Efficient And High Fidelity Song Generation Via Block Flow Matching
- 项目地址：https://github.com/xiaomi-research/diffrhythm2
- 模型权重：https://huggingface.co/ASLP-lab/DiffRhythm2
- 模型大小：约 4.8GB (model.safetensors: 4.3GB + decoder.bin: 499MB)

## 环境信息

- Python 版本：3.10.11
- GPU：NVIDIA GeForce RTX 2080 SUPER (8GB)
- CUDA 版本：12.1
- 系统：Windows
- 工作目录：`P:\diffrhythm2\diffrhythm2`

---

## 部署过程中遇到的问题及解决方案

### 1. 初始环境检查

**状态**：✅ 通过

- Python 版本：3.10.11
- PyTorch：2.3.1+cu121 已安装
- 无需额外操作

### 2. 依赖安装问题

**问题描述**：直接安装 requirements.txt 失败

- `pyopenjtalk==0.4.1` 需要 CMake 和 C++ 编译环境
- 遇到错误：`CMake Error: CMAKE_C_COMPILER not set`

**解决方案**：

- 跳过 requirements.txt 的完整安装
- 手动安装必要的缺失模块：

```bash
pip install pedalboard phonemizer unidecode py3langid
```

**结果**：✅ 成功解决依赖问题

### 3. 模块缺失问题

**问题描述**：ModuleNotFoundError

```
ModuleNotFoundError: No module named 'pedalboard'
ModuleNotFoundError: No module named 'muq'
```

**解决方案**：

```bash
pip install pedalboard
pip install --force-reinstall muq==0.1.0
```

**结果**：✅ 成功安装

### 4. NumPy 版本冲突问题

**问题描述**：

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**解决方案**：

```bash
pip install numpy==1.26.0
```

**结果**：✅ 解决了版本兼容性问题

### 5. Transformers 版本不兼容问题

**问题描述**：

```
ImportError: cannot import name 'StaticCache' from 'transformers.models.llama.modeling_llama'
```

**原因**：安装了 transformers 4.57.1，但项目需要的是 4.47.1

**解决方案**：

```bash
pip install transformers==4.47.1
```

**结果**：✅ 修复了 transformers 版本问题

### 6. PyTorch CPU 版本导致段错误

**问题描述**：

```
/usr/bin/bash: line 1: 869 Segmentation fault
```

**原因**：初始安装的是 PyTorch CPU 版本 (2.9.0)，加载 4.55GB 模型到内存导致崩溃

**检查**：

```python
import torch
print(torch.cuda.is_available())  # False (CPU版本)
```

**解决方案**：
安装 CUDA 12.1 版本：

```bash
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**验证**：

```python
PyTorch: 2.3.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 2080 SUPER
GPU Memory: 8.00 GB
```

**结果**：✅ GPU 加速功能正常工作

### 7. 编码问题 (UnicodeDecodeError)

**问题描述**：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xbc in position 955:
illegal multibyte sequence
```

**原因**：读取 vocab.json 文件时使用了系统默认的 GBK 编码，而文件实际是 UTF-8 编码

**解决方案**：
修改 `inference.py` 第 54 行：

```python
# 修改前
with open(vocab_path, 'r') as file:

# 修改后
with open(vocab_path, 'r', encoding='utf-8') as file:
```

**结果**：✅ 成功解决编码问题

### 8. PyOpenJTalk 依赖问题

**问题描述**：

```
ModuleNotFoundError: No module named 'pyopenjtalk'
```

**原因**：示例歌词是英文的，不需要日文支持，但代码默认导入了日文模块

**解决方案**：
修改 `P:\diffrhythm2\diffrhythm2\g2p\g2p\cleaners.py`：

```python
# 注释掉日文导入
# from g2p.g2p.japanese import japanese_to_ipa

# 在函数中修改日文处理
elif language == "ja":
    # return japanese_to_ipa(text, text_tokenizers["ja"])
    raise Exception("Japanese language support temporarily disabled due to missing pyopenjtalk dependency")
```

**结果**：✅ 解决了依赖问题，英文歌词可正常使用

### 9. espeak 后端缺失问题

**问题描述**：

```
RuntimeError: espeak not installed on your system
```

**原因**：phonemizer 库需要 espeak 后端支持，但 Windows 系统未安装

**解决方案**：

1. 下载并安装 [eSpeak NG for Windows](https://github.com/espeak-ng/espeak-ng/releases)
2. 在代码中设置库路径：

```python
if os.name == 'nt':  # Windows
    espeak_lib = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
    if os.path.exists(espeak_lib):
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeak_lib
```

**修改的文件**：

- `g2p\utils\g2p.py:14-18`

**结果**：✅ 成功安装 espeak-ng 后端

### 10. 加载未使用的语言语音

**问题描述**：

```
Voice 'ja' not found
Voice 'ko' not found
Voice 'fr' not found
Voice 'de' not found
```

**原因**：phonemizer 尝试加载日文、韩文、法文、德文语音，但这些语音数据不存在

**解决方案**：注释掉未使用的语言，只保留中文和英文

**修改的文件**：

1. `g2p\utils\g2p.py:37-67` - 注释日语、韩语、法语、德语语音初始化
2. `g2p\g2p\__init__.py:18-25` - 注释日语、韩语、法语、德语配置
3. `g2p\g2p\cleaners.py:7,20-21` - 注释日语导入和处理逻辑

**结果**：✅ 解决了语音加载错误

### 11. 多个文件的编码问题

**问题描述**：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0x8e in position 101
UnicodeDecodeError: 'gbk' codec can't decode byte 0xbc in position 955
```

**原因**：多个文件读取操作使用了系统默认编码而非 UTF-8

**解决方案**：在所有文件读取操作中显式指定 `encoding='utf-8'`

**修改的文件**：

1. `inference.py:54` - 读取 vocab.json

   ```python
   with open(vocab_path, 'r', encoding='utf-8') as file:
   ```

2. `inference.py:233` - 读取歌词文件

   ```python
   with open(lyrics, 'r', encoding='utf-8') as f:
   ```

3. `g2p\g2p_generation.py:118` - 读取 vocab.json

   ```python
   with open(vocab_path, "r", encoding='utf-8') as f:
   ```

**结果**：✅ 解决所有编码问题

### 12. 调试断点导致程序中断

**问题描述**：

```
File "P:\diffrhythm2\diffrhythm2\inference.py", line 143, in make_fake_stereo
-> left_channel = audio
(Pdb)
bdb.BdbQuit
```

**原因**：代码中包含 `breakpoint()` 调试语句，导致程序进入调试器后崩溃

**解决方案**：移除所有 `breakpoint()` 语句

**修改的文件**：

- `inference.py:141-150` - 移除 `make_fake_stereo` 函数中的两个断点

**修改前**：

```python
def make_fake_stereo(audio, sampling_rate):
    breakpoint()
    left_channel = audio
    # ...
    breakpoint()
    return stereo_audio
```

**修改后**：

```python
def make_fake_stereo(audio, sampling_rate):
    left_channel = audio
    # ...
    return stereo_audio
```

**结果**：✅ 程序顺利执行，成功生成音乐

---

## 最终的运行环境

### 已安装的关键包

```
torch==2.3.1+cu121
torchaudio==2.3.1+cu121
torchvision==0.18.1+cu121
transformers==4.47.1
numpy==1.26.0
muq==0.1.0
pedalboard
phonemizer
unidecode
py3langid
jieba
cn2an
pypinyin
safetensors
huggingface_hub
torchaudio
pedalboard
```

### 模型文件

所有模型文件已下载到 `P:\diffrhythm2\diffrhythm2\ckpt\`：

- `model.safetensors` (4.3 GB) - 主模型权重
- `decoder.bin` (499 MB) - 解码器
- `config.json` (262 Bytes) - 配置文件
- `decoder.json` (544 Bytes) - 解码器配置
- `model.json` (263 Bytes) - 模型配置
- MuLan 模型文件在 `models--OpenMuQ--MuQ-MuLan-large` 目录

### 修改过的文件

1. **inference.py**
   - 第 54 行：添加 UTF-8 编码读取 vocab.json
   - 第 233 行：添加 UTF-8 编码读取歌词文件
   - 第 141-150 行：移除 `make_fake_stereo` 函数中的 `breakpoint()` 调试语句

2. **g2p\g2p\cleaners.py**
   - 第 7 行：注释日文导入 `# from g2p.g2p.japanese import japanese_to_ipa`
   - 第 20-21 行：注释日文处理逻辑并添加异常提示

3. **g2p\utils\g2p.py**
   - 第 14-18 行：添加 Windows espeak 库路径配置
   - 第 37-67 行：注释未使用的语言语音（日语、韩语、法语、德语）

4. **g2p\g2p\__init__.py**
   - 第 18-25 行：注释未使用的语言配置（日语、韩语、法语、德语）

5. **g2p\g2p_generation.py**
   - 第 118 行：添加 UTF-8 编码读取 vocab.json

---

## 运行命令

```bash
python inference.py \
    --repo-id ASLP-lab/DiffRhythm2 \
    --output-dir ./results/test \
    --input-jsonl ./example/test.jsonl \
    --cfg-strength 2.0
```

### 输入配置

示例文件 `example/test.jsonl` 包含：

```json
{"song_name":"song1","style_prompt": "Pop, Piano, Bass, Drums, Happy", "lyrics":"example/lrc/1.lrc"}
{"style_prompt": "example/prompt/2.wav", "lyrics":"example/lrc/1.lrc"}
```

### 生成结果

成功生成的音乐文件：

- `P:\diffrhythm2\diffrhythm2\results\test\song1.mp3` (5.6MB) - 第一首歌已生成

生成时间：约 2-3 分钟/首歌（包括模型推理和音频解码）

---

### 13. 歌词文件空行导致语言检测失败

**问题描述**：

```
Exception: Unknown language:
```

**原因**：
`inference.py` 中的 `parse_lyrics` 函数在处理歌词文件时，没有跳过空行。当遇到空行或纯空白字符的行时，会将其传递给 tokenizer 进行语言检测和分词，导致语言检测返回空值，从而触发 "Unknown language" 异常。

在英文歌词测试中，问题出现在以下场景：

1. 歌词文件末尾的空行
2. 连续的结构标签之间的空行
3. 任何包含空白字符但不包含实际歌词的行

**解决方案**：
在 `parse_lyrics` 函数中添加空行检查，跳过所有空行和纯空白字符的行。

**修改的文件**：

- `inference.py:127-142` - 修改 `parse_lyrics` 函数，添加空行跳过逻辑

**修改前**：

```python
def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.split("\n")
    for line in lyrics:
        struct_idx = STRUCT_INFO.get(line, None)
        if struct_idx is not None:
            lyrics_with_time.append([struct_idx, STRUCT_INFO['[stop]']])
        else:
            tokens = lrc_tokenizer.encode(line.strip())
            tokens = tokens + [STRUCT_INFO['[stop]']]
            lyrics_with_time.append(tokens)
    return lyrics_with_time
```

**修改后**：

```python
def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.split("\n")
    for line in lyrics:
        # Skip empty lines
        line_stripped = line.strip()
        if not line_stripped:
            continue
        struct_idx = STRUCT_INFO.get(line_stripped, None)
        if struct_idx is not None:
            lyrics_with_time.append([struct_idx, STRUCT_INFO['[stop]']])
        else:
            tokens = lrc_tokenizer.encode(line_stripped)
            tokens = tokens + [STRUCT_INFO['[stop]']]
            lyrics_with_time.append(tokens)
    return lyrics_with_time
```

**关键点**：

1. 使用 `line.strip()` 去除首尾空白字符
2. 检查 `if not line_stripped:` 跳过空行
3. 对结构标签和歌词行都使用 `line_stripped` 变量

**结果**：✅ 成功解决空行导致的语言检测错误，英文歌曲可以正常生成

**测试案例**：

- 输入：`example/english_test.jsonl` 配合 `example/lrc/english.lrc`
- 输出：`yellow_submarine.mp3`
- 风格：Pop, Cheerful, Upbeat, Beatles Style, Rock and Roll

---

## 经验教训

1. **版本匹配很重要**：PyTorch、Transformers 和 NumPy 的版本需要与项目兼容
2. **GPU 加速是关键**：对于大模型（4.3GB），必须使用 GPU 版本避免内存不足和段错误
3. **编码问题在 Windows 上很常见**：需要显式指定 UTF-8 编码读取所有文本文件
4. **依赖管理要有针对性**：不是所有 requirements.txt 中的依赖都需要，可以根据实际使用场景禁用某些功能（如日文、韩文等）
5. **外部依赖要预先安装**：如 espeak-ng 等系统级依赖需要手动安装和配置
6. **调试代码要清理**：生产代码中不应包含 `breakpoint()` 等调试语句
7. **模型加载需要耐心**：4.3GB 模型加载到 GPU 需要 2-3 分钟时间
8. **生成过程较长**：每首歌的生成时间约 2-3 分钟（105 个 diffusion steps）

---

## 建议

1. **如需日文支持**：
   - 安装 pyopenjtalk：`pip install pyopenjtalk`
   - 但需要在 Windows 上配置 CMake 和 C++ 编译环境
   - 恢复 `g2p/g2p/cleaners.py` 中的日文导入和处理逻辑

2. **如需其他语言支持**（韩语、法语、德语）：
   - 取消注释 `g2p/utils/g2p.py` 中的相应语音初始化
   - 取消注释 `g2p/g2p/__init__.py` 中的语言配置
   - 取消注释 `g2p/g2p/cleaners.py` 中的语言处理逻辑

3. **推荐的稳定版本组合**：
   - PyTorch: 2.3.1 + CUDA 12.1
   - Transformers: 4.47.1
   - NumPy: 1.26.0 (< 2.0)
   - Python: 3.10.11

4. **硬件要求**：
   - 如需批量生成音乐，确保 GPU 显存足够（建议 8GB+）
   - RTX 2080 SUPER (8GB) 可以流畅运行

5. **Windows 系统建议**：
   - 预先安装 eSpeak NG
   - 所有文件操作使用 UTF-8 编码
   - 考虑使用虚拟环境隔离依赖

---

**记录日期**：2025-11-13
**最终状态**：✅ DiffRhythm2 成功运行并生成音乐
