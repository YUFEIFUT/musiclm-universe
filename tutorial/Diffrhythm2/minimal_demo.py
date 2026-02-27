#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffRhythm2 最小化 Demo

这是一个从 DiffRhythm2 框架中提取的核心功能演示脚本。
只需要这个文件即可生成 AI 音乐。

环境要求：
- Python 3.10+
- PyTorch 2.3.1+cu121 (GPU 版本)
- 依赖库:
  - torch, torchaudio
  - transformers==4.47.1
  - muq, safetensors
  - huggingface_hub
  - pedalboard, numpy
  - phonemizer + espeak-ng (Windows)
  - jieba, cn2an, pypinyin

模型文件：
- 约 4.8GB，首次运行会自动下载到 ./ckpt/ 目录
- 需要 8GB+ GPU 显存

使用方法：
    python minimal_demo.py

生成时间：约 2-3 分钟/首歌
"""

import torch
import torchaudio
import json
import os
from tqdm import tqdm
import pedalboard
import numpy as np

from muq import MuQMuLan
from diffrhythm2.cfm import CFM
from diffrhythm2.backbones.dit import DiT
from bigvgan.model import Generator
from huggingface_hub import hf_hub_download

# ==============================================================================
# 配置参数
# ==============================================================================

# 歌词（可以直接在这里修改）
LYRICS = """[start]
[intro]
[verse]
In the town where I was born
Lived a man who sailed to sea
And he told us of his life
In the land of submarines
[chorus]
So we sailed up to the sun
Till we found a sea of green
And we lived beneath the waves
In our yellow submarine
[verse]
We all live in a yellow submarine
Yellow submarine, yellow submarine
We all live in a yellow submarine
Yellow submarine, yellow submarine
[bridge]
[inst]
[outro]
"""

# 音乐风格（支持文本描述或音频文件路径）
STYLE_PROMPT = "Pop, Cheerful, Upbeat, Beatles Style, Rock and Roll"

# 输出设置
OUTPUT_DIR = "./results/demo"
SONG_NAME = "my_song"

# 模型配置
REPO_ID = "ASLP-lab/DiffRhythm2"
CFG_STRENGTH = 2.0  # 控制生成的一致性，1.0-3.0 之间
MAX_DURATION = 210.0  # 最大时长（秒）
SAMPLE_STEPS = 16  # 采样步数，16-32，越大质量越好但越慢

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16

# ==============================================================================
# 歌词解析（简化版）
# ==============================================================================

STRUCT_INFO = {
    "[start]": 500, "[end]": 501, "[intro]": 502, "[verse]": 503,
    "[chorus]": 504, "[outro]": 505, "[inst]": 506, "[solo]": 507,
    "[bridge]": 508, "[hook]": 509, "[break]": 510, "[stop]": 511,
}

class SimpleTokenizer:
    """简化的中文/英文 tokenizer"""
    def __init__(self):
        from g2p.g2p_generation import chn_eng_g2p
        vocab_path = "g2p/g2p/vocab.json"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.phone2id = json.load(f)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text.strip())
        token = [x+1 for x in token]
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])

_lrc_tokenizer = None

def parse_lyrics_simple(lyrics: str):
    """简化版歌词解析，跳过空行"""
    global _lrc_tokenizer
    if _lrc_tokenizer is None:
        _lrc_tokenizer = SimpleTokenizer()

    lyrics_with_time = []
    for line in lyrics.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        struct_idx = STRUCT_INFO.get(line_stripped, None)
        if struct_idx is not None:
            lyrics_with_time.append([struct_idx, 511])  # 511 = [stop]
        else:
            tokens = _lrc_tokenizer.encode(line_stripped)
            tokens = tokens + [511]
            lyrics_with_time.append(tokens)

    return lyrics_with_time

# ==============================================================================
# 模型加载
# ==============================================================================

def load_models(device, dtype):
    """加载所有需要的模型"""
    print("=" * 70)
    print("正在加载模型...")
    print("=" * 70)

    # 1. 加载 DiffRhythm2 主模型
    print(f"1. 加载 DiffRhythm2 (4.3GB)...")
    ckpt_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.safetensors",
        local_dir="./ckpt"
    )
    config_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="config.json",
        local_dir="./ckpt"
    )

    with open(config_path) as f:
        model_config = json.load(f)

    model_config['use_flex_attn'] = False
    model = CFM(
        transformer=DiT(**model_config),
        num_channels=model_config['mel_dim'],
        block_size=model_config['block_size'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")

    model = model.to(device)
    from safetensors.torch import load_file
    ckpt = load_file(ckpt_path)
    model.load_state_dict(ckpt)
    print("   [OK] DiffRhythm2 加载完成")

    # 2. 加载 MuLan 风格编码器
    print(f"\n2. 加载 MuLan 风格编码器...")
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./ckpt").to(device)
    print("   [OK] MuLan 加载完成")

    # 3. 加载音频解码器
    print(f"\n3. 加载音频解码器 (BigVGAN)...")
    decoder_ckpt = hf_hub_download(
        repo_id=REPO_ID,
        filename="decoder.bin",
        local_dir="./ckpt"
    )
    decoder_config = hf_hub_download(
        repo_id=REPO_ID,
        filename="decoder.json",
        local_dir="./ckpt"
    )
    decoder = Generator(decoder_config, decoder_ckpt)
    decoder = decoder.to(device)
    print("   [OK] 解码器加载完成")

    # 转换为半精度（更快）
    if device.type != 'cpu':
        model = model.half()
        decoder = decoder.half()

    print("\n" + "=" * 70)
    print("[OK] 所有模型加载完成！")
    print("=" * 70)

    return model, mulan, decoder

# ==============================================================================
# 音乐生成
# ==============================================================================

def generate_music(model, mulan, decoder, lyrics_text, style_text, output_path):
    """生成音乐主函数"""
    print("\n开始生成音乐...")
    print(f"歌词长度: {len(lyrics_text)} 字符")
    print(f"风格提示: {style_text}")

    # 1. 解析歌词
    print("\n1. 解析歌词...")
    lyrics_tokens = parse_lyrics_simple(lyrics_text)
    lyrics_tensor = torch.tensor(sum(lyrics_tokens, []), dtype=torch.long, device=DEVICE)
    print(f"   解析完成，token 数量: {len(lyrics_tensor)}")

    # 2. 处理风格提示
    print("\n2. 处理风格提示...")
    with torch.inference_mode():
        if os.path.isfile(style_text):
            # 如果是音频文件
            print("   检测到音频风格文件，提取特征...")
            prompt_wav, sr = torchaudio.load(style_text)
            prompt_wav = torchaudio.functional.resample(prompt_wav.to(DEVICE), sr, 24000)
            if prompt_wav.shape[1] > 24000 * 10:
                start = torch.randint(0, prompt_wav.shape[1] - 24000 * 10, (1,)).item()
                prompt_wav = prompt_wav[:, start:start+24000*10]
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            style_embed = mulan(wavs=prompt_wav)
        else:
            # 如果是文本描述
            print("   使用文本风格描述，编码中...")
            style_embed = mulan(texts=[style_text])

        style_embed = style_embed.to(DEVICE).squeeze(0).half()
    print("   [OK] 风格编码完成")

    # 3. 生成音频 latent
    print("\n3. 生成音频 (扩散模型推理)...")
    print(f"   采样步数: {SAMPLE_STEPS}")
    print(f"   CFG 强度: {CFG_STRENGTH}")

    with torch.inference_mode():
        latent = model.sample_block_cache(
            text=lyrics_tensor.unsqueeze(0),
            duration=int(MAX_DURATION * 5),
            style_prompt=style_embed.unsqueeze(0),
            steps=SAMPLE_STEPS,
            cfg_strength=CFG_STRENGTH,
            process_bar=True,
        )
        latent = latent.transpose(1, 2)
    print("   [OK] 音频 latent 生成完成")

    # 4. 解码为波形
    print("\n4. 解码为波形...")
    with torch.inference_mode():
        audio = decoder.decode_audio(latent, overlap=5, chunk_size=20)
    print("   [OK] 音频解码完成")

    # 5. 保存音频
    print("\n5. 保存音频文件...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio = audio.float().cpu().numpy().squeeze()[None, :]

    # 生成假立体声（左右声道有细微差异）
    left = audio
    right = audio.copy() * 0.8
    delay_samples = int(0.01 * decoder.h.sampling_rate)
    right = np.roll(right, delay_samples)
    right[:,:delay_samples] = 0
    stereo = np.concatenate([left, right], axis=0)

    with pedalboard.io.AudioFile(output_path, "w", decoder.h.sampling_rate, 2) as f:
        f.write(stereo)

    file_size = os.path.getsize(output_path) / (1024*1024)
    print(f"   [OK] 音频保存完成: {output_path}")
    print(f"      文件大小: {file_size:.1f} MB")
    print(f"      采样率: {decoder.h.sampling_rate} Hz")
    print(f"      声道: 立体声")

    return output_path

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数：演示如何使用 DiffRhythm2 生成音乐"""
    print("\n" + "=" * 70)
    print("DiffRhythm2 最小化 Demo")
    print("AI 音乐生成")
    print("=" * 70)

    # 检查 GPU
    if DEVICE.type == 'cuda':
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[警告] 未检测到 GPU，生成速度会非常慢")
        print("   建议安装 PyTorch CUDA 版本")

    # 加载模型
    model, mulan, decoder = load_models(DEVICE, DTYPE)

    # 生成音乐
    output_path = os.path.join(OUTPUT_DIR, f"{SONG_NAME}.mp3")
    generate_music(model, mulan, decoder, LYRICS, STYLE_PROMPT, output_path)

    print("\n" + "=" * 70)
    print("[OK] 音乐生成完成！")
    print(f"文件位置: {os.path.abspath(output_path)}")
    print("=" * 70)

    # 提示如何修改
    print("\n[提示] 如何自定义生成：")
    print("   1. 修改 LYRICS 变量为你的歌词")
    print("   2. 修改 STYLE_PROMPT 改变音乐风格")
    print("   3. 修改 SONG_NAME 改变输出文件名")
    print("   4. 调整 CFG_STRENGTH (1.0-3.0) 控制创意性")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
