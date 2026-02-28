# ACE-Step 1.5 开源代码分析

modeling_acestep_v15_base.py，这是 PyTorch 版本的完整模型结构定义文件。

```
AceStepConditionGenerationModel
⬇
AceStepDiTModel
⬇
AceStepDiTLayer × N
⬇
Qwen3 attention + MLP
```

这里它本质是 Qwen3 Transformer + Diffusion 外壳

因为里面 import 了：
- Qwen3MLP
- Qwen3RMSNorm
- Qwen3RotaryEmbedding

说明：它是基于 Qwen3 改造的 DiT。（注：DiT 是 Diffusion Transformer）

我们重点看以下类和代码：

1、class AceStepConditionGenerationModel
- def forward(
- def generate_audio(

2、class AceStepDiTModel
- def forward(

3、AceStepDiTLayer

4、TimestepEmbedding

---

## class AceStepConditionGenerationModel

### 一、定义 def **init**(

```python
class AceStepConditionGenerationModel(AceStepPreTrainedModel):  
    """  
    AceStep 核心条件生成模型  
    
    基于文本、歌词和音色条件生成音频的端到端模型。  
    整合了编码器（用于条件信息处理）、解码器（扩散模型）、分词器（用于离散化token编码）和解码器（用于音频重构）。  
    支持流匹配（flow matching）训练，以及多种采样方法的推理过程。  
    """  

    def __init__(self, config: AceStepConfig):  
        super().__init__(config)  
        self.config = config   # 把 config 存到当前模型实例中。  

        # Diffusion model components  
        self.decoder = AceStepDiTModel(config)  # 3️⃣Main diffusion transformer（DIT模型）  
        self.encoder = AceStepConditionEncoder(config)  # 1️⃣Condition encoder（文本输入处理）  
        self.tokenizer = AceStepAudioTokenizer(config)  # 2️⃣Audio tokenizer（音频输入处理）  
        self.detokenizer = AudioTokenDetokenizer(config)  # 最后 4️⃣Audio detokenizer（以音频形式输出）  

        # Null condition embedding for classifier-free guidance  
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, config.hidden_size))  

        # Initialize weights and apply final processing  
        self.post_init()  
```

按这个顺一下 DiT 模型生成音乐 的大概流程：

1️⃣ **条件编码器**

self.encoder = AceStepConditionEncoder(config)

作用：把文本 / 歌词 / 音色条件编码成 embedding

例如：

```
输入：
文本 token
歌词 token
speaker embedding

输出：
condition_embedding  (shape: [B, T, hidden_size])
```

这个 embedding 会送入 diffusion transformer （扩散DiT模型）。

2️⃣ **音频 tokenizer**

```
self.tokenizer = AceStepAudioTokenizer(config)
```

作用：把连续音频波形 → 离散 token

通常是：
VQ-VAE
EnCodec
SoundStream

流程：
waveform → 编码器 → quantizer → discrete codes

这个 embedding 也会送入 diffusion transformer （扩散DiT模型）。

3️⃣ **扩散 Transformer 模型（主要）**

```
self.decoder = AceStepDiTModel(config)
```

这是核心生成模块 DiT Model（Diffusion Transformer），它和GPT一样也是个Decoder-only哈。

作用：一步步根据 条件 + 噪声（ 输入文本的embedding + 输入音频 (带噪声) ） 预测去噪结果

扩散流程中它会：总是根据 带噪声的音频（就是压缩的音频）去预测  去噪向量

```
输入：
noisy_latent         1、带噪声的音频 / 压缩了的音频   --来源于2️⃣编码了的输入音频
time_step            2、时间步                        --来源于2️⃣编码了的输入音频

condition_embedding  3、条件向量                      --来源于1️⃣条件编码器得到的 文本 embedding  

输出：
预测噪声             预测去噪向量
```

4️⃣ **音频 detokenizer**

```
self.detokenizer = AudioTokenDetokenizer(config)
```

作用：把离散 token 还原为音频

流程：

```
token → embedding → decoder → waveform
```

输出音频就好了。

**所以，这样整个流程就是：**

```
编码输入文本 + 编码输入音频
↓
一步步 diffusion → 生成 token
↓
解码音频，输出
```

---

#### 接下来，Classifier-Free Guidance 相关，就是 CFG，之前在MusicGen也有哈。

```
self.null_condition_emb = nn.Parameter(
torch.randn(1, 1, config.hidden_size)
)
```

CFG，这是一个可训练参数，shape: (1, 1, hidden_size)

#### 什么是 CFG？

训练时：
- 有时用真实条件
- 有时用“空条件”

模型学会：
- p(x | condition)
- p(x | null)

推理时可以做：

```
x = x_uncond + scale * (x_cond - x_uncond)
```

增强条件控制能力。

最后，权重初始化，self.post_init()。

---

### 二、forward() 参数定义初始化

这段 forward() 是训练阶段的前向传播逻辑，核心思想是：
使用 Flow Matching 训练扩散 Transformer，让它学会从任意时间点的插值样本预测“流场（velocity）”。

```python
def forward(  
        self,  
        # Diffusion inputs  
        hidden_states: torch.FloatTensor,  
        attention_mask: torch.Tensor,  
        # Encoder inputs  
        # Text  
        text_hidden_states: Optional[torch.FloatTensor] = None,  
        text_attention_mask: Optional[torch.Tensor] = None,  
        # Lyric  
        lyric_hidden_states: Optional[torch.LongTensor] = None,  
        lyric_attention_mask: Optional[torch.Tensor] = None,  
        # Reference audio for timbre  
        refer_audio_acoustic_hidden_states_packed: Optional[torch.Tensor] = None,  
        refer_audio_order_mask: Optional[torch.LongTensor] = None,  
        src_latents: torch.FloatTensor = None,  
        chunk_masks: torch.FloatTensor = None,  
        is_covers: torch.Tensor = None,  
        silence_latent: torch.FloatTensor = None,  
        cfg_ratio: float = 0.15,  
    ):  
        """  
        Forward pass for training (computes training losses).  
        """  
        # Prepare conditioning inputs (encoder states, context latents)  
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(  
            text_hidden_states=text_hidden_states,  
            ...  
            ...  
        )  
```

这是训练阶段使用的 forward（不是推理）。

看看参数定义初始化都有啥：

**1、条件输入 [ 分别是 编码向量 和 mask ]**

支持三种条件：文本 歌词 参考音频（音色）
```
text_hidden_states
text_attention_mask
```
这是已经编码好的文本 embedding。
```
lyric_hidden_states
lyric_attention_mask
```
这是已经 歌词 token 或 embedding。
```
refer_audio_acoustic_hidden_states_packed
refer_audio_order_mask
```
这是已经编码好的参考音频 embedding：用于 timbre cloning。

**2、Diffusion 输入 [ 同样是 编码向量 和 mask ]**
```
hidden_states: torch.FloatTensor,
attention_mask: torch.Tensor,
```
- hidden_states 真实音频的 latent 表示（x₀）shape: [B, T, D]
- attention_mask 控制哪些 token 有效（padding mask）

**3、其他输入**
```
src_latents
chunk_masks
is_covers
silence_latent
```
这些通常用于：多段拼接、是否翻唱、静音控制、分块结构

**4、CFG 比例**

cfg_ratio: float = 0.15

表示 有 15% 概率丢弃条件 用于 Classifier-Free Guidance 训练。

---

### 三、融合条件输入

```python
        """  
        Forward pass for training (computes training losses).  
        """  
        # Prepare conditioning inputs (encoder states, context latents)  
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(  
            text_hidden_states=text_hidden_states,  
            ...  
            ...  
        )  
```

把所有条件：
```
文本
歌词
参考音频
chunk 信息
```

融合成：
```
encoder_hidden_states   → 主条件序列
encoder_attention_mask  → 条件mask
context_latents         → 额外上下文
```
此时已经完成：所有条件融合

---

### 四、基于 Classifier-Free Guidance 训练 DiT 模型

这是非常关键的一部分。

```python
        # 无分类器引导（Classifier-free guidance）：以概率 cfg_ratio 随机丢弃条件信息  
        # 该策略帮助模型学习在有/无条件信息的场景下均能有效工作  
        full_cfg_condition_mask = torch.where(  
            (torch.rand(size=(bsz,), device=device, dtype=dtype) < cfg_ratio),  
            torch.zeros(size=(bsz,), device=device, dtype=dtype),  
            torch.ones(size=(bsz,), device=device, dtype=dtype)  
        ).view(-1, 1, 1)  
        
        # 将被丢弃的条件信息替换为空白条件嵌入（null condition embedding）  
        encoder_hidden_states = torch.where(  
            full_cfg_condition_mask > 0,  
            encoder_hidden_states,  
            self.null_condition_emb.expand_as(encoder_hidden_states)  
        )  
```

我们给每个样本抽个 “签”（0~1 的随机数）：

- 抽到小于 0.15 的签（概率 15%）：标记为0 → 代表 “丢弃这个样本的文本条件”；
- 抽到大于等于 0.15 的签（概率 85%）：标记为1 → 代表 “保留这个样本的文本条件”；

最后把这个标记（mask）改成[B,1,1]的形状，这样条件 embedding 可以广播。

结合 Step1 的 mask：

- 样本 mask=1 → 保留原文本条件（模型学 “根据文本生成图片”）；
- 样本 mask=0 → 把文本条件换成空向量（self.null_condition_emb：一个“无意义”的空向量）；

**为什么要这么做？**

模型在训练时，一部分样本 “带着文本条件学”，一部分样本 “不带文本条件学”，最终会具备两种能力：

- 有条件生成：给文本→生成对应内容；
- 无条件生成：不给文本→生成随机但合理的内容

到了推理阶段，我们就可以把这两种生成结果加权融合（比如：有条件结果 ×7 - 无条件结果 ×6），让生成的图片既符合文本描述，又有更高的质量和细节。

---

### 五、Flow Matching 核心部分

这是本段代码最重要的思想。

```python
        # 流匹配（Flow Matching）的实现流程：采样噪声 x₁，并与数据 x₀ 进行插值运算。  
        x1 = torch.randn_like(hidden_states)  # Noise  
        x0 = hidden_states  # Data  
        # Sample timesteps t and r for flow matching  
        t, r = sample_t_r(bsz, device, dtype, self.config.data_proportion, self.config.timestep_mu, self.config.timestep_sigma, use_meanflow=False)  
        t_ = t.unsqueeze(-1).unsqueeze(-1)  
        # Interpolate: x_t = t * x1 + (1 - t) * x0  
        xt = t_ * x1 + (1.0 - t_) * x0  
```

这一段在 “纯噪声 x₁” 和 “真实数据 x₀” 之间，随机选一个时间点 t，构造一个中间状态 xₜ，让模型学会 “从这个位置往数据方向走” 。

#### 1️⃣ 两个端点

* x0 = hidden_states 是真实音频的 latent 表示（干净数据）
* x1 = torch.randn_like(hidden_states) 是同形状的高斯噪声（纯噪声）

#### 2️⃣ 随机采样一个时间点 t ∈ (0,1)

* t 接近 0 → 更像真实数据
* t 接近 1 → 更像噪声

然后构造：

```
xt = t * x1 + (1 - t) * x0
```

这是在“噪声”和“数据”之间做线性插值。

#### 比如

```
x0 = 真实钢琴音乐 latent
x1 = 完全随机噪声
t = 0.3
```

那：xt = 0.3 * 噪声 + 0.7 * 真实数据

这就是：“30% 噪声 + 70% 真实音乐”

模型的任务就是给模型这个 “半噪声状态”，它要学会往正确方向走。

注意：这和传统 DDPM 不一样。这更是在学一个 “流场（vector field）”。

* DDPM 是预测 ε（噪声）
* Flow Matching 是预测速度 v

---

### 六、把 xt 放进 DiT 模型，来预测“流”

```python
        # Predict flow (velocity) from diffusion model
        decoder_outputs = self.decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=t,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
```

这一部分把当前的中间状态 xt + 条件 + 时间步 t 输入 DiT，让它预测 “应该往哪里走”。

输入包括：

* hidden_states=xt 当前噪声-数据混合状态
* timestep=t 当前处于扩散过程的哪个位置
* encoder_hidden_states 文本 / 歌词 / 音色条件

模型需解决：“我现在在 30% 噪声状态，条件是‘悲伤的钢琴曲’，我应该往哪个方向走？”

DiT 会输出一个向量：

```
v̂  （预测的 flow / velocity）
```

这个向量和 xt 形状一样：shape: [B, T, D]

---

### 七、Flow Matching 损失函数

```python
        # Flow matching loss: predict the flow field v = x1 - x0
        flow = x1 - x0
        diffusion_loss = F.mse_loss(decoder_outputs[0], flow)
        return {
            "diffusion_loss": diffusion_loss,
        }
```

这里是整个训练的核心监督信号。

#### 真实“流”是是从数据点 x0 指向噪声点 x1 的向量

```
flow = x1 - x0
```

模型输出：

```
v̂ = decoder_outputs[0]
```

我们希望 v̂ ≈ (x1 - x0)

所以用 MSE(v̂, flow)

训练目标：让模型学会在任何中间状态 xt，都知道“噪声方向在哪里”。

---

### 八、输入输出举例

```
Batch size = 2

hidden_states: [2, 1024, 768]     # 音频 latent
text_hidden_states: [2, 128, 768] # 文本 embedding
```

文本示例：

* 样本1："悲伤的钢琴曲"
* 样本2："欢快的电子舞曲"

#### 训练流程

采样噪声：x1: [2, 1024, 768]

插值得到：xt: [2, 1024, 768]

DiT 预测：v̂: [2, 1024, 768]

真实 flow：flow = x1 - x0

损失：diffusion_loss = MSE(v̂, flow)


#### 推理时会发生什么？

推理不再用真实 x0。而是：x = 随机噪声。然后不断用模型预测 flow，做数值积分：

```
x ← x - Δt * v̂
```

ACE-Step 1.5 的核心训练目标不是“预测噪声”，而是学习一个从数据到噪声的“连续流场”。它本质是 LLM Transformer 内核 + 连续时间扩散生成框架。

```
Qwen3 Transformer 结构
+
Diffusion 外壳
+
Flow Matching 训练方式
```

---

## 接下来分析：class AceStepDiTModel(AceStepPreTrainedModel):

```
class AceStepDiTModel(AceStepPreTrainedModel):
    """
    用于 AceStep 的 DiT（扩散Transformer）模型。

    这是主扩散模型，基于文本、歌词和音色条件，生成音频隐变量。
    采用基于补丁（patch）的处理方式，搭配 Transformer 层、时间步条件注入，
    并通过交叉注意力机制与编码器输出进行交互。
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
```

这个类是 **主扩散模型**，也就是整个系统真正负责“从噪声里生成音频 latent”的核心模块。它的任务不是生成 waveform，而是生成音频的隐变量表示（latent），最后再交给 detokenizer 解码成声音。

它其实是把一个 **Qwen3 Transformer 改造成 Diffusion Transformer（DiT）**

---

### 二、Transformer 主干结构

```
        # Rotary position embeddings for transformer layers
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        # Stack of DiT transformer layers
        self.layers = nn.ModuleList(
            [AceStepDiTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        in_channels = config.in_channels
        inner_dim = config.hidden_size
        patch_size = config.patch_size
        self.patch_size = patch_size
```

这里是旋转位置编码和多层 Transformer
* `rotary_emb`：用 Qwen3 的 RoPE 做位置编码
* `layers`：堆叠 N 层 `AceStepDiTLayer`

它本质上是一个 **Decoder-only Transformer**，和 GPT 很像，只不过训练目标不是预测 token，而是预测连续向量（flow）。


### 三、Patch Embedding

一个很关键的部分：把长序列音频做 patch 化。

```
self.proj_in = nn.Sequential(
    Lambda(lambda x: x.transpose(1, 2)),
    nn.Conv1d(
        in_channels=in_channels,
        out_channels=inner_dim,
        kernel_size=patch_size,
        stride=patch_size,
        padding=0,
    ),
    Lambda(lambda x: x.transpose(1, 2)),
)
```

这里
1. 把输入从 `[B, T, C]` 转成 `[B, C, T]`
2. 用 `Conv1d` 做 patch 切分（kernel_size = stride = patch_size）
3. 再转回 `[B, T//patch, hidden_dim]`

这相当于把连续时间序列每 `patch_size` 个时间步压缩成一个 token。好处：

* 序列长度缩短
* Transformer 计算量下降
* 类似 ViT 里的 patch embedding 思路

比如原来 T=1024，patch_size=4，经过 patch 后变成 256 个 token，计算效率提升很多。


### 四、时间步嵌入（Diffusion 条件）

扩散模型必须知道 “当前噪声程度”，所以需要时间步 embedding：

```
self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
```

这里有两个时间嵌入：

* 一个是 t
* 一个是 r（通常表示 t-r 或某种时间差）

它们都会被映射成和 hidden_dim 一样的向量，然后注入到 Transformer 里，告诉模型“现在在第几步扩散”。

这一步是 GPT 没有的，是 Diffusion 的结构。


### 五、条件投影

```
self.condition_embedder = nn.Linear(inner_dim, inner_dim, bias=True)
```

这个线性层的作用很简单：

把 encoder 输出的条件 embedding（文本、歌词、音色融合后的结果）投影到当前模型的 hidden_size 维度，方便后续做 cross-attention。

---

### 六、输出部分（去 Patch + 自适应归一化）

输出部分分两块：归一化 + 反卷积恢复时间长度。

```
self.norm_out = Qwen3RMSNorm(inner_dim, eps=config.rms_norm_eps)
```

这是 RMSNorm，和 Qwen3 保持一致。

接下来反 patch 操作：

```
self.proj_out = nn.Sequential(
    Lambda(lambda x: x.transpose(1, 2)),
    nn.ConvTranspose1d(
        in_channels=inner_dim,
        out_channels=config.audio_acoustic_hidden_dim,
        kernel_size=patch_size,
        stride=patch_size,
        padding=0,
    ),
    Lambda(lambda x: x.transpose(1, 2)),
)
```

这里用的是 `ConvTranspose1d`，也就是反卷积，把 [B, T//patch, hidden_dim] 恢复成：[B, T, audio_hidden_dim]，相当于把 patch token 再还原回原始时间长度。

---

### 七、自适应 scale-shift（DiT 才有）

```
self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
```

这是一个可学习参数，用来做 **scale-shift modulation**，也就是：

* scale（缩放）
* shift（平移）

它通常结合时间 embedding 或条件 embedding，用来调制输出层的归一化结果，是 DiT 里常见的做法。让时间步和条件直接控制输出层的分布形态。

---

因此，这个函数是把一个原本做“文本预测”的 Transformer，改造成一个“处理音频噪声”的扩散模型核心：

- 第一，它准备好了一个标准的 Transformer 主干（基于 Qwen3 结构）。这个部分就是多层注意力 + MLP，本质和 GPT 很像，只不过输出不再是 token 概率，而是连续向量。

- 第二，它把很长的音频序列“切成小块”（patch）。这样做的目的很简单：音频太长，直接喂给 Transformer 计算量太大，所以先压缩一下长度，变成更少的“块”再处理，效率更高。

- 第三，它加入了“时间步 embedding”。因为扩散模型每一步噪声程度不同，模型必须知道“现在噪声有多大”，否则不知道该往哪个方向预测。所以这里专门加了时间信息。

- 第四，它准备了一个接口，用来接收文本、歌词、音色等条件信息。也就是说，这个模型不仅看噪声，还能“听懂”条件，从而按要求生成音乐。

- 第五，最后把处理完的结果再“展开”回原始长度，相当于把压缩的 patch 再还原成完整的时间序列。

---

#### 最后

对于其他比较重要的代码，大家可以自行阅读：

modeling_acestep_v15_base.py 中的：
- 3、AceStepDiTLayer
- 4、TimestepEmbedding

ACE-Step-1.5 在实际运行中使用的是 [仓库链接](https://github.com/ace-step/ACE-Step-1.5/tree/main/acestep/models) 中 models/mlx 文件夹下的代码，可以主要阅读：
- dit_generate.py
- dit_model.py

补充：mlx 是 Apple MLX 推理版本

```
mlx/
dit_model.py
dit_generate.py
vae_model.py
```

MLX 是 Apple 的：Metal 加速的 ML 框架

- base/ ：训练用的 PyTorch
- mlx/ ：推理用的 MLX

所以生成算法在这里 mlx/ 目录下这 3 个文件：
- dit_convert.py 是权重转换
- dit_model.py 是 MLX 版 DiT
- dit_generate.py 是 diffusion sampling
