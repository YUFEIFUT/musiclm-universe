# MusicGen 开源代码分析

首先我们在 MusicGen GitHub 项目中找到 models/musicgen.py 源码。 

它负责加载模型、设置生成参数、准备 conditioning、调用 LM、拼接分段生成、解码为音频。  

它不会具体实现 attention、实现 transformer  、实现 loss  、实现 CFG 控制强度等等。这些主要的底层计算都在其他的文件。可以通过以下文件找到源码：

- models/lm.py
- modules/transformer.py
- models/encodec.py
- modules/conditioners.py
- models/genmodel.py

---

## 一、import

根据 musicgen.py 这个文件 最初的import部分，我们可以看到它导入了哪些方法和函数。

```python
from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition, StyleConditioner
````

从而也可以知道它的源码结构大概是：

```
musicgen.py
   │
   ├── BaseGenModel 父类
   ├── LMModel 语言模型：自回归预测音频token。
   │      └── transformer.py
   │             └── attention
   │
   ├── CompressionModel 压缩模型，其实就是编码音频token。
   │      └── encodec implementation 音频编码器
   │
   └── Conditioners 调度器。看prompt是文字就处理文字给模型、看输入是旋律就处理旋律给模型。
```

---

## 二、定义 MusicGen 的核心模型类

MusicGen 继承自 BaseGenModel 底层通用生成框架，MusicGen 是对它的具体实现 + 参数封装

```python
class MusicGen(BaseGenModel):
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default duration
```

这段代码定义了 MusicGen 的核心模型类，并在初始化时设置了一个默认生成时长 15 秒。

1、输入参数包括 模型名、压缩模型是哪一个、语言模型是哪一个、可选：生成音频时长

2、super().**init** 调用父类构造函数：把所有核心组件交给 BaseGenModel 去初始化。

父类 BaseGenModel 做什么？

- 保存模型名
- 保存压缩模型
- 保存语言模型
- 设置最大生成长度

我们后面如何使用这个函数？

```python
model = MusicGen(
    name="musicgen-small",
    compression_model=encodec_model,
    lm=transformer_model
)
```

像这样，实现了一个MusicGen类，命名为model。即可使用这个MusicGen类（就是这个model）来生成音乐：

```python
audio = model.generate(["a happy piano melody"])
```

当然它的内部流程是：（可以自己去找 BaseGenModel 类的定义源码，在 genmodel.py 里）

- duration = 15 秒（默认）
- LM 生成对应 15 秒长度的离散 token
- CompressionModel 解码成音频
- 返回 tensor 返回张量

---

## 三、get_pretrained 自动下载并加载对应的预训练模型（语言模型 + 音频压缩模型）

并返回一个可直接生成音乐的 MusicGen 实例。

```python
@staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """
        Return pretrained model, we provide four models:
        - facebook/musicgen-small (300M), text to music,
          # see: https://huggingface.co/facebook/musicgen-small
        - facebook/musicgen-medium (1.5B), text to music,
          # see: https://huggingface.co/facebook/musicgen-medium
        - facebook/musicgen-melody (1.5B) text to music and text+melody to music,
          # see: https://huggingface.co/facebook/musicgen-melody
        """
        # 中间代码略：选 GPU cuda
        # 中间代码略：if name == 'debug'有一个 debug 模式（单元测试专用）
        # 中间代码略：兼容旧模型名字

        lm = load_lm_model(name, device=device) # 加载语言模型（LM）

        compression_model = load_compression_model(name, device=device) # 加载压缩模型（音频解码器）

        if 'self_wav' in lm.condition_provider.conditioners: # 对于 musicgen-melody 这类支持 melody 输入的模型
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True # 在推理阶段对齐长度
            lm.condition_provider.conditioners['self_wav']._use_masking = False # 不使用训练时的 mask 机制
 
        return MusicGen(name, compression_model, lm) # 返回最终模型
```

1、@staticmethod 这是一个 静态方法。

因此我们可以直接调用 而不需要先实例化类 MusicGen：

```python
model = MusicGen.get_pretrained("facebook/musicgen-small")
```

2、load_lm_model() 加载语言模型（LM）

3、load_compression_model(  加载压缩模型（音频解码器）

4、特殊处理 self_wav 条件器：（对于 musicgen-melody 这类支持 melody 输入的模型）

在推理阶段对齐长度，不使用训练时的 mask 机制

目的是：让 melody 条件输入在推理时更稳定。

5、返回最终模型

```python
return MusicGen(name, compression_model, lm)
```

现在我们就得到了一个：

已加载权重、已准备好推理、已放到 GPU 的模型实例。

---

## 四、设置 模型 在 生成音乐时的 采样策略和控制参数。

```python
    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False, extend_stride: float = 18,):
        """
        Set the generation parameters for MusicGen.
        """
        # 略：assert extend_stride < self.max_duration（滑动扩展时长 小于 一次最大生成时长）
        
        self.extend_stride = extend_stride    # 每次扩展 多少秒
        self.duration = duration              # 总生成时长 多少秒
        self.generation_params = {            # 保存参数（就是采样策略）这些参数会在 LM 采样时使用。
            'use_sampling': use_sampling,  # 随机采样，而不是更稳定的 argmax（最大概率）
            'temp': temperature,           # 温度 控制随机性
            'top_k': top_k,                # 每一步只在概率最高的 250 个 token 里采样。
            'top_p': top_p,                # 选前面概率累计达到 0.9（就是概率90%）的 token 集合。

            'cfg_coef': cfg_coef,            # 用于 text to music 的 MusicGen
            'cfg_coef_beta': cfg_coef_beta,  # 用于 melody + text to music 的 MusicGen

            'two_step_cfg': two_step_cfg,  # 是否分两次 forward 做 cfg，更稳定些？
        }
```

1、滑动扩展时长 extend_stride 到底是什么？

它会有用在“所需生成 超过 一次生成最大长度”时：

MusicGen 一次最多能生成 self.max_duration 秒（例如 30 秒）。

如果你想生成更长的音乐，比如 60 秒：

模型会采用一种 滑动扩展生成 的方式。

2、cfg：分类器自由引导系数

cfg_coef=3.0  控制 文本对生成的“控制强度”

公式是：

```
output = uncond + cfg * (cond - uncond)
输出   = 无条件 + cfg * (条件 - 无条件)
```

cfg 越大，越贴近文本描述，但可能音质变差

--

cfg_coef_beta = None  控制 旋律+文本对生成的“控制强度”

用于：melody 模型（melody + text to music 的 MusicGen）

作用：同时平衡 text 条件 和 audio 条件，只有在 text+melody 模型里才用。

#### 示例输入1：默认参数

```python
model.set_generation_params()
```

则实际上内部保存：

```
{
 'use_sampling': True,
 'temp': 1.0,
 'top_k': 250,
 'top_p': 0.0,
 'cfg_coef': 3.0,
 'two_step_cfg': False,
 'cfg_coef_beta': None,
}
```

按照 这些默认参数 配置模型，生成 30 秒音乐。

#### 示例输入2：换换参数，生成更有创造性的音乐

```python
model.set_generation_params(
    temperature=1.5,
    top_k=500,
    duration=20
)
```

效果：更随机 更实验风格 生成 20 秒

#### 这里再讲一下 CFG：

CFG 是用于 Diffusion 模型的。但是 MusicGen 作为一个完全使用 Transformer 自回归的模型，为什么用 CFG？

这就是 CFG for 自回归模型。

“classifier-free guidance for autoregressive models”

意思就是：在自回归生成模型里，用无条件输出和有条件输出做差值放大，来增强条件控制能力。**让模型更听话。**

#### cond / uncond 是啥？“有条件时的 logits 分数矩阵” 和 “无条件时的 logits 分数矩阵”

#### cond logits  就是 “带提示词输入的得到的 logits 分数矩阵” 有条件 logits

带文本输入：“epic orchestral battle music”

模型会倾向生成史诗交响风。

经过模型得到 logits 分数矩阵：每个 token 的概率多大的分数 cond_logits

#### uncond logits 无条件 logits

输入空文本：""   （什么都不写）

模型只根据“音乐本身的统计规律”生成。

得到这次的 logits 分数矩阵 uncond_logits

#### MusicGen 的做法是这样：

在 sampling 时：

输入带文本 → 得到 cond logits
输入空文本 → 得到 uncond logits

做：

```
logits = uncond + scale * (cond - uncond)
```

再去 softmax + sampling

- uncond = “自然生成”
- cond = “受文本影响的生成”
- cond - uncond = “文本带来的改变方向”

所以：这是在“放大文本的影响”

这就叫 “CFG for 自回归模型”

这个 CFG 的值比如说 3.0，那就是 指导强度（guidance strength）是 3.0，而 (cond - uncond) 是 指导方向。

```
logits = uncond + scale * (cond - uncond)
```

这就是：根据 CFG 的值是多大，来放大“文本带来的改变方向”。

---

## 五、模型的输入预处理函数

```python
@torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """
        准备模型输入数据。

        参数:
            descriptions (字符串列表): 用作文本条件控制的字符串列表。
            prompt (torch.Tensor): 用于音频续接生成的一批波形数据。
            melody_wavs (torch.Tensor, 可选): 用作旋律条件控制的一批波形数据
                （默认值为 None）。
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens
```

1、descriptions 是文本描述（Text Prompt），比如：

```
descriptions = [
    "A calm piano melody with soft strings",
    "Upbeat electronic dance music with strong bass"
]
```

第一段音乐：温柔钢琴 + 弦乐
第二段音乐：电子舞曲 + 重低音

会把description列表的这些文字转换成 text conditioning文本条件给到模型：

```
ConditioningAttributes(text={'description': description})
```

模型会用 description 作为“文本控制条件”。

2、prompt 是音频续写的开头音频（Audio Prompt）

给模型一段已有音频，让它接着往下生成

假设我们有 5 秒音频：

```
prompt.shape = (2, 1, 160000)
```

batch = 2 两段音频 ；单声道；160000 采样点（假设 32kHz ≈ 5秒），那么：

第一段音频：钢琴开头
第二段音频：鼓点开头
模型会“继续写”

如果没有开头音频？那就不续写。

```
prompt = None
```

说明：从零开始生成

3、melody_wavs 是 旋律条件（Melody Conditioning）

本质是一个 waveform 列表

它用于：提供旋律，让模型围绕这个旋律生成音乐

注意：只有 melody 模型才支持它。

比如：

```
melody_wavs = [
    torch.randn(32000),  # 第一段旋律 1 秒
    torch.randn(48000)   # 第二段旋律 1.5 秒
]
```

意思是：

第一首歌围绕 melody_wavs[0] 的旋律创作
第二首歌围绕 melody_wavs[1] 的旋律创作

注意 这个旋律是完全按照给定音频来，这是MusicGen-Melody模型。

至于MusicGen-Style模型，是参考给定音频的风格，生成相似风格的音乐，而不是完全按照给定音频的旋律来。

#### 举一个完整调用例子：

```python
# 按 文本 和 参考旋律 生成的例子：有 text descriptions文本描述，有“旋律参考音频” 没有“需要续写的音频”

descriptions = [
    "Epic orchestral trailer music"
]

prompt = None

melody_wavs = [
    torch.randn(32000)
]
```

意思是根据给定旋律，生成一段史诗管弦乐

#### 再举一个续写例子

```python
# 续写：有 text descriptions，audio prompt，没有“旋律参考音频”

descriptions = [
    "Lo-fi hip hop beat"
]

prompt = torch.randn(1, 1, 160000)  # 已有 5 秒音频

melody_wavs = None
```

意思是接着这 5 秒 lo-fi 音乐继续生成

#### 再举一个例子：如果三个都用？

```
descriptions = ["Jazz piano trio"]
prompt = 已有钢琴片段
melody_wavs = 给定旋律
```

模型会综合“文本风格、开头音频、指定旋律”一起生成。

#### 输入输出示例（续写） ：只有文本 + audio prompt（无 melody）

输入

```
descriptions = [
    "A happy pop song with guitar",
    "Sad piano ballad"
]

prompt = torch.randn(2, 1, 32000)  # 2条音频，每条1秒(假设16kHz采样)

melody_wavs = None
```

#### 函数内部发生什么？

第一步：构造 attributes

```
attributes = [
    ConditioningAttributes(text={'description': "A happy pop song with guitar"}),
    ConditioningAttributes(text={'description': "Sad piano ballad"})
]
```

此时 attributes 结构大概是：

```
[
  {
    text: {"description": "A happy pop song with guitar"},
    wav: {}
  },
  {
    text: {"description": "Sad piano ballad"},
    wav: {}
  }
]
```

第二步：因为 melody_wavs=None，所以给每个样本加一个“空melody”：

```
attr.wav['self_wav'] = WavCondition(
    torch.zeros((1, 1, 1)),
    torch.tensor([0]),
    sample_rate=[16000],
    path=[None]
)
```

所以现在 attributes 变成：

```
[
  {
    text: {"description": "A happy pop song with guitar"},
    wav: {
      "self_wav": WavCondition(
          wav = tensor([[[0.]]]),
          length = tensor([0]),
          sample_rate = [16000],
          path = [None]
      )
    }
  },
  {
    text: {"description": "Sad piano ballad"},
    wav: {
      "self_wav": WavCondition(
          wav = tensor([[[0.]]]),
          length = tensor([0]),
          sample_rate = [16000],
          path = [None]
      )
    }
  }
]
```

第三步：处理 prompt

```
prompt = prompt.to(self.device)
prompt_tokens, scale = self.compression_model.encode(prompt)
```

假设 encode 后：

```
prompt_tokens.shape = (2, 50)  # 每条音频被压缩成50个token
scale = None
```

最终输出

```
attributes, prompt_tokens
```

1、attributes：

长度=2 的 ConditioningAttributes 列表，每个包含:

text.description
wav.self_wav (空melody占位)

2、prompt_tokens：音频张量 Tensor shape: (2, 50)

---

## 六、支持长音频生成的“分段滑动生成策略”

```python
def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:

        """
        基于音频提示信息和/或条件信息生成离散的音频令牌。

        参数:
            attributes (ConditioningAttributes 列表): 用于生成任务的条件信息（文本/旋律）。
            prompt_tokens (torch.Tensor, 可选): 用于音频续接生成的音频提示令牌。
            progress (bool, 可选): 用于显示生成过程进度的标识位，默认值为 False。
        返回值:
            torch.Tensor: 生成的音频数据，形状为 [B, C, T]，其中 T 由生成参数定义。
        """

        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
```

#### 情况一：总时长 小于等于 单次最大时长

直接调用 lm.generate

#### 情况二：总时长 大于 单次最大时长

分段生成、重叠、滑动窗口拼接

比如举一个例子：

```
max_duration = 30 秒
duration = 70 秒
extend_stride = 10 秒
frame_rate = 50 token/s
```

- max_gen_len = 1500 tokens
- stride_tokens = 500 tokens
- 重叠 1000 token

每一轮输入长度 = 1500，正前进 = 500，重叠 = 1000

而 overlap 是防止音乐边界断裂。

---

## 补充：为什么直接 lm.generate(max_gen_len=4500) 会报错？

Transformer 的限制来自三个地方：

- 位置编码长度
- attention mask 尺寸
- KV cache 分配大小

只要其中一个是固定的，就不能无限增长。

---

## 七、最后一个步骤：文本 + 旋律条件生成音乐

```python
   def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
```

这一部分代码即为结合之前定义的各个函数，组成完整的生成流程。主要部分我们只需要看：

```python
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
```


**首先：重采样 + 声道转换**

假设用户输入音频：

- 44.1kHz
- stereo

模型内部会执行：

- 32kHz
- mono

自动转换为这样单声道 32KHz的音频。

**接下来：构造 ConditioningAttributes。** 调用我们之前的 _prepare_tokens_and_attributes 函数：构造 text conditioning，构造 wav conditioning，但不生成 prompt_tokens。

（这里说明当前不支持音频续写，必须纯文本 + 旋律生成）

**然后：真正生成 token。** 调用我们之前的 _generate_tokens(attributes, prompt_tokens, progress) 函数：注意这里真正经过 Transformer、Attention 流程。

**最后：解码为音频。** 调用 generate_audio(tokens) 解码我们最终生成的音频。输出：[B, C, T]

---

musicgen.py 完整流程分析完毕。大家可以自行去看函数细节的源码：

- models/lm.py（LMModel.generate 和 LMModel.forward）
- modules/transformer.py
- models/encodec.py
- modules/conditioners.py
- models/genmodel.py

