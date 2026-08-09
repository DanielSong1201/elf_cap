# 使用 ELF 实现 Audio Captioning 的可行性评估与工作计划

## 1. 项目目标

本项目计划将传统 Audio Captioning 系统最后的自回归文本生成阶段替换为
Embedded Language Flows（ELF）：模型从连续噪声出发，在冻结 T5 Encoder 的
文本 latent 空间中通过 ODE/SDE 流生成完整 caption latent，最后并行解码为
token。

近期目标是完成一个可验证的监督式 Audio Captioning MVP；中期目标是在保留
CLAP、projection-based decoding 和 Retrieval-Augmented Generation（RAG）的
前提下，实现 DRCap 风格的 text-only / zero-shot ELF captioner。

开发分工如下：

- 本地机器：代码开发、配置检查、单元测试、CPU 小张量测试和少量样本冒烟测试。
- RTX 4090 服务器：模型下载、特征预计算、完整训练、批量采样和指标评估。
- GitHub：只同步源码、配置、脚本和实验记录，不同步数据集、模型权重或生成产物。

## 2. 可行性结论

结论：技术上可行，但不能把传统 decoder 或 Vicuna 直接替换成现有 ELF
checkpoint。需要增加 CLAP-to-ELF 条件适配器，并修改 ELF 的训练数据接口，
使其能够接收外部 audio latent，而不仅是 T5 编码后的文本条件。

| 方面 | 结论 |
| --- | --- |
| 使用 ELF 生成 audio caption | 可行 |
| 直接将 CLAP embedding 输入现有 ELF | 不可行 |
| 复用 ELF Transformer、flow loss、ODE/SDE sampler 和 token decoder | 高度可行 |
| 从 ELF-B XSum checkpoint 初始化 | 可行，T5 latent 和 tokenizer 兼容 |
| 保留 DRCap 的 text-only / zero-shot 训练 | 可行，需要 CLAP adapter 和模态间隙处理 |
| 获得比自回归模型更低的延迟 | 尚不确定，需要真实测量 |

预估现有 ELF PyTorch 代码可复用约 60% 至 75%。新增工作主要集中在条件适配、
audio-caption 数据链路、AAC 指标和 zero-shot 训练协议。

## 3. 推荐模型结构

```text
Audio
  -> frozen CLAP audio encoder
  -> audio embedding Ea
  -> optional DRCap text-support projection
  -> text-like embedding E't
  -> CLAP-to-ELF adapter (D_clap -> K x D_t5)
  -> audio condition tokens ------------------------+
                                                       |
Retrieved captions -> frozen T5 -> retrieval tokens --+-> ELF flow
                                                           |
Gaussian noise --------------------------------------------+
                                                           |
                                                caption latent x_hat
                                                           |
                                                ELF token decoder
                                                           |
                                                     audio caption
```

目标 caption 始终使用冻结 T5 Encoder 转换为连续表示 `x0`。训练时只对目标
caption 区域加噪并计算 flow L2 与 token CE；audio/RAG 条件 token 始终保持干净。

### 3.1 CLAP-to-ELF Adapter

第一版使用轻量 MLP：

```text
CLAP embedding
  -> LayerNorm
  -> Linear(D_clap, K * D_t5)
  -> reshape(B, K, D_t5)
  -> learned position embeddings
```

建议初始配置：

- ELF backbone：ELF-B；
- T5 latent dimension：512（t5-small）；
- audio condition tokens：`K=8`；
- caption target length：48 或 64；
- CLAP 和 T5 全程冻结；
- 先训练 Adapter，再逐步解冻 ELF 后部 block。

### 3.2 RAG 条件

使用 CLAP embedding 检索 top-k captions，将检索结果通过冻结 T5 Encoder
编码为 retrieval tokens，再与 audio tokens 拼接。必须保留类似 DRCap 的
similarity selection，避免模型在训练阶段直接复制几乎相同的检索结果。

## 4. 两条训练路线

### 4.1 路线 A：监督式 MVP

```text
audio -> CLAP audio embedding -> adapter -> ELF -> ground-truth caption
```

优点是实现简单、容易验证模型是否真正使用音频；缺点是需要 audio-caption
配对数据，不属于严格 zero-shot。该路线用于建立技术基线和排除工程错误。

### 4.2 路线 B：DRCap 风格 zero-shot ELF

训练阶段只使用文本：

```text
caption -> CLAP text encoder -> Et -> adapter -----+
caption -> retrieval -> T5 retrieval tokens -------+-> ELF -> reconstruct caption
caption -> frozen T5 -> target latent x0 -----------+
```

推理阶段：

```text
audio -> CLAP audio encoder -> Ea
      -> projection onto text embedding support -> E't
      -> same adapter + RAG -> ELF -> caption
```

这样可以保留 text-only 训练，并在推理时将 CLAP text encoder 换成 audio
encoder。路线 B 是最终研究目标，但应在路线 A 的小规模过拟合成功后实施。

## 5. 实施阶段

### 阶段 0：建立可复现的开发骨架

任务：

1. 创建 Python 3.10/3.11 环境文件和固定依赖版本。
2. 将所需 ELF 模块复制或以明确依赖方式引入本项目，避免依赖本地相对路径。
3. 建立 `src/`、`configs/`、`scripts/`、`tests/` 目录。
4. 增加 CPU 小模型 fixture，确保本地不下载大模型也能执行 shape 测试。
5. 建立 4090 服务器启动脚本和环境检查脚本。

验收标准：新机器 clone 后可按 README 建立环境；本地 CPU 测试全部通过。

### 阶段 1：ELF caption-only 领域适配

任务：

1. 准备 AudioCaps/Clotho 的 caption-only 文本。
2. 使用与 ELF checkpoint 相同的 T5 tokenizer/encoder。
3. 从 ELF-B checkpoint 初始化，在短 caption 语料上继续训练 flow L2 + token CE。
4. 先用 100 至 1000 条 caption 做过拟合测试，再扩大到完整 caption corpus。

验收标准：能够从噪声生成自然的声音描述句子；EOS、长度和重复率正常；小数据
训练能够明显过拟合。

### 阶段 2：支持外部 latent 条件

新增 batch 接口：

```python
batch = {
    "condition_latents": ...,          # [B, K, D_t5]
    "condition_attention_mask": ...,   # [B, K]
    "target_input_ids": ...,           # [B, L]
    "target_attention_mask": ...,      # [B, L]
}
```

训练时构造：

```text
x0 = concat(condition_latents, target_t5_latents)
```

必须保证：

- 条件位置不加噪；
- L2 和 CE 仅在 target 区域计算；
- CFG dropout 能丢弃 audio/RAG 条件；
- ODE/SDE 每一步都恢复干净条件；
- decoder 对条件位置的输出不计损失。

建议先新增独立训练链路，不立即破坏原 ELF 入口：

- `src/models/audio_condition.py`
- `src/training/train_step_audio_caption.py`
- `src/train_audio_caption.py`
- `src/generate_audio_caption.py`

验收标准：shape/mask 单元测试通过，条件区域在完整采样过程中保持不变。

### 阶段 3：实现 CLAP Adapter 和监督式 MVP

任务：

1. 实现 Adapter 并将其作为模型子模块纳入 checkpoint。
2. 冻结 CLAP、T5 和 ELF，只训练 Adapter。
3. 若损失进入平台期，逐步解冻 ELF text projection、最后 2 至 4 个 block 和输出头。
4. 在 100 条 AudioCaps pair 上过拟合。
5. 执行条件打乱实验：随机打乱 audio embedding 后，指标必须显著下降。

验收标准：模型能够记忆小数据；生成结果随音频改变；打乱条件后性能明显下降。

### 阶段 4：接入 RAG

任务：

1. 建立 caption datastore 和 CLAP embedding 索引。
2. 为每条音频检索 top-k captions，初始 `k=3`。
3. 使用 T5 编码检索文本并与 audio tokens 拼接。
4. 比较 audio-only、RAG-only 和 audio+RAG。
5. 实现 similarity selection，防止复制检索文本。

验收标准：audio+RAG 在语义准确性或 AAC 指标上优于两个单条件基线。

### 阶段 5：实现 text-only zero-shot 训练

任务：

1. 训练时使用 `CLAP_text(caption)`，不读取对应音频。
2. 使用 Adapter + ELF 重建 caption。
3. 实现 DRCap 的 text embedding support 和 projection-based decoding。
4. 推理时使用 `CLAP_audio(audio)` 替换文本编码器。
5. 严格检查目标数据集音频和配对标注是否泄漏进训练。

验收标准：在严格 zero-shot 协议下完成 AudioCaps/Clotho 的 in-domain 和
cross-domain 评估。

### 阶段 6：采样、长度和速度优化

优先搜索：

- ODE steps：8、16、32、64；
- SDE steps：16、32；
- CFG：1、1.5、2、3；
- SC-CFG：1、2、3；
- audio tokens：4、8、16；
- caption max length：32、48、64。

增加 minimum length、EOS 统计、重复短语检测，以及可选的多候选生成和
CLAPScore reranking。

必须比较真实 wall-clock latency、吞吐量和峰值显存。由于 caption 通常较短，
32 至 64 次 ELF 全序列前向不一定比十几步自回归增量解码更快。

### 阶段 7：完整评估和消融

需要实现或接入：

- METEOR；
- CIDEr；
- SPICE；
- SPIDEr；
- FENSE；
- CLAPScore；
- 生成长度、重复率和推理速度。

实验矩阵：

| 实验 | Audio | CLAP 投影 | RAG | ELF | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| AR baseline | 是 | 可选 | 可选 | 否 | 原始自回归基线 |
| ELF caption LM | 否 | 否 | 否 | 是 | caption 语言能力 |
| ELF-RAG | 否 | 否 | 是 | 是 | 文本条件能力 |
| ELF-Audio | 是 | 否 | 否 | 是 | 音频条件能力 |
| ELF-Audio-RAG | 是 | 否 | 是 | 是 | 监督式完整模型 |
| ELF-ZeroShot | 仅推理 | 是 | 是 | 是 | 最终 zero-shot 模型 |

## 6. RTX 4090 实验策略

RTX 4090 通常有 24 GB 显存，因此默认以单卡可运行为硬约束：

- 首选 ELF-B，不从 ELF-M/L 开始；
- 使用 BF16；
- 开启 gradient checkpointing；
- 使用梯度累积获得所需有效 batch size；
- 预计算并缓存冻结的 CLAP/T5 features；
- PPL/FENSE 等大模型评估与训练分开运行；
- 训练中只保存必要 checkpoint，并在服务器磁盘保留最近若干份；
- 首轮关闭 `torch.compile` 完成功能验证，稳定后再测试其收益。

建议服务器实验顺序：

1. CPU/GPU shape smoke test；
2. 单 batch 前向和反向；
3. 100 样本过拟合；
4. 1% 数据短训练；
5. 完整训练；
6. 固定 checkpoint 的采样参数 sweep；
7. 独立执行完整指标评估。

## 7. 主要风险与应对

### CLAP 与 T5 latent 空间不一致

使用 Adapter 和 text-support projection；先通过 CLAP text embedding 训练对齐，再
切换到 audio embedding。

### 单个 CLAP 全局向量缺少时间信息

第一版用 global embedding 建立基线；若多事件描述较差，再引入 HTS-AT/CLAP
帧级特征并转换为多组 audio tokens。

### ELF 忽略音频条件

使用条件打乱、condition dropout、audio-only/RAG-only 消融和 embedding
敏感性测试，不仅观察训练 loss。

### 并行 token 解码出现重复或顺序异常

加强 CE decoder 分支训练，搜索采样步数和 CFG，增加 EOS/长度约束，并测试
多候选 reranking。

### Zero-shot 数据泄漏

记录 CLAP 预训练数据、caption datastore、text support 和所有目标数据 split；
严格区分 weakly paired 预训练与目标集人工 audio-caption pair。

## 8. 第一里程碑

第一个必须完成的研究里程碑是：

> 在 100 条 audio-caption pair 上，让 CLAP Adapter + ELF-B 稳定过拟合，并通过
> 打乱 audio condition 使性能显著下降，证明 ELF 确实使用了音频信息。

只有该里程碑完成后，才开始完整 AudioCaps 训练、RAG 和严格 zero-shot 实验。

