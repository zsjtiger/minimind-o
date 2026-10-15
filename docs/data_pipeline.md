# 多模态训练数据：选型、配比与预处理

本文说明训练数据的来源、混合配比与统一的预处理流程。代码位于 `baize_v/data/prep/`，配置在 `configs/data_mix.json`，入口是 `scripts/prepare_data.py`。

## 当前状态（重要）

- 本环境无法访问 Hugging Face，**没有下载任何数据集，也没有核对过各数据集的字段名**。转换器的列名都以参数形式暴露，首次使用前请用样本核对。
- 配置中的原始数据路径是约定路径，数据不存在时对应数据源会被跳过。
- 配比是初始草案，需要用真实数据做消融后再调整。

## 统一样本格式

所有数据源都转换为同一种 parquet（`baize_v/data/prep/schema.py`），与 `OmniDataset` 读取的列一致：

| 列 | 说明 |
| --- | --- |
| `conversations` | 多轮对话 JSON；图像用 `<image>` 标记位置 |
| `question_audios` | 每个 user 轮次的 16 kHz 单声道 PCM16 wav 字节 |
| `answer_audios` | 每个 assistant 轮次的 **Mimi 8 层码本**（仅语音输出训练需要） |
| `image_bytes` | 图像字节，模型只取第一张 |
| `ref_audios` / `spk_emb` | 音色条件，可为空 |
| `source` / `modality` | 数据源名与模态，用于统计与配比 |

## 数据源选型

按模态整理的候选数据集，许可证与用途请以各数据集的官方页面为准：

| 模态 | 数据集 | 规模（你提供的信息） | 用途 | 注意 |
| --- | --- | --- | --- | --- |
| 图文 | COCO Captions | 约 33 万张图，每张 5 句英文描述 | 图像描述 | 每张图取 `captions_per_image` 句，避免过多重复图 |
| 图文 | Visual Genome | 约 10.8 万张图，区域描述与场景图 | 区域描述、关系问答 | 当前转换器未实现 |
| 图文 | LAION-400M / 5B | 网络抓取图文对 | 大规模预训练 | 版权与质量争议，**不建议直接用于训练**，需先过滤 |
| 图文 | CC3M / CC12M | 网页筛选图文对 | 预训练 | 同上，需确认许可 |
| 图文 | VQAv2 / GQA / OK-VQA | 视觉问答 | 视觉问答；GQA 偏组合推理 | 使用 `vqa_jsonl` 转换；OK-VQA 需外部知识 |
| 图文 | DocVQA / InfoVQA / ChartQA / TextVQA | 文档、图表、图中文字问答 | 文档与图表理解 | 文字密集，图像分辨率要求高 |
| 图文 | ShareGPT4V / LLaVA-Instruct | 高质量描述与指令数据 | 描述能力、指令跟随 | 多为 GPT-4 生成，注意生成模型的许可条款 |
| 图文 | MMMU / MMBench / MME | 综合评测集 | **仅用于评测** | 不要混入训练数据，避免数据污染 |
| 视频 | MSR-VTT | 约 1 万段视频、20 万句描述 | 视频描述与检索 | 当前取中间帧作为代表图 |
| 视频 | ActivityNet Captions | 长视频时序片段描述 | 时序定位与描述 | 未实现；需要多帧支持 |
| 视频 | HowTo100M | 约 1 亿旁白片段 | 弱监督预训练 | 旁白噪声大，需过滤 |
| 视频 | Kinetics / HMDB51 / UCF101 | 动作识别 | 视觉分类，不含语言 | 本项目不使用 |
| 语音 | VoiceAssistant-400K | 约 47 万条，问题为语音，回答为文本与 SNAC 语音 token | 语音输入 → 文本输出 | 见下文“语音 token 不兼容” |
| 语音 | UltraChat-300K-SLAM-Omni | 约 31 万条（多轮，最多 7 轮），问题为语音，回答为文本与 CosyVoice 语义 token | 语音输入 → 文本输出（单轮） | 同上；多轮样本当前只取单轮 |
| 语音 | AudioSet / AudioCaps / Clotho | 音频事件与描述 | 音频理解 | 当前未实现 |
| 语音 | LibriSpeech / Common Voice | 语音识别语料 | ASR 训练与评测 | 偏识别，不适合对话训练 |
| 音视频 | VGGSound / AVSpeech | 音视频同步 | 音视频对齐 | 未实现 |

### 语音 token 不兼容（需要特别注意）

本模型的语音输出（talker）以 **Mimi 8 层码本**（每层 2048 码字）为训练目标，见 `baize_v/tokens.py`。而 VoiceAssistant-400K 的回答使用 SNAC 语音 token，UltraChat-SLAM 的回答使用 CosyVoice 语义 token，**二者都不能直接作为 talker 的监督目标**。

因此这两个数据集只能用于“语音输入 → 文本输出”的训练（`answer_audios` 留空，talker 损失为 0）。要获得语音输出的监督，需要：

1. 保留这些数据的文本回答，用 Mimi 重新合成语音再编码（需要 TTS，成本高，且合成音色会影响分布）；或
2. 使用已有的 Mimi 码本数据（如现有的 `sft_a2a` 类数据）做语音输出训练。

## 初始配比

`configs/data_mix.json` 中按样本数计的权重：

| 数据源 | 模态 | 权重 | 理由 |
| --- | --- | --- | --- |
| sft_text | 文本对话 | 0.35 | 语言能力是基座，占比最高 |
| coco_captions | 图文 | 0.15 | 基础的视觉-语言对齐 |
| vqa | 图文 | 0.12 | 视觉问答与推理 |
| doc_chart_vqa | 图文 | 0.08 | 文档与图表，文字密集，样本量小 |
| msrvtt | 视频-文本 | 0.10 | 视频内容用一张代表帧，覆盖视频描述 |
| voice_assistant | 语音输入 | 0.08 | 语音理解，无语音输出监督 |
| ultrachat_slam | 语音输入 | 0.12 | 语音理解，单轮样本 |

语音输入合计 20%：这类数据只训练理解，占比过高会挤占文本与图文能力。合计占比合理性需要用验证集的各模态指标验证。

## 预处理流程

每个数据源经过同一条流水线（`run_builder`）：

1. **转换**：按数据源格式生成统一记录（`builders.py`）。
2. **文本清洗**：NFKC 归一化、去除控制字符、统一换行与空白。
3. **文本过滤**：回答长度不在 `[min_chars, max_chars]` 范围内则丢弃。
4. **图像标准化**：转 RGB，长边缩放到 512 以内，重编码为 JPEG；过小（短边 < 64）或损坏的图像丢弃。
5. **音频标准化**：转单声道，重采样到 16 kHz，PCM16 wav；时长不在 [0.3, 30] 秒内丢弃。
6. **校验**：对话以 assistant 结尾、音频与轮次数量一致、语音码长度为 8 的倍数等。
7. **去重**：按（对话, 图像, 音频）的 SHA-1 去重。
8. **流式写入**：按 256 条一批写入 parquet，不把整个数据源放进内存。

视频与多帧：模型每个样本只接收一张图。视频取中间一帧，抽帧命令示例：

```bash
ffmpeg -i video.mp4 -vf fps=1 dataset/raw/msrvtt/frames/<video_id>/%04d.jpg
```

## 混合

`scripts/prepare_data.py` 读取配置，构建 enabled 的数据源，再按权重混合：

- 目标数量 = `total × weight / Σweight`。
- 数据源不足时有放回地过采样，报告中标记 `(过采样)`。
- 固定 `seed`，混合结果可复现（测试覆盖）。
- 输出 `<output>.report.json`，记录每个数据源的目标、可用量与实际占比。

```bash
uv run python scripts/prepare_data.py --config configs/data_mix.json
uv run python scripts/prepare_data.py --config configs/data_mix.json --only_mix   # 复用已构建的分源文件
```

## 许可证与合规

- 不同数据集的许可证差异很大。LAION、网络抓取数据、由商用模型生成的指令数据都有使用限制，商用前必须逐一确认。
- 评测集（MMMU、MMBench、MME 等）不应进入训练数据。
- 构建后的数据不随仓库分发，只保存在 `dataset/` 下（已在 `.gitignore` 中忽略）。

## 已知限制

- 当前转换器：COCO captions、VQA 风格 jsonl、视频中间帧、语音输入 parquet、纯文本 jsonl、已有 parquet 直通。Visual Genome、LAION/CC、ActivityNet、音频事件数据集未实现。
- 语音输出训练仍依赖 Mimi 码本数据（见上文）。
- 混合时整个分源文件会读入内存；数据量非常大时需要改为分批混合。
- 去重只针对完全相同的样本，不处理近似重复（如 LAION 中的近重复图）。
