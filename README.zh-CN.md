# CoBind-Count

[English](README.md) | 简体中文

**基于 CoGenT 的颜色–形状绑定与精确计数诊断基准。**

CoBind-Count 用于评测文生图模型的**同类多实例属性绑定与精确计数**。当一张图包含多个相同形状的物体时，模型需要生成各实例的正确颜色，并满足每种颜色×形状组合的数量。

例如，目标是“两个红球、一个蓝球、一个蓝方块”，生成“一个红球、两个蓝球、一个红方块”时，总物体数、颜色总数和形状总数都正确，但具体绑定错误。CoBind-Count 比较颜色×形状的联合数量，因此不会将该图判为完全匹配。

| 项目 | 标识 |
|---|---|
| 数据版本 | `cobind-count-v1.0` |
| 问答协议 | `cobind-count-qwen-v1` |
| 评分协议 | `cobind-count-score-v1` |

每个模型报告 **Full Match、Count Acc、联合计数 F1** 三个单值，并固定展示**总计、3 个物体、4 个物体、5 个物体、6 个物体**五组结果。

## 范围与来源

数据来源为 [CLEVR-CoGenT](https://cs.stanford.edu/people/jcjohns/clevr/) 的参考图和场景标注。本基准提供文生图提示词、固定数据划分、联合计数目标、诊断标签和 Qwen 评测工具。

核心评分覆盖 **count / color / shape**。提示词也包含大小和材质描述，但这些属性不参与核心评分；提示词不指定空间关系。`core_full_match` 表示计数、颜色和形状要求全部满足，不代表整条提示词的所有属性均正确。

生成模型只接收提示词。参考图用于标注追溯和评测器校准，不作为生成条件，也不要求生成图复现参考图的布局或相机角度。评测对象是无序的物体集合。

## 数据划分

| 划分 | 提示词 | 条件 A | 条件 B | 3 个物体 | 4 个物体 | 5 个物体 | 6 个物体 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,000 | 1,000 | 0 | 0 | 335 | 346 | 319 |
| development | 200 | 100 | 100 | 50 | 50 | 50 | 50 |
| test | 800 | 400 | 400 | 200 | 200 | 200 | 200 |
| 合计 | 2,000 | 1,500 | 500 | 250 | 585 | 596 | 569 |

开发集每个 A/B×物体数单元为 25 条，测试集每个单元为 100 条。正式评测使用 800 条测试提示词；训练集用于模型微调，成绩基于测试集计算。开发集中的 100 条另用于参考图校准，不增加独立场景数。

训练集仅包含 4–6 个物体的场景。3 个物体的测试分组用于检验模型或方法在该训练集上微调后，是否仍保留生成更少物体组合的能力。

仓库提供三个划分的提示词和标注，但参考 PNG **仅包含 800 张测试集图片**。训练集和开发集的参考 PNG 不包含在仓库中。

全部 2,000 条记录的场景 ID 和提示词文本互不重复。场景 ID、提示词文本、参考图哈希及核心颜色×形状多重集目标在不同划分之间均无重叠。数据来源、选择记录和内容哈希位于 `provenance/`。

条件 A/B 使用 CoGenT 的组合规则：

| 条件 | cube 颜色 | cylinder 颜色 | sphere 颜色 |
|---|---|---|---|
| A | gray, blue, brown, yellow | red, green, purple, cyan | 全部 8 色 |
| B | red, green, purple, cyan | gray, blue, brown, yellow | 全部 8 色 |

### 诊断标签

“重复组合”指同颜色、同形状的物体不止一个。以下四组互斥，覆盖全部记录：

| 标签 | 同形状不同颜色 | 重复组合 | train | development | test |
|---|---|---|---:|---:|---:|
| `basic` | 否 | 否 | 0 | 10 | 34 |
| `repeated_only` | 否 | 是 | 0 | 20 | 86 |
| `same_shape_diff_colors` | 是 | 否 | 637 | 103 | 416 |
| `same_shape_diff_colors_repeated` | 是 | 是 | 363 | 67 | 264 |

测试集中 680 条含同形状不同颜色，350 条含重复组合；另外有 298 条带有“同颜色不同形状”标签，该标签可与四组重叠。A/B 和诊断标签供分析使用，默认成绩表按总计和物体数分级。

全部 1,000 条训练提示词同时包含“同形状不同颜色”和“同颜色不同形状”。开发集和测试集还包含不具备这些诊断特征的场景。

## 数据格式

```text
CoBind-Count/
├── data/             # train、development、test JSONL
├── prompts/          # 供生成程序读取的精简 JSONL
├── images/           # 800 张测试集参考 PNG
├── evaluation/       # Qwen 问答、评分与报告工具
├── calibration/      # 开发集参考记录与目标对
├── provenance/       # 数据来源、选择设置与图像哈希
├── results/          # 800 张测试参考图的评测结果
├── questions.json    # 29 个计数问题
├── statistics.json   # 数据分布
├── subsets.json      # 诊断子集 ID
├── manifest.json     # 版本、划分、固定种子与文件 SHA256
└── build_dataset.py  # 从固定划分记录重建元数据
```

`data/*.jsonl` 每行包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 字符串场景 ID，不能用行号替代 |
| `prompt` / `text` | 生成提示词，两者内容相同 |
| `condition` / `object_count` | A/B 条件和目标物体数 |
| `objects` | 原始物体级属性与场景标注 |
| `target_joint_counts` | 全部 24 格颜色×形状数量，包含 0 |
| `core_target_counts` | 联合计数目标的稀疏形式 |
| `core_target_id` | 核心目标标识 |
| `diagnostic_group` / `difficulty` | 诊断分组及难度标签 |
| `reference_image` / `image_sha256` | 相对本包的参考图路径和内容哈希 |

例如 `CLEVR_valB_013250` 的提示词为：

```text
A 3D scene with three objects: a large brown rubber cylinder; a large gray rubber cylinder; a large red metal cube.
```

核心目标是 `brown/cylinder=1, gray/cylinder=1, red/cube=1`，其余 21 格为 0。

`prompts/*.jsonl` 包含生成所需的 `id`、`prompt`、条件、物体数、目标标识和种子列表。仓库中的 800 张测试集参考 PNG 使用 CLEVR-CoGenT 原始图像内容。训练集和开发集记录也包含参考图路径及哈希，对应 PNG 可从 CLEVR-CoGenT 取得。

## 评分协议

每张图片独立回答 29 个问题：总数 1 题、各形状数量 3 题、颜色×形状联合数量 24 题、无法归入词表的其他物体数量 1 题。Qwen 只看图片和当前问题，不接触生成提示词、目标答案、方法名或其他题的回答。

颜色词表为 gray、red、blue、green、brown、purple、cyan、yellow；形状词表为 cube、cylinder、sphere。gray 包含 silver，cyan 包含 turquoise/teal，box 视为 cube，ball 视为 sphere。阴影、反射和背景不计作物体。无法明确分类的前景物体计入 `other`。

| 指标 | 定义 |
|---|---|
| `core_full_match` | 全部 24 格正确、`other=0`、总数及各形状数量正确，且回答无内部矛盾 |
| `count_acc` | 总数和 3 个形状数量同时正确 |
| `joint_count_f1` | `2×Σ min(目标格数, 预测格数) / (目标总数 + 预测24格总数 + other)` |

完整且可解析的回答均产生分数。若总数、形状数量和联合数量相互矛盾，Full Match 为 0，Count Acc 与 F1 继续由观察计数计算；矛盾信息作为诊断记录。

例如目标为 2 个红球、1 个蓝球，观察为 1 个红球、2 个蓝球，则 Full Match 为 0，F1 为 2/3。主指标不对 24 格的逐格正确率取均值，避免大量零目标格抬高分数。

每项主指标是可计分图片的逐图均值。缺图、缺题或不可解析回答使整张图的三项分数为 `None`，不计入该指标均值；报告同时给出可计分数量、未计分数量及覆盖率。完整成绩要求全部预定图片均可计分。`evaluation_valid` 记录回答自洽性，不能用它筛除 Count Acc 或 F1 的矛盾样本。

### 五项分级

| 分组 | 测试提示词 | 参考图数量 | 每个生成模型图片数 |
|---|---:|---:|---:|
| 总计 | 800 | 800 | 3,200 |
| 3 个物体 | 200 | 200 | 800 |
| 4 个物体 | 200 | 200 | 800 |
| 5 个物体 | 200 | 200 | 800 |
| 6 个物体 | 200 | 200 | 800 |

分组依据目标物体数。每条测试提示词使用 **101、202、303、404** 四个固定种子；完整评测时，逐图均值等价于先对每条提示词的四个种子取平均，再对提示词等权平均。每张参考图只评一次，参考图成绩用于衡量评测器与标注的一致程度。

## 使用方法

```bash
git clone https://github.com/XHQuest/CoBind-Count.git
cd CoBind-Count
```

### 准备生成图片

从 `prompts/test.jsonl` 读取 800 条提示词，每条提示词按四个固定种子各生成一张图片。记录模型权重标识、生成步数、分辨率和引导系数，每个待评测模型及配置使用独立的模型标签。

图片清单为 UTF-8 JSONL，每行一张图：

```json
{"model":"model_a","id":"CLEVR_valB_013250","seed":101,"image":"images/model_a/CLEVR_valB_013250_101.png"}
```

`image` 为绝对路径或相对该清单的路径。正式生成评测需完整覆盖 800 条提示词×4 个种子；不同模型可以放在同一清单中。参考图清单使用对应参考图路径，每条记录只出现一次，`seed=101` 仅用作索引标识。

### Qwen 计数

使用兼容 vLLM 的本地 Qwen3.5-9B 环境，在本目录执行：

```bash
python evaluation/run_qwen_eval.py \
  --split test \
  --images-manifest /path/to/generated_images.jsonl \
  --model-path /path/to/Qwen3.5-9B \
  --output outputs/test_qwen
```

固定设置为 BF16、temperature=0、judge seed=20261103、thinking=False、最多输出 32 token、图片像素范围 512²～1024²。参考图评测显式传入 `--seeds 101`；该单种子任务与四种子生成评测分别标记。

相同命令支持断点续评，只处理缺失问题。运行工具提供 tqdm 进度条、目录锁和逐块持久化，记录数据、图像、代码、模型与运行配置。无效回答不自动重试，原始回答不作语义修正。

### 生成报告

报告工具读取已有问答，在 CPU 上评分：

```bash
python evaluation/summarize.py --output outputs/test_qwen
python evaluation/summarize.py \
  --source /path/to/observations \
  --output /path/to/report
```

主要产物为 `image_scores.jsonl`、`summary.json`、`summary.csv`、`summary.md` 和评分来源记录。详细输入、续评与输出规则见 [evaluation/README.txt](evaluation/README.txt)。

[800 张参考图的评测结果](results/RESULTS.md)按总计和物体数分级报告评测器与场景标注的一致程度。详细成绩提供 [Markdown](results/reference_test800/summary.md)、[CSV](results/reference_test800/summary.csv) 和 [JSON](results/reference_test800/summary.json) 格式。

## 校准目标对

从开发集固定选取 100 张参考图：A/B 各 50，3/4/5/6 个物体各 25。正例使用原始联合计数；负例包含数量偏差和保持边际数量的绑定交换。

仓库提供校准记录和目标对，不包含这 100 张开发集 PNG。运行图像校准前，请从 [CLEVR-CoGenT 官方数据](https://cs.stanford.edu/people/jcjohns/clevr/) 取得对应图片，按 `calibration/references.jsonl` 中各记录的 `reference_image` 相对路径放到仓库内。记录中的 `image_sha256` 对应所需原始图像内容。

| 目标类型 | 数量 | 理想判断 |
|---|---:|---|
| 正确目标 | 100 | 接受 |
| 数量错误目标 | 100 | 拒绝 |
| 边际保持的绑定交换目标 | 92 | 拒绝 |

8 张图不具备交换条件，已在校准统计中列出。多个目标对共用同一张图的 Qwen 计数，不是独立图像样本。评测器只看图片和通用计数问题，不看这些目标。

评测器与标注的一致性在 800 张测试参考图上报告，具体成绩与设置见 [results/RESULTS.md](results/RESULTS.md)。

## 数据构建

发布的数据包包含全部划分记录及 800 张测试参考 PNG，可直接使用。如需训练集和开发集参考 PNG，可从 CLEVR-CoGenT 官方压缩包恢复：

```bash
python3 provenance/source_build/prepare_dataset.py \
  --archive /path/to/CLEVR_CoGenT_v1.0.zip \
  --output . \
  --splits train development
```

该工具根据 `data/*.jsonl` 中的固定场景 ID 读取官方标注，核对 PNG 哈希，并将图片提取到记录指定的路径。

从固定划分记录重建提示词文件、统计、子集、问题、核验记录及文件清单：

```bash
python3 build_dataset.py
```

如需写入独立的数据包目录：

```bash
python3 build_dataset.py --output /path/to/CoBind-Count
```

`--source` 指定来源数据包目录，默认为本仓库；`--finalize` 刷新文件清单。划分成员与提示词文本以公开的 `data/*.jsonl` 为准，打包不重新抽取场景。数据来源与构建细节见 [provenance/README.md](provenance/README.md)。

## 归属与许可

原始图像和场景标注来自 Justin Johnson、Bharath Hariharan、Laurens van der Maaten、Li Fei-Fei、C. Lawrence Zitnick、Ross Girshick 的 **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning**（CVPR 2017）及其 CoGenT 数据。

[官方数据页面](https://cs.stanford.edu/people/jcjohns/clevr/) 使用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布数据。CoBind-Count 的衍生内容包括场景选择、数据划分、文本提示词、联合计数标签、诊断标签和校准目标；参考 PNG 保持源图像内容。本基准由独立项目提供，非 CLEVR 官方发布。许可与归属说明见 [LICENSE-DATA.txt](LICENSE-DATA.txt)。
