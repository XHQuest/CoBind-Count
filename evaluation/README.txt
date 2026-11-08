CoBind-Count：Qwen 离线评测
=========================

协议
----
数据版本：cobind-count-v1.0。
问答协议：cobind-count-qwen-v1。
评分协议：cobind-count-score-v1。

每张图片独立询问 29 个数字问题：总数 1 题、各形状数量 3 题、
8 色×3 形状联合数量 24 题、其他物体数量 1 题。
Qwen 只看图片与当前问题，不接触生成提示词、预期答案、方法名或其他题的答案。
评测范围为 count/color/shape；size、material 和空间布局不计入核心分数。
参考图用于校准评测器，不是生成输入，参考图成绩与生成图成绩分别报告。

输入
----
准备 UTF-8 JSONL，每行一张图：

{"model":"model_a","id":"CLEVR_valB_013250","seed":101,"image":"images/model_a/scene_101.png"}

id 必须使用对应 split 的场景 ID。image 可为绝对路径，也可相对于清单目录。
model 区分方法和生成设置；不同步数应使用独立标签。
按固定提示词和种子生成图片，记录各模型的权重标识、生成步数、分辨率和引导系数。

正式 test 为 800 条提示词×101、202、303、404 四个种子，
每个模型 3,200 张图、92,800 个问题。工具检查完整 prompt×seed×model 覆盖。
development 和 calibration 默认使用 seed 101。
参考图每张只评一次；test 参考图使用 --seeds 101，seed 仅作为记录索引。
该单种子参考图任务不会被标记为四种子标准生成评测。

运行
----
需要兼容的 vLLM、PyTorch、Transformers、Pillow 和 tqdm，以及本地 Qwen3.5-9B。
工具开启离线模式，不自动下载模型或安装依赖。实际库版本写入配置。

cd /path/to/CoBind-Count
python evaluation/run_qwen_eval.py \
  --split test \
  --images-manifest /path/to/images.jsonl \
  --model-path /path/to/Qwen3.5-9B \
  --output outputs/test_qwen

--dataset 默认为本工具所属数据目录；--split 支持 development、test、calibration。
--model-revision 可记录权重发布 revision，--model-id 应与本地权重的模型名一致。

固定设置：BF16、temperature=0、judge seed=20261103、thinking=False、
每题最多 32 token、min_pixels=512²、max_pixels=1024²、上下文长度 4096。
默认每块处理 8 张图片，最多并行 32 个请求，每个问题都是独立请求。
--images-per-chunk、--max-num-seqs、--gpu-memory-utilization 控制资源使用。
偏离标准模型或图像处理等设置时，standard_protocol_deviations 记录具体项目。

续评与记录
----------
相同命令可以续评，只补做缺失问题。每块答案写入后 flush + fsync，
未完成的末行可在读取时截除。输出目录带进程锁，避免同时写入。
全部答案存在时直接生成报告，不加载模型。无效回答不自动重试或语义纠正。

judge_config.json 记录数据、图片清单、图片与代码 SHA256、模型标识、
小配置文件哈希、权重文件元数据、库版本和运行设置。
不计算大权重文件的完整内容哈希，文件元数据不能单独证明权重内容相同。
续评要求相同配置和输入；不同配置应使用独立输出目录。
续评时保持工具代码、manifest 和输入快照一致。
对保存的回答进行 CPU 汇总不需要重新提问；使用 --source 和独立 --output
保留观察结果，并记录观察来源和评分代码。

answers.jsonl：原始回答、解析结果、问题 ID 和 token 数。
benchmark.json、tasks.jsonl：数据与图片任务快照。
progress.json：进度；终端同时显示按问题推进的 tqdm。
image_scores.jsonl：逐图分数和诊断。
summary.json、summary.csv、summary.md：五项分级成绩。

评分与分级
----------
每个模型固定报告总计、3 个物体、4 个物体、5 个物体、6 个物体五项。
测试集每个数量组含 200 条提示词，参考图每级 200 张，四种子生成图每级 800 张。
A/B 与绑定难度标签保留在数据中，默认成绩表不增加其他分组。

core_full_match：总数、各形状数量与全部颜色×形状数量正确，other=0，且回答自洽。
count_acc：总数及三个形状数量同时正确。
joint_count_f1：2×联合计数交集重数 /（目标总数 + 预测联合总数 + other）。

完整可解析回答均有分数。矛盾回答的 full match 为 0，count acc 和 F1 仍按计数计算。
缺图、缺题或不可解析回答使整张图三项分数为 None；均值排除 None，
同时显式记录 available_images、unscored_images 与覆盖率。
完整成绩要求全部预期图片均可计分。inconsistent_images 单列回答矛盾数量，
不用于筛除 count acc 或 F1 样本。每项主成绩为可计分图片的逐图平均单值。
固定四种子且全部可计分时，该平均与先平均种子、再等权平均提示词相同。

CPU 报告工具
------------
对已保存的问答生成报告：

python evaluation/summarize.py --output outputs/test_qwen

使用独立观察目录与报告目录：

python evaluation/summarize.py \
  --source /path/to/observations \
  --output /path/to/report

工具读取问答及输入快照，记录问答协议、评分协议与来源哈希，不调用 Qwen。
不要与正在写入同一源目录的 runner 同时执行。
可选 --bootstrap-samples 用于提示词层面的区间估计，默认关闭；
仅完整可计分的输入支持该选项，主表始终提供单值。

800 张测试参考图的公开成绩位于 results/reference_test800。
合并多个评测报告：

python evaluation/combine_reports.py \
  --summaries outputs/model_a/summary.json \
              outputs/model_b/summary.json \
  --output outputs/comparison

产物为 RESULTS.md、combined_results.json、combined_results.csv。
合并表保留每份报告的数据版本、划分、种子、标准协议标记与可评分数量。
合并不自动证明可比性；比较模型前需确认相同测试数据、种子、评分器和问答设置。
参考图分数衡量评测器与标注的一致程度，模型分数衡量生成结果对目标的符合程度。

--allow-partial 适合开发检查；缺图属于未计分图片并保留在预期覆盖率中。
非默认种子、partial 模式或非标准评测设置会被明确标记。
原始回答和运行日志位于 outputs/，该目录不纳入公开发布文件。
