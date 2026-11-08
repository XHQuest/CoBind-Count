CoBind-Count：参考图与目标对校准
===============================

数据与目的
----------
校准使用具有已知标注的参考图，检验评分能否接受正确目标、拒绝数量错误目标，
以及拒绝颜色总数和形状总数正确但具体颜色×形状组合错误的目标。
参考 PNG 使用原始像素内容，目标与标准答案不提供给 Qwen。

校准方案：cobind-count-calibration-v1。

references.jsonl 是 development 中固定选取的 100 条记录：
A/B 各 50，目标物体总数 N=3、4、5、6 各 25。
N=3、5 时 A/B 配额为 13/12，N=4、6 时为 12/13。
按 N=3..6、A/B 顺序遍历，每组按 id 排序，再由同一个
Python Random(20261106) 执行 sample。不依据模型成绩选择样本。
图片路径指向仓库 images/ 下的相应位置；这 100 图属于开发集，不加入测试集分母。

仓库只包含 800 张测试集参考 PNG，不包含这 100 张开发集校准 PNG。
运行 Qwen 校准前，请从 CLEVR-CoGenT 官方数据取得对应原始图片：
https://cs.stanford.edu/people/jcjohns/clevr/
按 references.jsonl 各记录中的 reference_image 路径放置图片，路径相对于仓库根目录。
image_sha256 是所需原始 PNG 的内容哈希。calibration/images.jsonl 中的图片路径
相对于该清单文件。标注与目标对构建本身不需要读取 PNG。

目标对
------
positive：原始 24 格联合计数，预期接受，共 100 对。
count_mismatch：固定选取词表顺序的首个非零组合，数量加 1；
  若原总数为 6 则减 1，使目标总数保持在 3..6，预期拒绝，共 100 对。
binding_swap：选取颜色不同、形状也不同的两个非零组合，各减 1，
  两个交叉组合各加 1。例如“红方块+蓝球”对应交换目标“蓝方块+红球”。
  总数和颜色、形状边际数量保持相同，联合组合不同，预期拒绝，共 92 对。
  按固定词表顺序选第一组可行交换；8 张无可行交换的图列于
  stats.json 的 binding_swap_unavailable_ids 中，不构造额外负例。

pairs.jsonl 含全部 24 格 target_joint_counts，expected_other=0。
同图的多个目标对共享观察计数，不是独立图像样本。
配额、抽样规则、目标对数与文件哈希见 stats.json。

构建
----
在 CoBind-Count 根目录执行：

python3 calibration/build_pairs.py --verify-oracle
python3 build_dataset.py --finalize

build_pairs.py 从 development 构造 references.jsonl、pairs.jsonl、images.jsonl 和 stats.json。
重建会覆盖这些产物并记录数据及代码哈希；使用发布的校准集时不必重建。
--verify-oracle 从原始标注推导理想计数，检验正例接受、负例拒绝和交换目标的边际保持。
verification.json 的 oracle_only 标记表示标注与评分逻辑检查，不是 Qwen 实测准确率。

Qwen 观察
---------
确保上述 100 张开发集 PNG 已放到对应路径，然后在 CoBind-Count 根目录执行：

python evaluation/run_qwen_eval.py \
  --split calibration \
  --images-manifest calibration/images.jsonl \
  --model-path /path/to/Qwen3.5-9B \
  --output outputs/calibration_qwen

每张参考图只评一次。seed 101 是清单索引，不表示渲染种子。
每图独立回答 29 个计数问题，总计 2,900 个问题。
Qwen 只看图和当前通用问题，不看 prompt、参考答案或某个目标对。
问答协议为 cobind-count-qwen-v1，评分协议为 cobind-count-score-v1。
相同命令支持断点续评；具体设置见 ../evaluation/README.txt。

目标比较
--------
Qwen 进程结束后，在 CPU 上比较各个目标：

python3 calibration/evaluate_pairs.py \
  --evaluation outputs/calibration_qwen \
  --output outputs/calibration_pairs

工具读取 benchmark.json、judge_config.json、tasks.jsonl、answers.jsonl，
核对固定 100 图 calibration 划分和参考图内容哈希。读取时持有共享锁，
避免与问答写入并发。只使用 reference_id 匹配的计数，不触发新提问。

pair_scores.jsonl：每个 model/seed/pair 的接受、拒绝或未计分状态及诊断。
summary.csv、report.json：positive、count_mismatch、binding_swap、all_negatives 四组目标对统计。
这四类用于校准目标诊断，模型测试成绩仍采用总计及 N3..N6 五项分级。

正例接受率越高越好，负例误接受率越低越好。
完整可解析计数有分数；内部矛盾使 Full Match 为 0，即该目标对被拒绝。
缺图、缺题或不可解析的目标对没有分数，其详细问题 ID 记录于 unavailable_question_ids。

acceptance = 接受数 / available_pairs，为单值。
available_pairs 为可计分目标对数量；未计分数量另列，分母为 0 时值为 null。
完整成绩要求全部预期目标对均可计分。同图目标对共享观察，不能将其视为独立样本计算置信区间。

适用范围
--------
校准可检验联合计数对绑定目标的区分能力。它不能证明 Qwen 对严重遮挡、
畸形、混色或模糊生成物体同样可靠。交换目标不要求服从原图 A/B 组合规则。

开发集 100 图与测试集 800 图是不同用途的数据划分。
公开的 800 张测试参考图结果位于 ../results/reference_test800。
verification.json 只提供标注 oracle 验证，不作为这 100 张开发图的 Qwen 实测成绩。
