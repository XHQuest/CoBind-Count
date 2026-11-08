# Reference-image evaluation / 参考图评测

Qwen3.5-9B was evaluated on the 800 test reference images using their scene annotations as targets. Each image was evaluated once with the 29-question protocol. These scores measure evaluator agreement with the annotations.

使用 Qwen3.5-9B 对 800 张测试参考图进行评测，以场景标注为目标。每张图片按 29 个问题评测一次，成绩衡量评测器与标注的一致程度。

Scoring protocol / 评分协议：`cobind-count-score-v1`。

| Group / 分组 | Images / 图片 | Full Match | Count Acc | Joint Count F1 / 联合计数 F1 |
|---|---:|---:|---:|---:|
| Overall / 总计 | 800 | 96.000% | 98.125% | 99.573% |
| 3 objects / 3 个物体 | 200 | 96.500% | 99.000% | 99.436% |
| 4 objects / 4 个物体 | 200 | 96.000% | 98.000% | 99.437% |
| 5 objects / 5 个物体 | 200 | 97.000% | 98.000% | 99.778% |
| 6 objects / 6 个物体 | 200 | 94.500% | 97.500% | 99.644% |

All 800 images received scores. Thirty images had internally contradictory answers: their Full Match is zero, while Count Acc and F1 are computed from the observed counts.

800 张图片均可计分。其中 30 张的回答存在内部矛盾：其 Full Match 为 0，Count Acc 与 F1 由观察计数计算。

Details / 详细成绩：[Markdown](reference_test800/summary.md) · [CSV](reference_test800/summary.csv) · [JSON](reference_test800/summary.json)。

Evaluator settings / 评测器配置：[settings.json](settings.json)。
