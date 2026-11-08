"""Combine existing five-group point-score summaries without loading models."""
import argparse
import csv
import json
import os
from pathlib import Path

from protocol import SCORING_VERSION, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_groups, sources, text = [], [], [
        '# CoBind-Count：评测结果', '',
        f'评分版本：`{SCORING_VERSION}`。所有分数以百分比展示。', '',
        'Count Acc 和 F1 保留完整但矛盾的回答；Full Match 对矛盾回答记 0。'
        '仅缺失或无法解析的整图不计入该指标均值，并单独记录可计分数量。', '',
    ]
    for source in args.summaries:
        report = json.loads(source.read_text(encoding='utf-8'))
        if report['scoring_version'] != SCORING_VERSION:
            raise ValueError('Cannot mix scoring versions')
        groups = report['groups']
        models = list(dict.fromkeys(group['model'] for group in groups))
        report_path = os.path.relpath(source.resolve(), args.output.resolve())
        sources.append(dict(path=report_path, sha256=sha256(source),
                            benchmark_version=report['benchmark_version'], split=report['split'],
                            status=report['status'], partial=report['partial'],
                            standard_protocol=report['standard_protocol'],
                            expected_seed_set=report['expected_seed_set']))
        for model in models:
            selected = [group for group in groups if group['model'] == model]
            if len(selected) != 5:
                raise ValueError('Expected exactly five groups per model')
            text.extend([f'## {model}', '',
                f"数据版本：`{report['benchmark_version']}`；划分：`{report['split']}`；"
                f"种子：`{report['expected_seed_set']}`；状态：`{report['status']}`；"
                f"标准生成评测：`{report['standard_protocol']}`。", '',
                '| 分组 | 提示词数 | 预期图片 | 可评分 | 未评分 | Full Match | Count Acc | 联合计数 F1 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|'])
            for group in selected:
                label = '总计' if group['scope'] == 'all' else f"{group['object_count']} 个物体"
                values = ['—' if group[key] is None else f"{100 * group[key]:.3f}%"
                          for key in ('core_full_match', 'count_acc', 'joint_count_f1')]
                text.append(f"| {label} | {group['prompts']} | {group['expected_images']} | "
                            f"{group['scored_images']} | {group['unscored_images']} | " + ' | '.join(values) + ' |')
            text.extend(['', f"[详细报告]({Path(report_path).with_suffix('.md').as_posix()})。", ''])
        all_groups.extend(groups)
    text.extend(['各分组按目标物体数划分。合并只汇集原报告，不确认不同数据划分、种子或评测设置之间的可比性。', ''])
    (args.output / 'RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    atomic_json(args.output / 'combined_results.json', dict(scoring_version=SCORING_VERSION, sources=sources, groups=all_groups))
    fields = list(dict.fromkeys(key for group in all_groups for key in group))
    with (args.output / 'combined_results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_groups)
    print(args.output / 'RESULTS.md')


if __name__ == '__main__':
    main()
