"""Reuse image-only Qwen counts to evaluate positive/counterfactual targets on CPU."""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'evaluation'))
from protocol import (VERSION, SCORING_VERSION, QUESTIONS, assess, atomic_json, image_key, jsonl,
                      read_answers, sha256, validate_answers)


def target_objects(pair):
    objects = []
    for key, count in pair['target_joint_counts'].items():
        color, shape = key.split('/')
        objects.extend(dict(color=color, shape=shape) for _ in range(count))
    return objects


def summarize(rows, model, seed, kind):
    passed = sum(row['accepted'] is True for row in rows)
    rejected = sum(row['accepted'] is False for row in rows)
    unresolved = len(rows) - passed - rejected
    return dict(
        model=model, seed=seed, kind=kind, pairs=len(rows),
        accepted=passed, rejected=rejected, unresolved=unresolved,
        pairs_with_missing_image=sum(row['missing_image'] for row in rows),
        pairs_with_missing_answers=sum(row['missing_answer_count'] > 0 for row in rows),
        pairs_with_invalid_answers=sum(row['invalid_answer_count'] > 0 for row in rows),
        pairs_with_inconsistencies=sum(bool(row['inconsistencies']) for row in rows),
        rate_name='positive_acceptance' if kind == 'positive' else 'negative_false_acceptance',
        available_pairs=passed + rejected,
        acceptance=passed / (passed + rejected) if passed + rejected else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True,
                        help='Output from run_qwen_eval.py --split calibration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    # Read a stable snapshot and avoid modifying/retrying the judge's answers.
    lock = (args.evaluation / 'run.lock').open('r')
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Qwen evaluation is still writing; rerun after it stops') from error
    calibration = args.benchmark / 'calibration'
    pairs = jsonl(calibration / 'pairs.jsonl')
    references = {r['id']: r for r in jsonl(calibration / 'references.jsonl')}
    statistics = json.loads((calibration / 'stats.json').read_text())
    for name in ('pairs.jsonl', 'references.jsonl'):
        if sha256(calibration / name) != statistics['files_sha256'][name]:
            raise ValueError('Calibration data differs from its fixed manifest: ' + name)
    config = json.loads((args.evaluation / 'judge_config.json').read_text())
    if config['protocol'] != VERSION or config['questions'] != QUESTIONS:
        raise ValueError('Judge counts use different observation questions or protocol')
    snapshot = json.loads((args.evaluation / 'benchmark.json').read_text())
    if (snapshot['split'] != 'calibration' or {p['id'] for p in snapshot['prompts']} != set(references)
            or snapshot['prompts_sha256'] != sha256(calibration / 'references.jsonl')):
        raise ValueError('Expected the fixed 100-reference calibration split')
    tasks = jsonl(args.evaluation / 'tasks.jsonl')
    records = read_answers(args.evaluation / 'answers.jsonl')
    validate_answers(records, tasks, snapshot['prompts'])
    task_keys = {task['key'] for task in tasks}
    for task in tasks:
        if task['image_sha256'] != references[task['id']]['image_sha256']:
            raise ValueError('Calibration labels require the unmodified official reference image: ' + task['id'])
    observations, coverage = {}, {}
    for model in snapshot['models']:
        for seed in snapshot['seeds']:
            for reference_id in references:
                key = image_key(dict(model=model, id=reference_id, seed=seed))
                observations[model, seed, reference_id] = {
                    q['id']: records.get((key, q['id']), {}).get('value') for q in QUESTIONS
                }
                image_records = [records.get((key, q['id'])) for q in QUESTIONS]
                coverage[model, seed, reference_id] = dict(
                    missing_image=key not in task_keys,
                    missing_answer_count=sum(record is None for record in image_records),
                    invalid_answer_count=sum(record is not None and record['value'] is None for record in image_records),
                )
    details = []
    groups = defaultdict(list)
    for model in snapshot['models']:
        for seed in snapshot['seeds']:
            for pair in pairs:
                answers = observations[model, seed, pair['reference_id']]
                scored = assess(target_objects(pair), answers)
                accepted = None if scored['core_full_match'] is None else bool(scored['core_full_match'])
                detail = dict(model=model, seed=seed, pair_id=pair['pair_id'],
                    reference_id=pair['reference_id'], kind=pair['kind'],
                    expected_accept=pair['expected_accept'], accepted=accepted,
                    evaluation_valid=scored['evaluation_valid'],
                    unavailable_question_ids=scored['missing_answers'],
                    inconsistencies=scored['inconsistencies'],
                    **coverage[model, seed, pair['reference_id']])
                details.append(detail)
                groups[model, seed, pair['kind']].append(detail)
                if pair['kind'] != 'positive':
                    groups[model, seed, 'all_negatives'].append(detail)
    summaries = [summarize(rows, *key) for key, rows in sorted(groups.items())]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'pair_scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in details))
    with (args.output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = dict(
        calibration_version=statistics['version'], protocol=VERSION, scoring_version=SCORING_VERSION,
        pair_data_sha256=sha256(calibration / 'pairs.jsonl'),
        judge_config_sha256=sha256(args.evaluation / 'judge_config.json'),
        answers_sha256=sha256(args.evaluation / 'answers.jsonl') if (args.evaluation / 'answers.jsonl').exists() else None,
        qwen_called_by_this_script=False,
        unresolved_reason_note='Only missing or invalid counts are unscored; inconsistent complete observations do not pass Full Match.',
        score_note='Single acceptance rate over available pairs; available and unscored counts are reported explicitly.',
        limitation='Known reference images with alternate targets do not establish accuracy on distorted or occluded generated images.',
        summaries=summaries,
    )
    atomic_json(args.output / 'report.json', report)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
