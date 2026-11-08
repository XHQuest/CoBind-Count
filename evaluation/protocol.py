"""Frozen, CPU-only counting protocol and scoring for CoBind-Count."""
import csv
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 'cobind-count-qwen-v1'
SCORING_VERSION = 'cobind-count-score-v1'
BENCHMARK_VERSION = 'cobind-count-v1.0'
TEST_SEEDS = [101, 202, 303, 404]
COLORS = ('gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow')
SHAPES = ('cube', 'cylinder', 'sphere')
METRICS = ('core_full_match', 'joint_count_f1', 'count_acc')
RULES = '''Inspect the entire image. Count separate solid foreground objects, including small
and partly occluded ones. Ignore shadows, reflections, floor, background and subdivisions
on a single connected object. Count repeated objects separately. The color vocabulary is
gray, red, blue, green, brown, purple, cyan, yellow. Gray includes silver; cyan includes
turquoise/teal. Classify the object's body color, not a highlight or reflection.
The shapes are cube, cylinder, sphere. A box is a cube and a ball is a sphere.
Do not assume the scene contains a particular number or combination of objects.
Answer the current question only. Return just one nonnegative integer, with no explanation.'''


def questions():
    rows = [{'id': 'total', 'kind': 'total', 'filters': {},
             'question': 'How many solid foreground objects are there in total?'}]
    for shape in SHAPES:
        rows.append({'id': 'shape/' + shape, 'kind': 'marginal', 'filters': {'shape': shape},
                     'question': f'How many foreground objects have shape {shape}?'})
    for color, shape in itertools.product(COLORS, SHAPES):
        filters = {'color': color, 'shape': shape}
        rows.append({'id': f'binding/{color}/{shape}', 'kind': 'binding', 'filters': filters,
                     'question': 'How many foreground objects simultaneously have '
                     + ', '.join(f'{k} {v}' for k, v in filters.items()) + '?'})
    rows.append({'id': 'other', 'kind': 'other', 'filters': {},
                 'question': 'How many foreground objects cannot be assigned exactly one value '
                 'in EACH of these categories: color: gray, red, blue, green, brown, purple, cyan, yellow; '
                 'shape: cube, cylinder, sphere? Count each such object once, even if multiple attributes '
                 'are outside the vocabulary. Include ambiguous or malformed objects that cannot be '
                 'classified; do not force them into a category.'})
    for row in rows:
        row['text'] = RULES + '\n\n' + row['question']
    return rows


QUESTIONS = questions()


def expected_answer(objects, question):
    if question['kind'] == 'total':
        return len(objects)
    if question['kind'] == 'other':
        return 0
    return sum(all(obj[key] == value for key, value in question['filters'].items()) for obj in objects)


def parse_answer(text, finish_reason='stop'):
    if finish_reason != 'stop':
        return None, 'truncated' if finish_reason == 'length' else 'unexpected_finish_reason'
    if not isinstance(text, str) or not re.fullmatch(r'\d+', text.strip()):
        return None, 'expected_only_nonnegative_integer'
    try:
        count = int(text.strip())
    except ValueError:
        return None, 'implausible_count'
    if count > 10000:
        return None, 'implausible_count'
    return count, None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def image_key(task):
    return json.dumps([task['model'], task['id'], task['seed']], separators=(',', ':'))


def read_answers(path, repair_tail=False):
    """Recover only an unterminated final record, never a corrupt completed line."""
    path = Path(path)
    records = {}
    if not path.exists():
        return records
    data = path.read_bytes()
    offset = 0
    for line in data.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if offset + len(line) != len(data) or line.endswith(b'\n'):
                raise ValueError('Corrupt completed answer record; only a partial final line can be recovered')
            if repair_tail:
                with path.open('r+b') as handle:
                    handle.truncate(offset)
            break
        key = (row['key'], row['question_id'])
        if key in records:
            raise ValueError('Duplicate answer key: ' + str(key))
        records[key] = row
        offset += len(line)
    if repair_tail and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as handle:
            handle.write(b'\n')
    return records


def load_inputs(dataset, images_manifest, split, seeds, allow_partial):
    dataset, images_manifest = Path(dataset), Path(images_manifest).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != BENCHMARK_VERSION:
        raise ValueError('Unsupported benchmark version: ' + str(manifest['version']))
    if split not in manifest['splits']:
        raise ValueError('Unknown benchmark split: ' + split)
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Specify a nonempty list of distinct integer seeds')
    seeds = sorted(seeds)
    relative_path = manifest['splits'][split]
    prompt_path = dataset / relative_path
    prompt_hash = sha256(prompt_path)
    if manifest['files_sha256'][relative_path] != prompt_hash:
        raise ValueError('Prompt data differs from the frozen benchmark manifest')
    prompts = jsonl(prompt_path)
    prompt_map = {prompt['id']: prompt for prompt in prompts}
    if not prompts or len(prompt_map) != len(prompts) or any(not isinstance(p, str) for p in prompt_map):
        raise ValueError('Benchmark needs unique string prompt ids')
    tasks, keys = [], set()
    for row in jsonl(images_manifest):
        if not isinstance(row.get('model'), str) or not row['model'].strip():
            raise ValueError('Every image needs a nonempty model label')
        if not isinstance(row.get('id'), str) or row['id'] not in prompt_map:
            raise ValueError('Unknown/nonstring prompt id: ' + str(row.get('id')))
        if type(row.get('seed')) is not int or row['seed'] not in seeds:
            raise ValueError('Every image seed must belong to the explicitly requested seed set')
        path = Path(row['image'])
        if not path.is_absolute():
            path = images_manifest.parent / path
        task = {key: row[key] for key in ('model', 'id', 'seed')}
        task.update(image=str(path.resolve()), image_sha256=sha256(path))
        key = image_key(task)
        if key in keys:
            raise ValueError('Duplicate model/prompt/seed: ' + key)
        keys.add(key)
        task['key'] = key
        tasks.append(task)
    if not tasks:
        raise ValueError('The image manifest is empty')
    models = sorted({task['model'] for task in tasks})
    expected = len(models) * len(seeds) * len(prompts)
    complete = len(tasks) == expected
    if not complete and not allow_partial:
        raise ValueError(f'Incomplete image coverage: {len(tasks)}/{expected}; all prompts x specified seeds x models required')
    tasks.sort(key=lambda task: (task['id'], task['seed'], task['model']))
    snapshot = dict(
        version=manifest['version'], split=split, prompts_sha256=prompt_hash,
        source_manifest_sha256=sha256(dataset / 'manifest.json'),
        prompts=prompts, models=models, seeds=seeds,
        complete_image_coverage=complete, expected_images=expected, listed_images=len(tasks),
        partial_requested=bool(allow_partial),
        standard_protocol=split == 'test' and seeds == TEST_SEEDS and not allow_partial,
        partial_denominator='all split prompts x explicitly specified seeds x listed models',
    )
    return snapshot, tasks


def validate_answers(records, tasks, prompts):
    task_map = {task['key']: task for task in tasks}
    prompt_map = {prompt['id']: prompt for prompt in prompts}
    question_ids = {question['id'] for question in QUESTIONS}
    if len(task_map) != len(tasks):
        raise ValueError('Duplicate frozen task key')
    for task in tasks:
        if task['key'] != image_key(task) or task['id'] not in prompt_map:
            raise ValueError('Invalid frozen task key or prompt id')
    for (key, qid), record in records.items():
        if key not in task_map:
            raise ValueError('Answer belongs to an unknown image: ' + key)
        if qid not in question_ids or record['key'] != key or record['question_id'] != qid:
            raise ValueError('Invalid answer question/key')
        if record['image_sha256'] != task_map[key]['image_sha256']:
            raise ValueError('Image hash does not match its recorded answer: ' + key)
        value, error = parse_answer(record['raw_response'], record['finish_reason'])
        if record['value'] != value or record['parse_error'] != error or (value is not None and type(record['value']) is not int):
            raise ValueError('Stored answer does not match the frozen parser')


def assess(objects, answers, qs=None):
    """Compute point scores and record answer-consistency diagnostics."""
    qs = QUESTIONS if qs is None else qs
    if qs != QUESTIONS:
        raise ValueError('Only the frozen 29-question protocol is supported')
    if any(obj['color'] not in COLORS or obj['shape'] not in SHAPES for obj in objects):
        raise ValueError('Target objects must use the frozen color/shape vocabulary')
    counts = {question['id']: answers.get(question['id']) for question in qs}
    missing = [qid for qid, value in counts.items() if value is None]
    invalid = [qid for qid, value in counts.items()
               if value is not None and (type(value) is not int or not 0 <= value <= 10000)]
    result = dict(evaluation_valid=False, missing_answers=missing, invalid_counts=invalid,
                  inconsistencies=[], observed_counts=counts, error_type='unscored',
                  scored=False, scoring_version=SCORING_VERSION, status='unscored', **{metric: None for metric in METRICS})
    result.update({metric: None for metric in ('total_count_acc', 'shape_count_acc', 'color_count_acc',
                                             'marginal_pass', 'marginal_pass_joint_fail', 'tp', 'fp', 'fn')})
    if missing or invalid:
        return result
    bindings = [question for question in qs if question['kind'] == 'binding']
    total, other = counts['total'], counts['other']
    inconsistencies = result['inconsistencies']
    known_total = sum(counts[question['id']] for question in bindings)
    if known_total + other != total:
        inconsistencies.append('binding_sum_plus_other_differs_from_total')
    for question in qs:
        value = counts[question['id']]
        if value > total and question['kind'] != 'total':
            inconsistencies.append('exceeds_total:' + question['id'])
        if question['kind'] == 'marginal':
            subtotal = sum(counts[b['id']] for b in bindings
                           if b['filters']['shape'] == question['filters']['shape'])
            # An unknown-color object may still have a known shape.
            if not subtotal <= value <= subtotal + other:
                inconsistencies.append('marginal_disagrees:' + question['id'])
    shape_sum = sum(counts['shape/' + shape] for shape in SHAPES)
    if not total - other <= shape_sum <= total:
        inconsistencies.append('marginal_partition_disagrees:shape')
    result['observed_count_from_bindings'] = known_total + other
    target = Counter((obj['color'], obj['shape']) for obj in objects)
    observed = Counter({(question['filters']['color'], question['filters']['shape']): counts[question['id']]
                        for question in bindings})
    tp = sum(min(target[pair], observed[pair]) for pair in target)
    total_ok = total == len(objects)
    shape_ok = all(counts['shape/' + shape] == sum(obj['shape'] == shape for obj in objects) for shape in SHAPES)
    color_ok = all(sum(observed[color, shape] for shape in SHAPES) == sum(obj['color'] == color for obj in objects)
                   for color in COLORS)
    joint_ok = all(observed[color, shape] == target[color, shape] for color, shape in itertools.product(COLORS, SHAPES))
    count_ok = total_ok and shape_ok
    marginal_ok = count_ok and color_ok
    full = joint_ok and other == 0 and count_ok and not inconsistencies
    if inconsistencies:
        error_type = 'inconsistent'
    elif not total_ok:
        error_type = 'total'
    elif not shape_ok:
        error_type = 'shape'
    elif other:
        error_type = 'other'
    elif not color_ok:
        error_type = 'color'
    elif not joint_ok:
        error_type = 'binding'
    else:
        error_type = 'pass'
    denominator = len(objects) + known_total + other
    result.update(evaluation_valid=not inconsistencies, scored=True,
                  core_full_match=full, count_acc=count_ok,
                  joint_count_f1=2 * tp / denominator if denominator else 1.0,
                  total_count_acc=total_ok, shape_count_acc=shape_ok, color_count_acc=color_ok,
                  marginal_pass=marginal_ok, marginal_pass_joint_fail=marginal_ok and not joint_ok,
                  tp=tp, fp=known_total + other - tp, fn=len(objects) - tp,
                  error_type=error_type, status='pass' if full else 'fail')
    return result


def score_images(snapshot, tasks, records):
    listed = {task['key'] for task in tasks}
    scores = []
    for model in snapshot['models']:
        for prompt in snapshot['prompts']:
            color_shapes = defaultdict(set)
            for obj in prompt['objects']:
                color_shapes[obj['color']].add(obj['shape'])
            for seed in snapshot['seeds']:
                task = dict(model=model, id=prompt['id'], seed=seed)
                key = image_key(task)
                answers = {question['id']: records.get((key, question['id'])) for question in QUESTIONS}
                values = {qid: None if record is None else record['value'] for qid, record in answers.items()}
                result = assess(prompt['objects'], values)
                scores.append(dict(
                    **task, key=key, condition=prompt['condition'], object_count=prompt['object_count'],
                    diagnostic_group=prompt['diagnostic_group'], difficulty=prompt['difficulty'],
                    core_target_id=prompt['core_target_id'],
                    same_color_different_shapes=any(len(shapes) > 1 for shapes in color_shapes.values()),
                    image_listed=key in listed, questions=len(QUESTIONS),
                    missing=sum(record is None for record in answers.values()),
                    invalid=sum(record is not None and record['value'] is None for record in answers.values()),
                    parse_errors={qid: record['parse_error'] for qid, record in answers.items()
                                  if record is not None and record['parse_error'] is not None}, **result))
    return scores


def aggregate(rows):
    output = dict(prompts=len({row['id'] for row in rows}), expected_images=len(rows),
                  listed_images=sum(row['image_listed'] for row in rows),
                  fully_answered_images=sum(row['missing'] == 0 for row in rows),
                  scored_images=sum(row['scored'] for row in rows),
                  unscored_images=sum(not row['scored'] for row in rows),
                  consistent_images=sum(row['evaluation_valid'] for row in rows),
                  inconsistent_images=sum(bool(row['inconsistencies']) for row in rows),
                  expected_questions=sum(row['questions'] for row in rows),
                  invalid_answers=sum(row['invalid'] for row in rows),
                  pending_questions=sum(row['missing'] for row in rows))
    output['answered_questions'] = output['expected_questions'] - output['pending_questions']
    output['inconsistent_rate'] = output['inconsistent_images'] / len(rows) if rows else None
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        output[metric] = sum(values) / len(values) if values else None
        output[metric + '_available_images'] = len(values)
    output['auxiliary_marginal_pass_joint_fail_images'] = sum(
        row['evaluation_valid'] and row['marginal_pass_joint_fail'] is True for row in rows)
    output['auxiliary_marginal_pass_images_denominator'] = sum(
        row['evaluation_valid'] and row['marginal_pass'] is True for row in rows)
    for error_type in ('unscored', 'inconsistent', 'total', 'shape', 'other', 'color', 'binding', 'pass'):
        output['error_' + error_type] = sum(row['error_type'] == error_type for row in rows)
    return output


def _percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap_points(rows, samples, seed):
    """Prompt-cluster bootstrap, stratified by condition x N; seeds stay together."""
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row['id']].append(row)
    strata = defaultdict(list)
    for items in by_prompt.values():
        if any(row[metric] is None for row in items for metric in METRICS):
            raise ValueError('Point-score bootstrap requires complete, parseable answers for every image')
        vector = [sum(float(row[metric]) for row in items) for metric in METRICS] + [len(items)]
        strata[items[0]['condition'], items[0]['object_count']].append(vector)
    if not by_prompt:
        return None
    rng, draws = random.Random(seed), [[] for _ in METRICS]
    for _ in range(samples):
        total = [0.0] * (len(METRICS) + 1)
        for vectors in strata.values():
            for _ in vectors:
                chosen = rng.choice(vectors)
                total = [left + right for left, right in zip(total, chosen)]
        for values, value in zip(draws, total[:-1]):
            values.append(value / total[-1])
    return {metric: [_percentile(draws[i], .025), _percentile(draws[i], .975)]
            for i, metric in enumerate(METRICS)}


def write_summary(output, snapshot, tasks, records, bootstrap_samples=0, bootstrap_seed=20261106):
    output = Path(output)
    if bootstrap_samples < 0:
        raise ValueError('bootstrap_samples must be nonnegative')
    scores = score_images(snapshot, tasks, records)
    if bootstrap_samples and any(not row['scored'] for row in scores):
        raise ValueError('Bootstrap is available only after every image has complete, parseable answers')
    temp = output / 'image_scores.jsonl.tmp'
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scores), encoding='utf-8')
    temp.replace(output / 'image_scores.jsonl')
    groups, intervals = [], []
    for model in snapshot['models']:
        selected = [row for row in scores if row['model'] == model]
        subsets = [('all', {}, selected)]
        for count in (3, 4, 5, 6):
            subsets.append(('object_count', {'object_count': count},
                            [row for row in selected if row['object_count'] == count]))
        for scope, labels, subset in subsets:
            group = dict(model=model, scope=scope, level='all' if scope == 'all' else 'N' + str(labels['object_count']),
                         object_count='')
            group.update(labels)
            group.update(aggregate(subset))
            groups.append(group)
            if bootstrap_samples and subset:
                intervals.append(dict(model=model, scope=scope, **labels,
                                      intervals=bootstrap_points(subset, bootstrap_samples, bootstrap_seed)))
    pending = sum(row['missing'] for row in scores)
    invalid = sum(row['invalid'] for row in scores)
    unscored = sum(not row['scored'] for row in scores)
    inconsistent = sum(bool(row['inconsistencies']) for row in scores)
    partial = not snapshot['complete_image_coverage'] or pending > 0
    summary = dict(
        protocol=VERSION, scoring_version=SCORING_VERSION,
        benchmark_version=snapshot['version'], split=snapshot['split'],
        standard_protocol=snapshot.get('standard_protocol', False),
        standard_protocol_deviations=snapshot.get('standard_protocol_deviations', []),
        judge_settings=snapshot.get('judge_settings', {}),
        standard_report_eligible=snapshot.get('standard_protocol', False) and not partial and unscored == 0,
        point_estimate_available=not partial and unscored == 0,
        complete_image_coverage=snapshot['complete_image_coverage'],
        all_expected_answers_recorded=pending == 0,
        complete_score=not partial and unscored == 0,
        partial=partial, invalid_answers=invalid, pending_questions=pending,
        scored_images=len(scores) - unscored, unscored_images=unscored, inconsistent_images=inconsistent,
        status='incomplete' if partial else ('complete_with_unscored' if unscored else 'complete'),
        expected_images=len(scores), listed_images=len(tasks), expected_seed_set=snapshot['seeds'],
        aggregation='Arithmetic mean of every available per-image score, including inconsistent but parseable judgments. Equal fixed seed counts make this equivalent to prompt-first averaging on complete runs.',
        inconsistency_policy='Full Match is zero for contradictory answers; Count Acc and joint count F1 are computed from the observed counts.',
        missing_policy='Incomplete or unparseable image judgments have no score and are excluded from that metric mean. Available-image denominators and unscored counts are explicit; incomplete results are not a complete benchmark score.',
        levels=['all', 'N3', 'N4', 'N5', 'N6'],
        auxiliary_diagnostic='Marginal-pass/joint-fail counts use only self-consistent judgments and are not a separate ranking metric.',
        error_priority=['unscored', 'inconsistent', 'total', 'shape', 'other', 'color', 'binding', 'pass'],
        groups=groups,
    )
    if bootstrap_samples:
        summary['bootstrap'] = dict(samples=bootstrap_samples, seed=bootstrap_seed, confidence=.95,
                                    method='Percentile, prompt-cluster resampling within condition x N strata; all seeds retained together.',
                                    interpretation='Optional confidence intervals for the point-score estimators; they do not correct judge recognition errors. Primary reports remain single values.',
                                    groups=intervals)
    atomic_json(output / 'summary.json', summary)
    fields = list(dict.fromkeys(key for row in groups for key in row))
    temp = output / 'summary.csv.tmp'
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups)
    temp.replace(output / 'summary.csv')
    lines = ['# CoBind-Count：评测结果', '',
             f'评分版本：`{SCORING_VERSION}`。问答协议：`{VERSION}`。', '',
             '主指标为逐图均值。矛盾回答的 Full Match 为 0，Count Acc 与联合计数 F1 继续按原始计数计算。',
             '缺失或不可解析的图片不产生分数，单独记录；完整成绩要求全部图片可评分。', '',
             '| 模型 | 分级 | 提示词 | 图片 | 可评分 | Full Match | Count Acc | 联合计数 F1 | 矛盾图片 | 不可评分 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def percent(value):
        return '—' if value is None else f'{value * 100:.3f}%'
    for group in groups:
        label = '总计' if group['scope'] == 'all' else f"{group['object_count']} 个物体"
        lines.append(f"| {group['model']} | {label} | {group['prompts']} | {group['expected_images']} | "
                     f"{group['scored_images']} | {percent(group['core_full_match'])} | "
                     f"{percent(group['count_acc'])} | {percent(group['joint_count_f1'])} | "
                     f"{group['inconsistent_images']} | {group['unscored_images']} |")
    temp = output / 'summary.md.tmp'
    temp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temp.replace(output / 'summary.md')
    return summary
