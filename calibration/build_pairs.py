"""Build fixed reference/target calibration pairs, without running a judge."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

COLORS = ('gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow')
SHAPES = ('cube', 'cylinder', 'sphere')
CELLS = tuple(f'{color}/{shape}' for color in COLORS for shape in SHAPES)
SEED = 20261106


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, records):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def counts(objects):
    sparse = Counter(f"{o['color']}/{o['shape']}" for o in objects)
    return {key: sparse[key] for key in CELLS}


def marginals(joint):
    color, shape = Counter(), Counter()
    for key, value in joint.items():
        c, s = key.split('/')
        color[c] += value
        shape[s] += value
    return dict(color), dict(shape)


def make_pairs(references):
    pairs, unswappable = [], []
    for reference in references:
        actual = counts(reference['objects'])
        image = reference['reference_image']
        common = dict(reference_id=reference['id'], reference_image=image)
        pairs.append(dict(**common, pair_id=reference['id'] + ':positive',
            kind='positive', expected_accept=True, target_joint_counts=actual,
            target_count=sum(actual.values()), expected_other=0))
        count_target = actual.copy()
        cell = next(key for key in CELLS if count_target[key] > 0)
        delta = -1 if sum(actual.values()) == 6 else 1
        count_target[cell] += delta
        pairs.append(dict(**common, pair_id=reference['id'] + ':count_mismatch',
            kind='count_mismatch', expected_accept=False, target_joint_counts=count_target,
            target_count=sum(count_target.values()), expected_other=0,
            edit=dict(cell=cell, delta=delta), preserves_marginals=False))
        choices = [key for key in CELLS if actual[key] > 0]
        for first, second in itertools.combinations(choices, 2):
            color1, shape1 = first.split('/')
            color2, shape2 = second.split('/')
            if color1 == color2 or shape1 == shape2:
                continue
            swapped = actual.copy()
            swapped[first] -= 1
            swapped[second] -= 1
            swapped[f'{color1}/{shape2}'] += 1
            swapped[f'{color2}/{shape1}'] += 1
            pairs.append(dict(**common, pair_id=reference['id'] + ':binding_swap',
                kind='binding_swap', expected_accept=False, target_joint_counts=swapped,
                target_count=sum(swapped.values()), expected_other=0,
                preserves_marginals=True, edit=dict(decrement=[first, second],
                    increment=[f'{color1}/{shape2}', f'{color2}/{shape1}'])))
            break
        else:
            unswappable.append(reference['id'])
    return pairs, unswappable


def build(benchmark):
    benchmark = Path(benchmark).resolve()
    output = benchmark / 'calibration'
    source = benchmark / 'data/development.jsonl'
    benchmark_manifest = json.loads((benchmark / 'manifest.json').read_text())
    development = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rng, selected = random.Random(SEED), []
    quotas = {}
    for count in (3, 4, 5, 6):
        for condition in ('A', 'B'):
            quota_a = 13 if count % 2 else 12
            quota = quota_a if condition == 'A' else 25 - quota_a
            population = sorted(
                (r for r in development if r['condition'] == condition and r['object_count'] == count),
                key=lambda r: r['id'])
            selected.extend(rng.sample(population, quota))
            quotas[f'{condition}/{count}'] = quota
    selected.sort(key=lambda r: (r['condition'], r['object_count'], r['id']))
    pairs, unswappable = make_pairs(selected)
    output.mkdir(exist_ok=True)
    write_jsonl(output / 'references.jsonl', selected)
    write_jsonl(output / 'pairs.jsonl', pairs)
    write_jsonl(output / 'images.jsonl', [
        dict(model='reference', id=r['id'], seed=101, image='../' + r['reference_image'])
        for r in selected
    ])
    # This only checks known labels and counterfactual construction, not any VLM.
    for pair in pairs:
        reference = next(r for r in selected if r['id'] == pair['reference_id'])
        actual = counts(reference['objects'])
        assert (pair['target_joint_counts'] == actual) == pair['expected_accept']
        assert set(pair['target_joint_counts']) == set(CELLS)
        assert all(type(v) is int and v >= 0 for v in pair['target_joint_counts'].values())
        assert 3 <= pair['target_count'] <= 6
        if pair['kind'] == 'binding_swap':
            assert marginals(pair['target_joint_counts']) == marginals(actual)
            assert pair['target_count'] == reference['object_count']
    statistics = dict(
        version='cobind-count-calibration-v1', benchmark_version=benchmark_manifest['version'],
        source='data/development.jsonl', source_sha256=digest(source),
        seed=SEED, selection='Sort by id within condition/N cell; fixed seeded random.sample; no model results used.',
        cell_quotas=quotas, reference_images=len(selected),
        condition_counts=dict(Counter(r['condition'] for r in selected)),
        object_count_counts=dict(Counter(r['object_count'] for r in selected)),
        pair_counts=dict(Counter(p['kind'] for p in pairs)), pair_total=len(pairs),
        binding_swap_unavailable_ids=unswappable,
        binding_swap_scope='Requires two occupied cells with distinct colors and shapes; no artificial pairs for ineligible scenes.',
        no_additional_independent_dataset_samples=True,
        split_note='These 100 reference images are a subset of the 200 development scenes, not test scenes.',
        files_sha256={name: digest(output / name) for name in ['references.jsonl', 'pairs.jsonl', 'images.jsonl']},
    )
    (output / 'stats.json').write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: statistics[k] for k in ['reference_images', 'pair_counts', 'pair_total']}, indent=2))


def verify_oracle(benchmark):
    """Check pair labels against perfect annotation-derived counts, never a VLM."""
    benchmark = Path(benchmark).resolve()
    sys.path.insert(0, str(benchmark / 'evaluation'))
    from protocol import VERSION, SCORING_VERSION, QUESTIONS, assess, expected_answer
    calibration = benchmark / 'calibration'
    references = {r['id']: r for r in map(json.loads, (calibration / 'references.jsonl').read_text().splitlines())}
    pairs = [json.loads(line) for line in (calibration / 'pairs.jsonl').read_text().splitlines()]
    verified = Counter()
    for pair in pairs:
        reference = references[pair['reference_id']]
        answers = {q['id']: expected_answer(reference['objects'], q) for q in QUESTIONS}
        target = []
        for cell, count in pair['target_joint_counts'].items():
            color, shape = cell.split('/')
            target.extend(dict(color=color, shape=shape) for _ in range(count))
        score = assess(target, answers)
        assert score['evaluation_valid']
        assert score['core_full_match'] is pair['expected_accept']
        if pair['kind'] == 'binding_swap':
            assert marginals(pair['target_joint_counts']) == marginals(counts(reference['objects']))
            assert score['marginal_pass_joint_fail'] and score['error_type'] == 'binding'
        if pair['kind'] == 'count_mismatch':
            assert not score['total_count_acc'] and score['error_type'] == 'total'
        verified[pair['kind']] += 1
    result = dict(
        oracle_only=True, no_model_inference=True, protocol=VERSION, scoring_version=SCORING_VERSION,
        input='Perfect counts derived from original scene annotations; not Qwen answers.',
        pairs_sha256=digest(calibration / 'pairs.jsonl'),
        references_sha256=digest(calibration / 'references.jsonl'),
        protocol_sha256=digest(benchmark / 'evaluation/protocol.py'),
        checked_pair_counts=dict(verified), checked_pairs=len(pairs),
        all_positive_targets_accepted=True, all_negative_targets_rejected=True,
        all_binding_swap_marginals_preserved=True,
        all_binding_swap_targets_classified_as_binding_error=True,
        all_count_mismatch_targets_classified_as_total_error=True,
        note='Verifies construction and CPU scoring logic only; no empirical judge accuracy claim.',
    )
    (calibration / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Oracle-only construction/scoring check passed:', len(pairs), 'pairs; no Qwen inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--verify-oracle', action='store_true', help='Also check CPU scoring against perfect annotation counts; does not call Qwen')
    args = parser.parse_args()
    build(args.benchmark)
    if args.verify_oracle:
        verify_oracle(args.benchmark)
