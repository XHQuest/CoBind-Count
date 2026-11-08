"""Build CoBind-Count indexes and its content manifest from the published splits."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

VERSION = 'cobind-count-v1.0'
COLORS = ('gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow')
SHAPES = ('cube', 'cylinder', 'sphere')
SEEDS = [101, 202, 303, 404]
SPLIT_COUNTS = {'train': 1000, 'development': 200, 'test': 800}
GROUPS = {
    (False, False): 'basic',
    (False, True): 'repeated_only',
    (True, False): 'same_shape_diff_colors',
    (True, True): 'same_shape_diff_colors_repeated',
}
PACKAGE_DIRECTORIES = {'data', 'prompts', 'provenance', 'evaluation', 'calibration', 'results'}
PACKAGE_ROOT_FILES = {
    '.gitattributes', '.gitignore', 'LICENSE-DATA.txt', 'README.md', 'README.zh-CN.md',
    'build_dataset.py', 'statistics.json', 'subsets.json', 'questions.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def package_files(root):
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in PACKAGE_ROOT_FILES:
            yield path
        elif relative.parts[0] in PACKAGE_DIRECTORIES:
            if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
                continue
            if path.suffix not in ('.pyc', '.log', '.lock', '.tmp'):
                yield path


def validate_row(row, split):
    if row['benchmark_version'] != VERSION or row['role'] != split:
        raise ValueError('Incorrect version or split: ' + row['id'])
    if row['text'] != row['prompt']:
        raise ValueError('Prompt fields differ: ' + row['id'])
    bag = tuple(sorted((obj['color'], obj['shape']) for obj in row['objects']))
    counts = Counter(bag)
    allowed_counts = (4, 5, 6) if split == 'train' else (3, 4, 5, 6)
    if len(bag) != row['object_count'] or len(bag) not in allowed_counts:
        raise ValueError('Incorrect object count: ' + row['id'])
    if any(c not in COLORS or s not in SHAPES for c, s in counts):
        raise ValueError('Unknown color or shape: ' + row['id'])
    core_id = hashlib.sha256(json.dumps(bag, separators=(',', ':')).encode()).hexdigest()
    core_counts = [{'color': c, 'shape': s, 'count': n} for (c, s), n in sorted(counts.items())]
    difficulty = {
        'same_shape_different_colors': any(len({c for c, sh in counts if sh == s}) > 1 for s in SHAPES),
        'same_color_different_shapes': any(len({s for co, s in counts if co == c}) > 1 for c in COLORS),
        'repeated_color_shape': any(n > 1 for n in counts.values()),
        'distinct_shapes': len({s for c, s in counts}),
        'distinct_colors': len({c for c, s in counts}),
        'max_color_shape_multiplicity': max(counts.values()),
    }
    expected = {
        'core_target_id': core_id, 'core_target_counts': core_counts, 'difficulty': difficulty,
        'target_joint_counts': {c + '/' + s: counts[c, s] for c in COLORS for s in SHAPES},
        'diagnostic_group': GROUPS[difficulty['same_shape_different_colors'], difficulty['repeated_color_shape']],
    }
    for field, value in expected.items():
        if row[field] != value:
            raise ValueError('Annotation mismatch: ' + row['id'] + '/' + field)


def load_splits(root):
    splits = {}
    for split, expected_count in SPLIT_COUNTS.items():
        rows = read_jsonl(root / 'data' / (split + '.jsonl'))
        if len(rows) != expected_count:
            raise ValueError('Unexpected split size: ' + split)
        for row in rows:
            validate_row(row, split)
        splits[split] = rows
    order = json.loads((root / 'provenance/source_build/training_order.json').read_text())
    if order != [r['id'] for r in splits['train']]:
        raise ValueError('Training records differ from the published order')
    all_rows = [row for rows in splits.values() for row in rows]
    for field in ('id', 'prompt', 'image_sha256'):
        if len({row[field] for row in all_rows}) != len(all_rows):
            raise ValueError('Non-unique scene field: ' + field)
    overlaps = {}
    for i, first in enumerate(splits):
        for second in list(splits)[i + 1:]:
            counts = {name: len({r[field] for r in splits[first]} & {r[field] for r in splits[second]})
                      for name, field in [('scene_ids', 'id'), ('prompt_texts', 'prompt'),
                                          ('core_targets', 'core_target_id'), ('reference_image_sha256', 'image_sha256')]}
            if any(counts.values()):
                raise ValueError('Overlap between splits: ' + first + '/' + second)
            overlaps[first + '/' + second] = counts
    return splits, overlaps


def split_statistics(rows):
    return {
        'prompts': len(rows),
        'condition': dict(sorted(Counter(r['condition'] for r in rows).items())),
        'object_count': dict(sorted(Counter(str(r['object_count']) for r in rows).items())),
        'condition_by_count': dict(sorted(Counter(r['condition'] + '/' + str(r['object_count']) for r in rows).items())),
        'diagnostic_group': dict(sorted(Counter(r['diagnostic_group'] for r in rows).items())),
        'difficulty': {name: sum(r['difficulty'][name] for r in rows)
                       for name in ('same_shape_different_colors', 'same_color_different_shapes', 'repeated_color_shape')},
        'unique_scene_ids': len({r['id'] for r in rows}),
        'unique_core_targets': len({r['core_target_id'] for r in rows}),
    }


def write_indexes(root, splits):
    for split, rows in splits.items():
        prompts = [{'id': r['id'], 'prompt': r['prompt'], 'condition': r['condition'],
                    'object_count': r['object_count'], 'core_target_id': r['core_target_id'],
                    'seeds': SEEDS if split == 'test' else [101]} for r in rows]
        path = root / 'prompts' / (split + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in prompts), encoding='utf-8')
    write_json(root / 'statistics.json', {split: split_statistics(rows) for split, rows in splits.items()})
    write_json(root / 'subsets.json', {
        split: {
            'condition': {c: [r['id'] for r in rows if r['condition'] == c] for c in ('A', 'B')},
            'object_count': {str(n): [r['id'] for r in rows if r['object_count'] == n] for n in range(3, 7)},
            'diagnostic_group': {name: [r['id'] for r in rows if r['diagnostic_group'] == name] for name in GROUPS.values()},
            'same_color_different_shapes': [r['id'] for r in rows if r['difficulty']['same_color_different_shapes']],
        } for split, rows in splits.items()
    })


def finalize(root):
    splits, overlaps = load_splits(root)
    all_rows = [row for rows in splits.values() for row in rows]
    image_hashes = {r['reference_image']: r['image_sha256'] for r in all_rows}
    for row in splits['test']:
        if digest(root / row['reference_image']) != row['image_sha256']:
            raise ValueError('Reference image hash mismatch: ' + row['id'])
    write_json(root / 'provenance/reference_image_hashes.json', image_hashes)
    write_json(root / 'provenance/export_verification.json', {
        'version': VERSION,
        'split_counts': {split: len(rows) for split, rows in splits.items()},
        'unique_scene_ids': len({r['id'] for r in all_rows}),
        'unique_core_targets': len({r['core_target_id'] for r in all_rows}),
        'unique_prompt_texts': len({r['prompt'] for r in all_rows}),
        'cross_split_overlaps': overlaps,
        'training_order_sha256': digest(root / 'provenance/source_build/training_order.json'),
        'training_membership_and_order_match': True,
        'targets_and_difficulty_labels_match_objects': True,
        'split_files_sha256': {f'data/{split}.jsonl': digest(root / 'data' / (split + '.jsonl')) for split in splits},
        'reference_images_verified': len(splits['test']),
        'packaged_reference_image_splits': ['test'],
    })
    spec = importlib.util.spec_from_file_location('cobind_packaging_protocol', root / 'evaluation/protocol.py')
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    if protocol.BENCHMARK_VERSION != VERSION:
        raise ValueError('Dataset and evaluation versions differ')
    write_json(root / 'questions.json', {
        'protocol': protocol.VERSION, 'questions': protocol.QUESTIONS,
        'judge_sees_target_prompt': False, 'judge_sees_expected_answers': False,
        'scope': ['count', 'color', 'shape'],
    })
    files = {p.relative_to(root).as_posix(): digest(p) for p in package_files(root)}
    test_images = sorted({r['reference_image'] for r in splits['test']})
    files.update({p: image_hashes[p] for p in test_images})
    write_json(root / 'manifest.json', {
        'name': 'CoBind-Count', 'version': VERSION,
        'protocol_version': protocol.VERSION, 'scoring_version': protocol.SCORING_VERSION,
        'source': 'https://cs.stanford.edu/people/jcjohns/clevr/', 'data_license': 'CC-BY-4.0',
        'source_provenance': 'provenance/data_sources.json',
        'splits': {**{split: f'data/{split}.jsonl' for split in splits}, 'calibration': 'calibration/references.jsonl'},
        'split_counts': {split: len(rows) for split, rows in splits.items()},
        'total_unique_prompts': len({r['prompt'] for r in all_rows}),
        'total_unique_core_targets': len({r['core_target_id'] for r in all_rows}),
        'test_generation_seeds': SEEDS, 'development_generation_seeds': [101], 'calibration_generation_seeds': [101],
        'vocabulary': {'colors': list(COLORS), 'shapes': list(SHAPES)},
        'scored_attributes': ['count', 'color', 'shape'], 'unscored_prompt_attributes': ['size', 'material'],
        'spatial_layout_scored': False, 'reference_image_is_generation_input': False,
        'report_groups': ['all', 'N3', 'N4', 'N5', 'N6'],
        'calibration_reference_count': len(read_jsonl(root / 'calibration/references.jsonl')),
        'questions_available': True,
        'packaged_reference_image_count': len(test_images), 'packaged_reference_image_splits': ['test'],
        'packaged_file_count_excluding_manifest': len(files), 'files_sha256': files,
    })
    print('Package manifest:', len(files), 'files;', len(test_images), 'reference images', flush=True)


def build(source, output):
    splits, _ = load_splits(source)
    if source.resolve() != output.resolve():
        output.mkdir(parents=True, exist_ok=True)
        for path in package_files(source):
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for row in splits['test']:
            destination = output / row['reference_image']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / row['reference_image'], destination)
    write_indexes(output, splits)
    finalize(output)
    print('Split sizes:', {split: len(rows) for split, rows in splits.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent,
                        help='Published benchmark directory containing the fixed data splits')
    parser.add_argument('--output', type=Path, help='Output directory; defaults to --source')
    parser.add_argument('--finalize', action='store_true', help='Refresh content hashes without rebuilding indexes')
    args = parser.parse_args()
    output = args.output if args.output is not None else args.source
    if args.finalize:
        finalize(output)
    else:
        build(args.source, output)


if __name__ == '__main__':
    main()
