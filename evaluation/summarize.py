"""Compute CoBind-Count reports from saved Qwen answers on CPU."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path

from protocol import (VERSION, SCORING_VERSION, QUESTIONS, atomic_json, jsonl,
                      read_answers, sha256, validate_answers, write_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Original Qwen results; defaults to --output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=0)
    parser.add_argument('--bootstrap-seed', type=int, default=20261106)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error('--bootstrap-samples must be nonnegative')
    source = (args.source or args.output).resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.output / 'run.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if source != args.output and (source / 'run.lock').exists():
            source_lock = stack.enter_context((source / 'run.lock').open('r'))
            fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        config = json.loads((source / 'judge_config.json').read_text(encoding='utf-8'))
        # Validate observation questions and the parsing of every raw answer.
        if config['protocol'] != VERSION or config['questions'] != QUESTIONS:
            raise ValueError('Saved observations use different Qwen questions or protocol')
        snapshot = json.loads((source / 'benchmark.json').read_text(encoding='utf-8'))
        tasks = jsonl(source / 'tasks.jsonl')
        records = read_answers(source / 'answers.jsonl')
        validate_answers(records, tasks, snapshot['prompts'])
        result = write_summary(args.output, snapshot, tasks, records,
                               bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
        atomic_json(args.output / 'scoring_provenance.json', dict(
            observation_protocol=VERSION, scoring_version=SCORING_VERSION,
            source_results=str(source), qwen_called=False, images_loaded=False,
            source_observation_code_sha256=config['code_sha256'],
            source_sha256={name: sha256(source / name) for name in
                           ('judge_config.json', 'benchmark.json', 'tasks.jsonl', 'answers.jsonl')},
            scoring_code_sha256={name: sha256(Path(__file__).with_name(name))
                                 for name in ('protocol.py', 'summarize.py')},
        ))
    print(json.dumps({key: result[key] for key in ('status', 'partial', 'invalid_answers', 'pending_questions',
                                                  'unscored_images', 'inconsistent_images')}))


if __name__ == '__main__':
    main()
