"""Offline, resumable image-only counting with Qwen for CoBind-Count."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from protocol import (
    VERSION, SCORING_VERSION, QUESTIONS, atomic_json, load_inputs, parse_answer,
    read_answers, sha256, validate_answers, write_summary,
)

TEST_SEEDS = [101, 202, 303, 404]


def package_versions():
    versions = {}
    for name in ('vllm', 'torch', 'transformers', 'Pillow', 'tqdm'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_identity(path):
    """Record small config hashes and weight metadata, without rereading big weights."""
    configs = {}
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json',
                 'chat_template.jinja', 'model.safetensors.index.json'):
        file = path / name
        if file.is_file():
            configs[name] = sha256(file)
    weights = []
    for file in sorted(path.glob('*.safetensors')):
        stat = file.stat()
        weights.append(dict(name=file.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return dict(config_sha256=configs, weight_file_metadata=weights,
                note='Weight metadata is not a content hash. Supply a known --model-revision when available.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--split', choices=('development', 'test', 'calibration'), default='test')
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: test 101 202 303 404; development/calibration 101')
    parser.add_argument('--images-manifest', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-id', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--model-revision', default='local-unspecified')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Development only; absent images stay in the full declared denominator')
    parser.add_argument('--images-per-chunk', type=int, default=8)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--max-pixels', type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = TEST_SEEDS.copy() if args.split == 'test' else [101]
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    args.seeds = sorted(args.seeds)
    if args.images_per_chunk < 1 or args.max_num_seqs < 1 or args.max_pixels < 512 * 512:
        parser.error('Chunk/sequence counts must be positive; max-pixels must be at least 262144')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('gpu-memory-utilization must be between 0 and 1')
    args.model_path = args.model_path.resolve()
    if not args.model_path.is_dir():
        parser.error('model-path must be an existing local model directory')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    snapshot, tasks = load_inputs(
        args.dataset, args.images_manifest, args.split, args.seeds, args.allow_partial,
    )
    judge_settings = dict(
        model_id=args.model_id, dtype='bfloat16', temperature=0.0, judge_seed=20261103,
        enable_thinking=False, max_new_tokens=32, min_pixels=512 * 512,
        max_pixels=args.max_pixels, max_model_len=4096,
    )
    standard_settings = dict(
        model_id='Qwen/Qwen3.5-9B', dtype='bfloat16', temperature=0.0, judge_seed=20261103,
        enable_thinking=False, max_new_tokens=32, min_pixels=512 * 512,
        max_pixels=1024 * 1024, max_model_len=4096,
    )
    actual = dict(split=args.split, generation_seeds=args.seeds, allow_partial=args.allow_partial, **judge_settings)
    standard = dict(split='test', generation_seeds=TEST_SEEDS, allow_partial=False, **standard_settings)
    deviations = [dict(field=field, actual=actual[field], standard=value)
                  for field, value in standard.items() if actual[field] != value]
    standard_test = not deviations
    snapshot.update(standard_protocol=standard_test, standard_protocol_deviations=deviations,
                    judge_settings=judge_settings)
    if not standard_test:
        print('Nonstandard protocol settings: ' + json.dumps(deviations, ensure_ascii=False), flush=True)
    code_files = ['run_qwen_eval.py', 'protocol.py', 'summarize.py']
    spec = dict(
        protocol=VERSION, scoring_version=SCORING_VERSION, dataset_version=snapshot['version'], split=args.split,
        generation_seeds=args.seeds, standard_protocol=standard_test,
        standard_protocol_deviations=deviations,
        dataset_sha256=snapshot['prompts_sha256'],
        dataset_manifest_sha256=snapshot['source_manifest_sha256'],
        images_manifest_sha256=sha256(args.images_manifest),
        image_sha256={t['key']: t['image_sha256'] for t in tasks},
        model_path=str(args.model_path),
        model_revision=args.model_revision, model_identity=model_identity(args.model_path),
        software_versions=package_versions(),
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in code_files},
        **judge_settings, max_num_batched_tokens=32768,
        tensor_parallel_size=1, max_num_seqs=args.max_num_seqs,
        images_per_chunk=args.images_per_chunk,
        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
        flashinfer_sampler=os.environ.get('VLLM_USE_FLASHINFER_SAMPLER', 'default'),
        questions=QUESTIONS,
        message_template=[{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': '<one QUESTIONS entry text, including counting rules>'}]}],
        independent_questions=True, judge_sees_full_generation_prompt=False,
        judge_sees_expected_answer=False, judge_sees_model_label=False,
        invalid_policy='Complete inconsistent counts retain count_acc/F1 and full_match=0; missing or invalid counts are unscored. No automatic retry.',
        evaluated_scope=['count', 'color', 'shape'],
        unevaluated_scope=['size', 'material', 'spatial_layout'],
        allow_partial=args.allow_partial,
        generation_configuration='Use separate model labels for different generation configurations, e.g. model_a and model_b; record generation settings separately.',
    )
    config = args.output / 'judge_config.json'
    if config.exists():
        if json.loads(config.read_text()) != spec:
            raise ValueError('Configuration does not match the saved run; use a separate output directory.')
    else:
        if (args.output / 'answers.jsonl').exists():
            raise ValueError('answers.jsonl exists without its frozen judge_config.json')
        atomic_json(config, spec)
    atomic_json(args.output / 'benchmark.json', snapshot)
    tasks_temp = args.output / 'tasks.jsonl.tmp'
    tasks_temp.write_text(''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks), encoding='utf-8')
    tasks_temp.replace(args.output / 'tasks.jsonl')
    records = read_answers(args.output / 'answers.jsonl', repair_tail=True)
    validate_answers(records, tasks, snapshot['prompts'])
    expected = len(tasks) * len(QUESTIONS)

    def progress(status, stage):
        atomic_json(args.output / 'progress.json', dict(
            status=status, stage=stage, recorded_answers=len(records),
            scheduled_answers=expected, listed_images=len(tasks),
            expected_benchmark_images=snapshot['expected_images'],
            complete_image_coverage=snapshot['complete_image_coverage'],
            standard_protocol=standard_test, split=args.split, generation_seeds=args.seeds,
            standard_protocol_deviations=deviations,
            elapsed_seconds=round(time.time() - started, 1),
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        ))

    summary = write_summary(args.output, snapshot, tasks, records)
    if len(records) == expected:
        progress(summary['status'], 'all_scheduled_answers_present_no_model_loaded')
        print('All scheduled answers exist. Report saved without loading Qwen.', flush=True)
        return
    progress('running', 'loading_model')
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from tqdm import tqdm

        processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
        chats = {
            q['id']: processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': q['text']}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=judge_settings['enable_thinking'])
            for q in QUESTIONS
        }
        llm = LLM(
            model=str(args.model_path), tokenizer=str(args.model_path), dtype=judge_settings['dtype'],
            tensor_parallel_size=1, max_model_len=judge_settings['max_model_len'], max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=32768, gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True, seed=judge_settings['judge_seed'],
            limit_mm_per_prompt={'image': 1, 'video': 0},
            mm_processor_kwargs={'min_pixels': judge_settings['min_pixels'], 'max_pixels': judge_settings['max_pixels']},
        )
        params = SamplingParams(temperature=judge_settings['temperature'],
                                max_tokens=judge_settings['max_new_tokens'], seed=judge_settings['judge_seed'])
        with (args.output / 'answers.jsonl').open('a', encoding='utf-8') as file, tqdm(
            total=expected, initial=len(records), desc='Qwen count questions', unit='question',
        ) as bar:
            for start in range(0, len(tasks), args.images_per_chunk):
                requests, metadata, images = [], [], []
                for task in tasks[start:start + args.images_per_chunk]:
                    todo = [q for q in QUESTIONS if (task['key'], q['id']) not in records]
                    if not todo:
                        continue
                    with Image.open(task['image']) as im:
                        image = im.convert('RGB')
                    images.append(image)
                    for question in todo:
                        requests.append(dict(
                            prompt=chats[question['id']], multi_modal_data={'image': image},
                            multi_modal_uuids={'image': task['image_sha256']},
                        ))
                        metadata.append((task, question))
                if not requests:
                    continue
                outputs = llm.generate(requests, params, use_tqdm=False)
                if len(outputs) != len(metadata):
                    raise RuntimeError('Inference returned a different number of answers than requests')
                for (task, question), result in zip(metadata, outputs):
                    response = result.outputs[0]
                    value, error = parse_answer(response.text, response.finish_reason)
                    row = dict(
                        key=task['key'], model=task['model'], id=task['id'], seed=task['seed'],
                        question_id=question['id'], question=question['question'],
                        image_sha256=task['image_sha256'], raw_response=response.text,
                        finish_reason=response.finish_reason, value=value, parse_error=error,
                        input_tokens=len(result.prompt_token_ids), output_tokens=len(response.token_ids),
                    )
                    file.write(json.dumps(row, ensure_ascii=False) + '\n')
                    records[(task['key'], question['id'])] = row
                file.flush()
                os.fsync(file.fileno())
                for image in images:
                    image.close()
                bar.update(len(metadata))
                summary = write_summary(args.output, snapshot, tasks, records)
                bar.set_postfix(unscored=summary['unscored_images'], inconsistent=summary['inconsistent_images'])
                progress('running', 'judging')
        progress(summary['status'], 'finished_scheduled_answers')
        print('Finished. See summary.md/json for point scores and five count groups; consistency diagnostics are recorded.', flush=True)
    except BaseException:
        write_summary(args.output, snapshot, tasks, records)
        progress('failed', 'interrupted_or_error')
        (args.output / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
