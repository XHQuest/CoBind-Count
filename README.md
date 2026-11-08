# CoBind-Count

English | [简体中文](README.zh-CN.md)

**A CoGenT-derived diagnostic benchmark for color–shape binding and exact counting.**

CoBind-Count evaluates **attribute binding and exact counting across multiple instances of the same object category** in text-to-image generation. When a scene contains several objects of the same shape, a model must assign the correct colors to those instances and produce the exact count of every color–shape combination.

For example, suppose the target is “two red spheres, one blue sphere, and one blue cube.” An image with “one red sphere, two blue spheres, and one red cube” has the correct total number of objects, color totals, and shape totals, but the bindings are wrong. CoBind-Count compares joint color–shape counts, so this image does not receive a full match.

| Component | Identifier |
|---|---|
| Dataset | `cobind-count-v1.0` |
| Question protocol | `cobind-count-qwen-v1` |
| Scoring protocol | `cobind-count-score-v1` |

Each model receives three point scores—**Full Match, Count Acc, and Joint Count F1**—reported for five groups: **overall, 3 objects, 4 objects, 5 objects, and 6 objects**.

## Scope and source

Reference images and scene annotations come from [CLEVR-CoGenT](https://cs.stanford.edu/people/jcjohns/clevr/). This benchmark provides text-to-image prompts, fixed dataset splits, joint-count targets, diagnostic labels, and Qwen evaluation tools.

Core scoring covers **count / color / shape**. Prompts also describe size and material, but those attributes are outside the core scoring scope; prompts do not specify spatial relationships. `core_full_match` means that all count, color, and shape requirements are satisfied, not that every attribute in the full prompt is correct.

Generation models receive only the text prompt. Reference images support annotation traceability and evaluator calibration; they are not generation inputs. Generated images need not reproduce the reference layout or camera angle. Evaluation operates on an unordered collection of objects.

## Dataset splits

| Split | Prompts | Condition A | Condition B | 3 objects | 4 objects | 5 objects | 6 objects |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,000 | 1,000 | 0 | 0 | 335 | 346 | 319 |
| development | 200 | 100 | 100 | 50 | 50 | 50 | 50 |
| test | 800 | 400 | 400 | 200 | 200 | 200 | 200 |
| Total | 2,000 | 1,500 | 500 | 250 | 585 | 596 | 569 |

Each condition × object-count cell contains 25 development prompts and 100 test prompts. Formal evaluation uses the 800 test prompts. The training split supports model fine-tuning; scores are computed on the test split. A subset of 100 development scenes is also used for reference-image calibration; it does not add independent scenes.

The training split contains only scenes with 4–6 objects. The 3-object test group is intended to assess whether a model or method retains the ability to generate combinations with fewer objects after fine-tuning on this training split.

The repository includes prompts and annotations for all three splits, but bundles reference PNGs **only for the 800 test scenes**. Training and development reference PNGs are not included.

All 2,000 scene IDs and prompt texts are unique. Scene IDs, prompt texts, reference-image hashes, and core color–shape multiset targets do not overlap between splits. Data sources, selection records, and content hashes are in `provenance/`.

Conditions A and B follow the CoGenT combination rules:

| Condition | Cube colors | Cylinder colors | Sphere colors |
|---|---|---|---|
| A | gray, blue, brown, yellow | red, green, purple, cyan | All 8 colors |
| B | red, green, purple, cyan | gray, blue, brown, yellow | All 8 colors |

### Diagnostic labels

A “repeated combination” means that more than one object has the same color and shape. The following four groups are mutually exclusive and cover every record:

| Label | Same shape, different colors | Repeated combination | train | development | test |
|---|---|---|---:|---:|---:|
| `basic` | No | No | 0 | 10 | 34 |
| `repeated_only` | No | Yes | 0 | 20 | 86 |
| `same_shape_diff_colors` | Yes | No | 637 | 103 | 416 |
| `same_shape_diff_colors_repeated` | Yes | Yes | 363 | 67 | 264 |

The test split contains 680 prompts with the same shape in different colors and 350 with repeated combinations. Another 298 prompts have the “same color, different shapes” label, which can overlap with the four groups. A/B conditions and diagnostic labels support analysis; the default score table reports overall results and object-count groups.

All 1,000 training prompts contain both the same shape in different colors and the same color on different shapes. Development and test also include scenes without these diagnostic properties.

## Data format

```text
CoBind-Count/
├── data/             # Train, development, and test JSONL
├── prompts/          # Compact JSONL for generation programs
├── images/           # 800 test reference PNGs
├── evaluation/       # Qwen questions, scoring, and reporting tools
├── calibration/      # Development reference records and target pairs
├── provenance/       # Data sources, selection settings, and image hashes
├── results/          # Evaluation of the 800 test reference images
├── questions.json    # 29 counting questions
├── statistics.json   # Dataset distributions
├── subsets.json      # Diagnostic subset IDs
├── manifest.json     # Versions, splits, fixed seeds, and file SHA256 hashes
└── build_dataset.py  # Rebuild metadata from the fixed split records
```

Each record in `data/*.jsonl` contains these fields:

| Field | Meaning |
|---|---|
| `id` | String scene ID; do not substitute a row number |
| `prompt` / `text` | Generation prompt; both fields contain the same text |
| `condition` / `object_count` | A/B condition and target object count |
| `objects` | Original object attributes and scene annotations |
| `target_joint_counts` | All 24 color–shape counts, including zeros |
| `core_target_counts` | Sparse representation of the joint-count target |
| `core_target_id` | Core target identifier |
| `diagnostic_group` / `difficulty` | Diagnostic group and difficulty labels |
| `reference_image` / `image_sha256` | Reference-image path relative to this package and its content hash |

For example, the prompt for `CLEVR_valB_013250` is:

```text
A 3D scene with three objects: a large brown rubber cylinder; a large gray rubber cylinder; a large red metal cube.
```

Its core target is `brown/cylinder=1, gray/cylinder=1, red/cube=1`; the other 21 cells are zero.

`prompts/*.jsonl` contains the fields needed for generation: `id`, `prompt`, condition, object count, target identifier, and seed list. The 800 bundled test reference PNGs use the original CLEVR-CoGenT image content. Training and development records also contain reference-image paths and hashes; their PNGs can be obtained from CLEVR-CoGenT.

## Scoring protocol

Each image is evaluated with 29 independent questions: one total count, three shape counts, 24 joint color–shape counts, and one count of objects outside the vocabulary. Qwen sees only the image and the current question, not the generation prompt, target answers, method name, or answers to other questions.

The color vocabulary is gray, red, blue, green, brown, purple, cyan, and yellow. The shape vocabulary is cube, cylinder, and sphere. Gray includes silver; cyan includes turquoise/teal; a box counts as a cube and a ball as a sphere. Shadows, reflections, and background are not objects. Foreground objects that cannot be classified unambiguously are counted as `other`.

| Metric | Definition |
|---|---|
| `core_full_match` | All 24 joint counts are correct, `other=0`, total and shape counts are correct, and answers are internally consistent |
| `count_acc` | The total count and all three shape counts are correct |
| `joint_count_f1` | `2 × Σ min(target cell count, predicted cell count) / (target total + predicted total across 24 cells + other)` |

Complete, parseable responses receive scores. If total, shape, and joint counts contradict one another, Full Match is zero, while Count Acc and F1 are still computed from the observed counts. Contradictions are recorded as diagnostics.

For a target of two red spheres and one blue sphere, observing one red sphere and two blue spheres gives Full Match = 0 and F1 = 2/3. The main metrics do not average correctness across the 24 cells, which would allow numerous zero-target cells to inflate the score.

Each main metric is the per-image mean over images with an available score. A missing image, missing answer, or unparseable answer makes all three scores for that image `None`; that image is excluded from the metric mean. Reports also include the numbers of scored and unscored images and coverage. A complete result requires a score for every scheduled image. `evaluation_valid` records internal consistency and must not be used to exclude contradictory responses from Count Acc or F1.

### Five reporting groups

| Group | Test prompts | Reference images | Generated images per model |
|---|---:|---:|---:|
| Overall | 800 | 800 | 3,200 |
| 3 objects | 200 | 200 | 800 |
| 4 objects | 200 | 200 | 800 |
| 5 objects | 200 | 200 | 800 |
| 6 objects | 200 | 200 | 800 |

Groups are defined by the target object count. Each test prompt uses four fixed seeds: **101, 202, 303, and 404**. For a complete evaluation, the per-image mean equals averaging the four seeds within each prompt and then weighting prompts equally. Each reference image is evaluated once; reference-image scores measure agreement between the evaluator and known annotations.

## Usage

```bash
git clone https://github.com/XHQuest/CoBind-Count.git
cd CoBind-Count
```

### Prepare generated images

Read the 800 prompts from `prompts/test.jsonl` and generate one image per prompt for each of the four fixed seeds. Record the model-weight identifier, inference steps, resolution, and guidance settings. Assign a distinct model label to each evaluated model and configuration.

The image manifest is UTF-8 JSONL, with one image per line:

```json
{"model":"model_a","id":"CLEVR_valB_013250","seed":101,"image":"images/model_a/CLEVR_valB_013250_101.png"}
```

`image` may be an absolute path or a path relative to the manifest. Formal generation evaluation requires all 800 prompts × 4 seeds; multiple models can share one manifest. A reference-image manifest points to the corresponding reference images, with one record per image. Its `seed=101` is only an indexing label.

### Run Qwen counting

Use a local Qwen3.5-9B environment with compatible vLLM support. From this directory, run:

```bash
python evaluation/run_qwen_eval.py \
  --split test \
  --images-manifest /path/to/generated_images.jsonl \
  --model-path /path/to/Qwen3.5-9B \
  --output outputs/test_qwen
```

Fixed settings are BF16, temperature = 0, judge seed = 20261103, thinking disabled, at most 32 output tokens, and an image pixel-count range of 512²–1024². For reference images, explicitly pass `--seeds 101`; this single-seed task is marked separately from four-seed generation evaluation.

Run the same command to resume an interrupted evaluation; only missing questions are processed. The runner provides a tqdm progress bar, a directory lock, and durable writes after each chunk. It records data, images, code, model identity, and run settings. Invalid responses are not automatically retried, and raw answers are not semantically corrected.

### Generate reports

The reporting tool reads saved answers and computes scores on CPU:

```bash
python evaluation/summarize.py --output outputs/test_qwen
python evaluation/summarize.py \
  --source /path/to/observations \
  --output /path/to/report
```

Main outputs are `image_scores.jsonl`, `summary.json`, `summary.csv`, `summary.md`, and scoring-provenance records. See [evaluation/README.txt](evaluation/README.txt) for detailed input, resume, and output instructions.

The [800-reference-image evaluation](results/RESULTS.md) reports evaluator agreement with scene annotations, overall and by object count. Detailed scores are available in [Markdown](results/reference_test800/summary.md), [CSV](results/reference_test800/summary.csv), and [JSON](results/reference_test800/summary.json).

## Calibration target pairs

A fixed subset of 100 development reference images contains 50 images from each of A/B and 25 images for each object count from 3 to 6. Positive targets use the original joint counts. Negative targets introduce count errors or binding swaps that preserve marginal counts.

The repository provides the calibration records and target pairs, but does not bundle these 100 development PNGs. To run image-based calibration, obtain the corresponding images from the [official CLEVR-CoGenT dataset](https://cs.stanford.edu/people/jcjohns/clevr/) and place each PNG at the path in its `reference_image` field in `calibration/references.jsonl`, relative to the repository root. The recorded `image_sha256` identifies the expected original image content.

| Target type | Pairs | Expected decision |
|---|---:|---|
| Correct target | 100 | Accept |
| Count mismatch | 100 | Reject |
| Binding swap with preserved marginals | 92 | Reject |

Eight images do not support a binding swap and are listed in the calibration statistics. Multiple target pairs share the same Qwen counts for one image; they are not independent image samples. The evaluator sees the image and generic counting questions, not these targets.

Evaluator agreement is reported on the 800 test reference images; see [results/RESULTS.md](results/RESULTS.md) for measurements and settings.

## Dataset construction

The published package is ready to use and includes all split records and the 800 test reference PNGs. To obtain the optional training and development reference PNGs from the official CLEVR-CoGenT archive, run:

```bash
python3 provenance/source_build/prepare_dataset.py \
  --archive /path/to/CLEVR_CoGenT_v1.0.zip \
  --output . \
  --splits train development
```

The tool uses the fixed scene IDs in `data/*.jsonl`, checks annotations and PNG hashes against the official archive, and extracts images to their recorded paths.

To rebuild prompts, statistics, subsets, questions, verification records, and the file manifest from the fixed split records, run:

```bash
python3 build_dataset.py
```

To write a separate package directory:

```bash
python3 build_dataset.py --output /path/to/CoBind-Count
```

`--source` selects a source package directory and defaults to this repository. `--finalize` refreshes the file manifest. Split membership and prompt text are defined by the published `data/*.jsonl` files; packaging does not sample new scenes. Data sources and construction details are in [provenance/README.md](provenance/README.md).

## Attribution and license

Original images and scene annotations come from **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning** (CVPR 2017) and its CoGenT data, by Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick.

The [official data page](https://cs.stanford.edu/people/jcjohns/clevr/) distributes the data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CoBind-Count’s derived content includes scene selection, dataset splits, text prompts, joint-count labels, diagnostic labels, and calibration targets. Reference PNGs retain the source image content. This is an independent project, not an official CLEVR release. See [LICENSE-DATA.txt](LICENSE-DATA.txt) for attribution and license details.
