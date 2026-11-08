# Data provenance

Reference images and scene annotations originate from [CLEVR-CoGenT v1.0](https://cs.stanford.edu/people/jcjohns/clevr/), distributed under CC BY 4.0. CoBind-Count provides fixed splits, text prompts, joint color–shape counts, diagnostic labels, and calibration targets. Reference PNGs use the original source image content.

## Fixed splits

CoBind-Count contains 1,000 training records, 200 development records, and 800 test records. Training uses condition A and contains 335 scenes with 4 objects, 346 with 5 objects, and 319 with 6 objects. The 3-object test group assesses whether models or methods retain the ability to generate combinations with fewer objects after fine-tuning on this training split.

The published `data/train.jsonl`, `data/development.jsonl`, and `data/test.jsonl` define split membership, record order, prompt text, and targets. `source_build/training_order.json` lists the 1,000 training scene IDs in order. All scene IDs are unique. Scene IDs, prompt texts, reference-image hashes, and core color–shape targets do not overlap between splits. A core target is an unordered color–shape multiset including multiplicities.

## Provenance files

| File | Contents |
|---|---|
| `data_sources.json` | Official data source, source hashes, and split provenance |
| `selection_config.json` | Split definitions, selection criteria, and distributions |
| `reference_image_hashes.json` | SHA256 of each original reference PNG; only the 800 test PNGs are bundled |
| `export_verification.json` | Checks on scene identities, prompt text, targets, split separation, and image content |
| `source_build/training_order.json` | Fixed list of the 1,000 training scene IDs in order |
| `source_build/prepare_dataset.py` | Reference-image extraction for the published scene records |

## Extract reference images

The repository includes all 800 test reference PNGs. To extract the optional training and development reference PNGs, obtain the official `CLEVR_CoGenT_v1.0.zip` and run from the repository root:

```bash
python3 provenance/source_build/prepare_dataset.py \
  --archive /path/to/CLEVR_CoGenT_v1.0.zip \
  --output . \
  --splits train development
```

The script checks split files against the package manifest, compares scene annotations with the official archive, and verifies PNG hashes before writing images. Each PNG is written to its record's `reference_image` path under `--output`. Existing images with matching content are retained. The default `--splits test` extracts test images; `--splits train development test` covers all three splits. `--benchmark` selects another package directory.

## Rebuild package metadata

From the repository root, run:

```bash
python3 build_dataset.py
```

The script reads the fixed split records and rebuilds prompts, statistics, subsets, questions, verification records, and the file manifest. It checks joint counts against object annotations, split separation, and reference-image content hashes.

To write a separate package directory:

```bash
python3 build_dataset.py --output /path/to/CoBind-Count
```

`--source` selects a source package directory and defaults to this repository. `--finalize` refreshes the file manifest. The published split records define scene selection and prompt text; rebuilding metadata does not resample scenes.
