#!/usr/bin/env python3
"""Recover original reference PNGs for the fixed CoBind-Count scene records.

This tool uses the released split membership. It does not select scenes, generate
captions, or change annotations. The official CLEVR-CoGenT archive is required.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Official CLEVR_CoGenT_v1.0.zip')
    parser.add_argument('--output', type=Path, required=True, help='Destination for the images/ directory')
    parser.add_argument('--benchmark', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--splits', choices=('train', 'development', 'test'), nargs='+', default=['test'])
    args = parser.parse_args()
    root = args.benchmark.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    grouped = defaultdict(list)
    seen = set()
    for split in dict.fromkeys(args.splits):
        data = (root / manifest['splits'][split]).read_bytes()
        if digest(data) != manifest['files_sha256']['data/' + split + '.jsonl']:
            raise ValueError('Split content differs from the release manifest: ' + split)
        for line in data.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in seen:
                raise ValueError('Duplicate scene ID: ' + row['id'])
            seen.add(row['id'])
            grouped[row['source_split']].append(row)
    output = args.output.resolve()
    total, completed = len(seen), 0
    with zipfile.ZipFile(args.archive) as archive:
        for source_split, rows in sorted(grouped.items()):
            if source_split not in ('trainA', 'valA', 'valB'):
                raise ValueError('Unknown CLEVR-CoGenT source split')
            member = 'CLEVR_CoGenT_v1.0/scenes/CLEVR_' + source_split + '_scenes.json'
            scenes = json.loads(archive.read(member))['scenes']
            selected = {row['image_filename'] for row in rows}
            by_name = {scene['image_filename']: scene for scene in scenes
                       if scene['image_filename'] in selected}
            del scenes
            for row in rows:
                scene = by_name[row['image_filename']]
                if row['id'] != Path(row['image_filename']).stem:
                    raise ValueError('Scene ID and filename differ: ' + row['id'])
                for field in ('objects', 'relationships', 'directions', 'image_index'):
                    if row[field] != scene[field]:
                        raise ValueError('Annotation differs from official source: ' + row['id'] + '/' + field)
                relative = Path('images') / source_split / row['image_filename']
                if relative.as_posix() != row['reference_image'] or Path(row['image_filename']).name != row['image_filename']:
                    raise ValueError('Unexpected reference image path: ' + row['id'])
                image = archive.read('CLEVR_CoGenT_v1.0/' + relative.as_posix())
                if digest(image) != row['image_sha256']:
                    raise ValueError('Reference image hash mismatch: ' + row['id'])
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(target.read_bytes()) != row['image_sha256']:
                        raise ValueError('Destination contains different image bytes: ' + str(target))
                else:
                    temporary = target.with_suffix('.png.tmp')
                    temporary.write_bytes(image)
                    temporary.replace(target)
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == total:
                    print(f'References: {completed}/{total}', flush=True)
    print('Reference recovery complete:', output, flush=True)


if __name__ == '__main__':
    main()
