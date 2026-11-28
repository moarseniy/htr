#!/usr/bin/env python3
"""Convert benchmark train manifests to the ms-swift multimodal SFT JSONL format."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


PROMPT = (
    "Выполни точную транскрипцию ВСЕГО текста на изображении документа "
    "(включая рукописный и печатный). Прочитай все строки в естественном порядке. "
    "Не исправляй орфографию, не угадывай отсутствующие слова, не пересказывай, "
    "не добавляй комментарии или Markdown. Сохраняй цифры и пунктуацию. "
    "Верни только распознанный текст."
)


def row_split(row: dict) -> str | None:
    return row.get("source_split") or row.get("split")


def get_images(row: dict, manifest_path: Path) -> list[str]:
    values = row.get("images")
    if not isinstance(values, list) or not values:
        image = row.get("image")
        values = [image] if isinstance(image, str) and image else []
    if not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"Task {row.get('task_id')!r} has no usable image path")

    paths = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found for task {row.get('task_id')!r}: {path}")
        paths.append(str(path))
    return paths


def convert(manifests: list[Path], output: Path, expected_split: str):
    output_rows = []
    seen_ids = set()
    source_counts = Counter()
    image_count = 0

    for manifest in manifests:
        manifest = manifest.expanduser().resolve()
        with manifest.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id", ""))
                if not task_id:
                    raise ValueError(f"Missing task_id in {manifest}:{line_number}")
                if task_id in seen_ids:
                    raise ValueError(f"Duplicate task_id across training manifests: {task_id}")
                seen_ids.add(task_id)

                split = row_split(row)
                if split != expected_split:
                    raise ValueError(
                        f"Refusing non-{expected_split} row {task_id!r} "
                        f"(manifest split is {split!r})"
                    )

                gt = row.get("gt")
                if not isinstance(gt, str) or not gt.strip():
                    raise ValueError(f"Task {task_id!r} has empty/non-string gt")
                images = get_images(row, manifest)
                user_content = "\n".join("<image>" for _ in images) + "\n" + PROMPT
                output_rows.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": gt},
                    ],
                    "images": images,
                })
                source_counts[str(row.get("source_dataset", "unknown"))] += 1
                image_count += len(images)

    if not output_rows:
        raise ValueError("No training examples were found")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in output_rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)

    print(f"Training examples: {len(output_rows)}")
    print(f"Images: {image_count}")
    print(f"Datasets: {dict(source_counts)}")
    print(f"Manifest: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True,
                        help="One or more manifests; every row must explicitly be from the train split")
    parser.add_argument("--output", required=True, help="Output ms-swift JSONL dataset")
    parser.add_argument("--expected-split", default="train",
                        help="Required split label in each input row (default: train)")
    args = parser.parse_args()
    convert([Path(item) for item in args.manifest], Path(args.output), args.expected_split)


if __name__ == "__main__":
    main()
