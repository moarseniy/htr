#!/usr/bin/env python3
"""Convert the School Notebooks RU COCO annotations to page-level JSONL manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


TEXT_CATEGORIES = {"pupil_text", "pupil_comment", "teacher_comment"}
LINE_CATEGORY = "text_line"


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polygon_bbox(segmentation) -> tuple[float, float, float, float] | None:
    points = []
    if not isinstance(segmentation, list):
        return None
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        points.extend(zip(polygon[0::2], polygon[1::2]))
    if not points:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def line_for_word(word_box, line_boxes, image_height: int) -> int | None:
    """Assign a word polygon to the closest annotated text-line polygon."""
    x, y = bbox_center(word_box)
    best = None
    for index, line_box in enumerate(line_boxes):
        x_gap = max(0.0, line_box[0] - x, x - line_box[2])
        y_gap = max(0.0, line_box[1] - y, y - line_box[3])
        score = y_gap + 0.25 * x_gap
        if best is None or score < best[0]:
            best = (score, index)
    if best is None or best[0] > max(120.0, image_height * 0.08):
        return None
    return best[1]


def assemble_page_gt(annotations: list[dict], categories: dict[int, str], width: int, height: int):
    line_boxes = []
    for annotation in annotations:
        if categories.get(annotation.get("category_id")) == LINE_CATEGORY:
            box = polygon_bbox(annotation.get("segmentation"))
            if box is not None:
                line_boxes.append(box)

    lines = [{"box": box, "words": []} for box in line_boxes]
    fallback_groups = defaultdict(list)
    text_annotations = 0
    for index, annotation in enumerate(annotations):
        if categories.get(annotation.get("category_id")) not in TEXT_CATEGORIES:
            continue
        attributes = annotation.get("attributes") or {}
        text = attributes.get("translation") if isinstance(attributes, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        box = polygon_bbox(annotation.get("segmentation"))
        if box is None:
            continue
        text_annotations += 1
        word = {"text": text.strip(), "box": box, "source_index": index}
        line_index = line_for_word(box, line_boxes, height)
        if line_index is not None:
            lines[line_index]["words"].append(word)
        else:
            group_id = annotation.get("group_id")
            fallback_key = ("group", str(group_id)) if group_id is not None else ("word", index)
            fallback_groups[fallback_key].append(word)

    for words in fallback_groups.values():
        all_x = [w["box"][i] for w in words for i in (0, 2)]
        all_y = [w["box"][i] for w in words for i in (1, 3)]
        lines.append({"box": (min(all_x), min(all_y), max(all_x), max(all_y)), "words": words})

    lines = [line for line in lines if line["words"]]
    for line in lines:
        line["words"].sort(key=lambda word: (bbox_center(word["box"])[0], word["source_index"]))
        line["text"] = " ".join(word["text"] for word in line["words"])

    # Most School Notebooks images are open spreads. Read the left page from top
    # to bottom, then the right page. Use ordinary top-to-bottom order otherwise.
    left = [line for line in lines if bbox_center(line["box"])[0] < width * 0.44]
    right = [line for line in lines if bbox_center(line["box"])[0] > width * 0.56]
    middle = [line for line in lines if line not in left and line not in right]
    is_spread = len(left) >= 5 and len(right) >= 5 and len(middle) <= len(lines) * 0.2
    if is_spread:
        left = [line for line in lines if bbox_center(line["box"])[0] < width / 2]
        right = [line for line in lines if bbox_center(line["box"])[0] >= width / 2]
        ordered = sorted(left, key=lambda line: (bbox_center(line["box"])[1], bbox_center(line["box"])[0]))
        ordered += sorted(right, key=lambda line: (bbox_center(line["box"])[1], bbox_center(line["box"])[0]))
        reading_order = "spread-left-page-then-right-page"
    else:
        ordered = sorted(lines, key=lambda line: (bbox_center(line["box"])[1], bbox_center(line["box"])[0]))
        reading_order = "top-to-bottom-left-to-right"
    return "\n".join(line["text"] for line in ordered), text_annotations, reading_order


def convert_split(root: Path, split: str, output: Path):
    annotation_path = root / f"annotations_{split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(category["id"]): category["name"] for category in dataset.get("categories", [])}
    image_root = root / "images"
    annotations_by_image = defaultdict(list)
    for annotation in dataset.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    rows = []
    skipped = defaultdict(int)
    used_ids = set()
    for image in dataset.get("images", []):
        image_id = image["id"]
        filename = image["file_name"]
        image_path = image_root / filename
        if not image_path.is_file():
            skipped["image_missing"] += 1
            continue
        gt, annotation_count, reading_order = assemble_page_gt(
            annotations_by_image.get(image_id, []), categories,
            int(image["width"]), int(image["height"]),
        )
        if not gt.strip():
            skipped["empty_gt"] += 1
            continue
        task_id = f"school-notebooks-{split}-{image_id}"
        if task_id in used_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        used_ids.add(task_id)
        rows.append({
            "task_id": task_id,
            "image": str(image_path.resolve()),
            "gt": gt,
            "image_sha256": image_hash(image_path),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "source_dataset": "school_notebooks_RU",
            "source_split": split,
            "text_annotation_count": annotation_count,
            "reading_order": reading_order,
        })

    if not rows:
        raise ValueError(f"No valid labeled images found in {annotation_path}")
    rows.sort(key=lambda row: row["task_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    print(f"Split={split}; images={len(dataset.get('images', []))}; manifest_rows={len(rows)}; skipped={dict(skipped)}")
    print(f"Manifest: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/arseniy/data/school_notebooks_RU")
    parser.add_argument("--split", choices=["val", "test", "train"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_split(Path(args.root).expanduser().resolve(), args.split,
                  Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
