#!/usr/bin/env python3
"""Build document-level HWR200 manifests from annotations and page-image folders."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import re
from pathlib import Path


MODALITIES = {
    "scans": "Сканы",
    "light": "ФотоСветлое",
    "dark": "ФотоТемное",
}
SOURCES = ("FPR", "Originals", "Reuse")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def annotation_files(root: Path, source: str):
    base = root / "annotations" / source
    return sorted(
        p for p in base.rglob("*.json")
        if ".ipynb_checkpoints" not in p.parts and "-checkpoint" not in p.name
    )


def document_key(source: str, annotation_path: Path, root: Path) -> str:
    return annotation_path.relative_to(root).with_suffix("").as_posix()


def build_document_splits(root: Path, seed: int = 42):
    """Assign linked HWR200 documents to stable 80/10/10 splits.

    Reuse documents that cite the same original are kept together. Documents
    with identical normalized full_text are grouped as well. The assignment is
    computed from all source classes, so it stays identical across modalities.
    """
    records = []
    for source in SOURCES:
        for annotation_path in annotation_files(root, source):
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            gt = data.get("full_text")
            if not isinstance(gt, str) or not normalized(gt):
                continue
            refs = []
            if source == "Reuse":
                refs = [
                    value["id"]
                    for name, value in data.items()
                    if name.startswith("reuse_") and isinstance(value, dict)
                    and isinstance(value.get("id"), str) and value["id"].strip()
                ]
            records.append({
                "key": document_key(source, annotation_path, root),
                "source": source,
                "annotation_path": annotation_path,
                "text": normalized(gt).casefold(),
                "reuse_refs": refs,
            })

    if not records:
        raise ValueError("No non-empty HWR200 full_text annotations found for splitting")

    parent = {record["key"]: record["key"] for record in records}

    def find(key):
        parent.setdefault(key, key)
        root_key = key
        while parent[root_key] != root_key:
            root_key = parent[root_key]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root_key
            key = next_key
        return root_key

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    originals_by_name = defaultdict(list)
    for record in records:
        if record["source"] == "Originals":
            path = record["annotation_path"]
            originals_by_name[path.name.casefold()].append(record["key"])
            originals_by_name[path.stem.casefold()].append(record["key"])

    for record in records:
        key = record["key"]
        text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        union(key, f"text:{text_hash}")

        for reference in record["reuse_refs"]:
            ref_path = Path(reference)
            aliases = {ref_path.name.casefold(), ref_path.stem.casefold()}
            matches = sorted({
                original_key
                for alias in aliases
                for original_key in originals_by_name.get(alias, [])
            })
            if matches:
                for original_key in matches:
                    union(key, original_key)
            else:
                # Preserve links even if the referenced original is absent.
                union(key, f"missing-original:{ref_path.name.casefold()}")

    groups = defaultdict(list)
    for record in records:
        groups[find(record["key"])].append(record)
    if len(groups) < 3:
        raise ValueError(
            f"Cannot create train/val/test splits from only {len(groups)} independent document groups"
        )

    rng = random.Random(seed)
    grouped_records = list(groups.values())
    rng.shuffle(grouped_records)
    grouped_records.sort(key=len, reverse=True)

    target_total = {name: len(records) * ratio for name, ratio in SPLIT_RATIOS.items()}
    class_totals = Counter(record["source"] for record in records)
    target_class = {
        split: {source: class_totals[source] * ratio for source in SOURCES}
        for split, ratio in SPLIT_RATIOS.items()
    }
    split_doc_counts = Counter()
    split_class_counts = {split: Counter() for split in SPLIT_RATIOS}
    assignment = {}
    split_names = list(SPLIT_RATIOS)

    for group in grouped_records:
        group_classes = Counter(record["source"] for record in group)

        def projected_load(split):
            overall = (split_doc_counts[split] + len(group)) / target_total[split]
            class_loads = [
                (split_class_counts[split][source] + group_classes[source]) / target_class[split][source]
                for source in SOURCES if target_class[split][source] > 0
            ]
            return 0.5 * overall + 0.5 * (sum(class_loads) / len(class_loads))

        scores = {split: projected_load(split) for split in split_names}
        best = min(scores.values())
        candidates = [split for split in split_names if abs(scores[split] - best) < 1e-12]
        selected = rng.choice(candidates)
        split_doc_counts[selected] += len(group)
        split_class_counts[selected].update(group_classes)
        for record in group:
            assignment[record["key"]] = selected

    stats = {
        "documents": len(records),
        "groups": len(groups),
        "split_documents": dict(split_doc_counts),
        "split_classes": {split: dict(counts) for split, counts in split_class_counts.items()},
    }
    return assignment, stats


def write_manifest(output: Path, rows: list[dict]):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, output)


def split_output_path(output: Path, split: str) -> Path:
    suffix = output.suffix or ".jsonl"
    return output.with_name(f"{output.stem}_{split}{suffix}")


def convert(root: Path, modality_key: str, output: Path, source_filter: str = "all",
            split_manifests: bool = False, split_seed: int = 42):
    modality = MODALITIES[modality_key]
    sources = SOURCES if source_filter == "all" else (source_filter,)
    rows = []
    skipped = {"missing_image_dir": 0, "missing_modality": 0, "missing_full_text": 0, "empty_full_text": 0}
    sidecar = {"matched": 0, "different": 0, "missing": 0}
    split_assignments = {}
    split_stats = None
    if split_manifests:
        split_assignments, split_stats = build_document_splits(root, split_seed)

    for source in sources:
        for annotation_path in annotation_files(root, source):
            source_id = annotation_path.parent.name
            item_name = annotation_path.stem
            image_dir = root / "hw_dataset" / source_id / item_name
            if not image_dir.is_dir():
                skipped["missing_image_dir"] += 1
                continue
            variant_dir = image_dir / modality
            if not variant_dir.is_dir():
                skipped["missing_modality"] += 1
                continue
            images = sorted(
                (p for p in variant_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")),
                key=natural_key,
            )
            if not images:
                skipped["missing_modality"] += 1
                continue

            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            gt = data.get("full_text")
            if not isinstance(gt, str):
                skipped["missing_full_text"] += 1
                continue
            if not normalized(gt):
                skipped["empty_full_text"] += 1
                continue

            txt_path = root / "hw_dataset" / source_id / f"{item_name}.txt"
            if txt_path.is_file():
                if normalized(txt_path.read_text(encoding="utf-8-sig")) == normalized(gt):
                    sidecar["matched"] += 1
                else:
                    sidecar["different"] += 1
            else:
                sidecar["missing"] += 1

            task_id = f"hwr200-{source.lower()}-{source_id}-{item_name}-{modality_key}"
            image_paths = [str(p.resolve()) for p in images]
            row = {
                "task_id": task_id,
                "image": image_paths[0],
                "images": image_paths,
                "image_sha256s": [sha256(p) for p in images],
                "gt": gt,
                "source_dataset": "HWR200",
                "source_class": source,
                "image_variant": modality_key,
                "page_count": len(images),
            }
            if split_manifests:
                row["split"] = split_assignments[document_key(source, annotation_path, root)]
            rows.append(row)

    rows.sort(key=lambda row: row["task_id"])
    if not rows:
        raise ValueError(f"No annotated HWR200 documents found for modality {modality!r}")
    if split_manifests:
        print(f"Split seed: {split_seed}; grouping: linked reuse/original docs and identical full_text")
        print(f"Annotation split totals: {json.dumps(split_stats, ensure_ascii=False, sort_keys=True)}")
        for split in SPLIT_RATIOS:
            split_rows = [row for row in rows if row["split"] == split]
            split_output = split_output_path(output, split)
            write_manifest(split_output, split_rows)
            split_class_counts = Counter(row["source_class"] for row in split_rows)
            split_pages = sum(row["page_count"] for row in split_rows)
            print(
                f"{split}: documents={len(split_rows)}; pages={split_pages}; "
                f"source_classes={dict(split_class_counts)}; manifest={split_output.resolve()}"
            )
    else:
        write_manifest(output, rows)
    print(f"Source classes: {', '.join(sources)}; modality: {modality}")
    print(f"Matched documents: {len(rows)}; pages: {sum(row['page_count'] for row in rows)}; skipped: {skipped}")
    print(f"Sidecar full_text check: {sidecar}")
    if not split_manifests:
        print(f"Manifest: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="Root of the full HWR200 copy (contains annotations/ and hw_dataset/)")
    parser.add_argument("--modality", choices=sorted(MODALITIES), required=True)
    parser.add_argument("--source", choices=["all", *SOURCES], default="all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifests", action="store_true",
                        help="Write separate deterministic 80/10/10 train, val, and test manifests")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for group-level split assignment (default: 42)")
    args = parser.parse_args()
    convert(Path(args.root).expanduser().resolve(), args.modality,
            Path(args.output).expanduser().resolve(), args.source,
            args.split_manifests, args.split_seed)


if __name__ == "__main__":
    main()
