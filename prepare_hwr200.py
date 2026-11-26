#!/usr/bin/env python3
"""Build document-level HWR200 manifests from annotations and page-image folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


MODALITIES = {
    "scans": "Сканы",
    "light": "ФотоСветлое",
    "dark": "ФотоТемное",
}
SOURCES = ("FPR", "Originals", "Reuse")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


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


def convert(root: Path, modality_key: str, output: Path, source_filter: str = "all"):
    modality = MODALITIES[modality_key]
    sources = SOURCES if source_filter == "all" else (source_filter,)
    rows = []
    skipped = {"missing_image_dir": 0, "missing_modality": 0, "missing_full_text": 0, "empty_full_text": 0}
    sidecar = {"matched": 0, "different": 0, "missing": 0}

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
            rows.append({
                "task_id": task_id,
                "image": image_paths[0],
                "images": image_paths,
                "image_sha256s": [sha256(p) for p in images],
                "gt": gt,
                "source_dataset": "HWR200",
                "source_class": source,
                "image_variant": modality_key,
                "page_count": len(images),
            })

    rows.sort(key=lambda row: row["task_id"])
    if not rows:
        raise ValueError(f"No annotated HWR200 documents found for modality {modality!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    print(f"Source classes: {', '.join(sources)}; modality: {modality}")
    print(f"Documents: {len(rows)}; pages: {sum(row['page_count'] for row in rows)}; skipped: {skipped}")
    print(f"Sidecar full_text check: {sidecar}")
    print(f"Manifest: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/arseniy/data/HWR200")
    parser.add_argument("--modality", choices=sorted(MODALITIES), required=True)
    parser.add_argument("--source", choices=["all", *SOURCES], default="all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(Path(args.root).expanduser().resolve(), args.modality,
            Path(args.output).expanduser().resolve(), args.source)


if __name__ == "__main__":
    main()
