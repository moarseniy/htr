#!/usr/bin/env python3
"""End-to-end full-page OCR benchmark of Label Studio documents.

Backends: OpenAI-compatible vLLM (Qwen), Chandra OCR 2 vLLM.
No line/word segmentation, no manual image enhancement or downsampling.

Commands:
  prepare  : Label Studio annotations.json -> immutable dev/test manifest
  run      : OCR -> append-only result file; interrupted runs can be resumed
  compare  : compare separate model runs on overlapping tasks

Requires: pip install pillow rapidfuzz beautifulsoup4 openai chandra-ocr
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import mimetypes
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    """Equalize line breaks and spaces but do NOT change spelling/case/punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def score_page(gt: str, prediction: str) -> dict:
    """Exact edit-distance CER/WER, including insertions and deletions."""
    from rapidfuzz.distance import Levenshtein

    a, b = normalize(gt), normalize(prediction)
    if not a:
        raise ValueError("Empty normalized ground truth")
    a_words, b_words = a.split(), b.split()
    return {
        "cer": Levenshtein.distance(a, b) / len(a),
        "wer": Levenshtein.distance(a_words, b_words) / len(a_words),
        "char_edits": Levenshtein.distance(a, b),
        "char_count": len(a),
        "word_edits": Levenshtein.distance(a_words, b_words),
        "word_count": len(a_words),
    }


def get_gt(task: dict) -> str | None:
    """Last non-cancelled human annotation, never model predictions."""
    anns = [a for a in task.get("annotations", [])
            if isinstance(a, dict) and not a.get("was_cancelled") and a.get("result")]
    if not anns:
        return None
    if len(anns) > 1:
        print(f"[WARN] task={task.get('id')} has {len(anns)} annotations; selecting latest", file=sys.stderr)
    # Prefer timestamp over arbitrary export order; preserve last list item on ties.
    ann = max(enumerate(anns), key=lambda p: (
        str(p[1].get("updated_at") or p[1].get("created_at") or ""), p[0]
    ))[1]

    segments = []
    for idx, r in enumerate(ann["result"]):
        if not isinstance(r, dict):
            continue
        value = r.get("value") or {}
        texts = value.get("text")
        if not isinstance(texts, list):
            continue
        text = "\n".join(t for t in texts if isinstance(t, str)).strip()
        if not text:
            continue
        y, x = value.get("y"), value.get("x")
        # If coordinates are absent, retain the original annotation result order.
        if isinstance(y, (int, float)) and isinstance(x, (int, float)):
            order = (0, float(y), float(x), idx)
        elif isinstance(y, (int, float)):
            order = (0, float(y), 0.0, idx)
        else:
            order = (1, idx, 0.0, idx)
        segments.append((order, text))
    segments.sort(key=lambda v: v[0])
    gt = "\n".join(text for _, text in segments).strip()
    return gt or None


def field(task: dict, dotted_path: str):
    cur = task
    for part in dotted_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def build_image_index(root: Path) -> dict[str, list[Path]]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
    index = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            index[p.name].append(p.resolve())
    return index


def resolve_image(ref: str, root: Path, index: dict[str, list[Path]]) -> Path:
    """Support relative paths and Label Studio /data/upload/... references."""
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("data.image must be a nonempty path")
    if ref.startswith(("http://", "https://")):
        parsed = urlparse(ref)
        # We only resolve local copies, NEVER download documents from remote hosts.
        ref = parsed.path
    ref = unquote(ref.split("?", 1)[0]).replace("\\", "/")
    p = Path(ref)
    candidates = [p] if p.is_absolute() else []
    candidates.extend([root / ref.lstrip("/"), root / "images" / p.name, root / p.name])
    for c in candidates:
        if c.is_file():
            return c.resolve()
    matches = index.get(p.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image '{ref}': {len(matches)} files named '{p.name}'")
    raise FileNotFoundError(f"Image '{ref}' not found under '{root}'")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {path}:{n}: {e}") from e
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(stable_json(row) + "\n")
    os.replace(temp, path)


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def prepare(args):
    root = Path(args.dataset).expanduser().resolve()
    source = root / "annotations.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    tasks = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("annotations.json must be a JSON list of Label Studio tasks")
    index = build_image_index(root)
    rows = []
    skipped = Counter()
    used_ids = set()
    for task in tasks:
        task_id = task.get("id")
        if task_id is None:
            skipped["missing_id"] += 1
            continue
        tid = str(task_id)
        if tid in used_ids:
            raise ValueError(f"Duplicate task id: {tid}")
        used_ids.add(tid)
        gt = get_gt(task)
        if not gt or not normalize(gt):
            skipped["missing_gt"] += 1
            continue
        try:
            img = resolve_image(field(task, "data.image"), root, index)
        except (ValueError, FileNotFoundError) as e:
            print(f"[SKIP] task={tid}: {e}", file=sys.stderr)
            skipped["image_missing_or_ambiguous"] += 1
            continue
        with img.open("rb") as f:
            from PIL import Image
            with Image.open(f) as image:
                image.verify()
        with img.open("rb") as f:
            from PIL import Image
            with Image.open(f) as image:
                width, height = image.size
        sha = digest_file(img)
        group = field(task, args.group_field) if args.group_field else None
        # When no document ID is provided, exact duplicate images always stay together.
        group_id = str(group) if group is not None and str(group).strip() else f"sha256:{sha}"
        rows.append({
            "task_id": tid,
            "image": str(img),
            "gt": gt,
            "image_sha256": sha,
            "group_id": group_id,
            "width": width,
            "height": height,
        })

    if len(rows) < 2:
        raise ValueError("Fewer than 2 valid labeled images found")
    # Connect all records sharing a group OR the same image, even across group IDs.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    by_group, by_sha = {}, {}
    for i, r in enumerate(rows):
        for cache, key in ((by_group, r["group_id"]), (by_sha, r["image_sha256"])):
            if key in cache:
                union(i, cache[key])
            else:
                cache[key] = i
    # Identical bytes must not have contradictory GT.
    gt_by_sha = defaultdict(set)
    for r in rows:
        gt_by_sha[r["image_sha256"]].add(normalize(r["gt"]))
    conflicts = sum(len(texts) > 1 for texts in gt_by_sha.values())
    if conflicts:
        raise ValueError(f"{conflicts} duplicate image hashes have conflicting GT texts; please review annotations")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(i)
    keys = list(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    dev_keys = set()
    dev_count = 0
    for key in keys:
        size = len(groups[key])
        # Prevent ending with empty test split.
        if dev_count < args.dev_size and len(dev_keys) < len(keys) - 1:
            dev_keys.add(key)
            dev_count += size
    if not dev_keys:
        raise ValueError("Could not produce dev split; check group field")
    for key, indices in groups.items():
        split = "dev" if key in dev_keys else "test"
        for i in indices:
            rows[i]["split"] = split
    rows.sort(key=lambda row: (row["split"], row["task_id"]))
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite explicitly")
    write_jsonl_atomic(output, rows)
    print(f"Tasks in source: {len(tasks)}; valid GT/images: {len(rows)}; skipped: {dict(skipped)}")
    print(f"Manifest: {output}")
    print(f"Dev: {sum(r['split'] == 'dev' for r in rows)}, test: {sum(r['split'] == 'test' for r in rows)}")
    print("IMPORTANT: For multi-page documents, specify --group-field data.document_id if available.")


def image_as_data_url(path: Path) -> str:
    from PIL import Image
    mime, _ = mimetypes.guess_type(path.name)
    if mime in {"image/jpeg", "image/png", "image/webp"}:
        blob = path.read_bytes()
    else:
        # Container format conversion only, NO resize, crop, denoise or enhancement.
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "PNG")
            blob = buffer.getvalue()
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"


PROMPT = (
    "Выполни точную транскрипцию ВСЕГО текста на изображении документа "
    "(включая рукописный и печатный). Прочитай все строки в естественном порядке. "
    "Не исправляй орфографию, не угадывай отсутствующие слова, не пересказывай, "
    "не добавляй комментарии или Markdown. Сохраняй цифры и пунктуацию. "
    "Верни только распознанный текст."
)


class OpenAIBackend:
    def __init__(self, args):
        from openai import OpenAI
        self.args = args
        self.client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                             timeout=args.timeout, max_retries=0)

    def infer(self, path: Path) -> dict:
        kwargs = {}
        if not self.args.allow_thinking and "qwen" in self.args.model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        reply = self.client.chat.completions.create(
            model=self.args.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_as_data_url(path)}},
            ]}],
            temperature=0,
            max_tokens=self.args.max_tokens,
            **kwargs,
        )
        choice = reply.choices[0]
        text = choice.message.content or ""
        if isinstance(text, list):
            text = "\n".join(str(x.get("text", "")) for x in text if isinstance(x, dict))
        return {
            "prediction": str(text),
            "raw": str(text),
            "finish_reason": choice.finish_reason,
            "tokens": getattr(getattr(reply, "usage", None), "total_tokens", None),
        }

    def infer_many(self, paths: list[Path], max_workers: int) -> list[dict]:
        """Send requests concurrently so vLLM can continuously batch them."""
        def safe_infer(path: Path) -> dict:
            try:
                return self.infer(path)
            except Exception as exc:
                return {"_error": f"{type(exc).__name__}: {exc}"}

        if len(paths) == 1:
            return [safe_infer(paths[0])]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(paths))) as pool:
            return list(pool.map(safe_infer, paths))



def chandra_html_to_text(html: str) -> str:
    """Extract text without introducing artificial spaces around inline HTML tags.

    For example, ``<b>При</b>вет`` must stay ``Привет``; table cells and
    paragraphs must remain separated. HTML/layout markup is not part of GT.
    """
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "svg", "img"]):
        tag.decompose()
    block_tags = {
        "address", "article", "aside", "blockquote", "caption", "dd", "div", "dl",
        "dt", "figcaption", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "li", "main", "ol", "p", "pre", "section", "table", "td", "th",
        "tr", "ul",
    }
    for tag in list(soup.find_all(True)):
        if tag.parent is None:
            continue
        if tag.name == "br":
            tag.replace_with(NavigableString("\n"))
        elif tag.name in block_tags:
            tag.insert_before(NavigableString(" "))
            tag.insert_after(NavigableString(" "))
    return normalize(soup.get_text(separator=""))


class ChandraBackend:
    def __init__(self, args):
        # Set before import: pydantic settings may read environment at module import.
        os.environ["VLLM_API_BASE"] = args.base_url
        os.environ["VLLM_API_KEY"] = args.api_key
        os.environ["VLLM_MODEL_NAME"] = args.model
        from openai import OpenAI
        from chandra.model.schema import BatchInputItem
        from chandra.model.util import detect_repeat_token, scale_to_fit
        from chandra.model.vllm import image_to_base64
        from chandra.output import parse_html, parse_markdown
        from chandra.prompts import PROMPT_MAPPING
        self.BatchInputItem = BatchInputItem
        self.detect_repeat_token = detect_repeat_token
        self.scale_to_fit = scale_to_fit
        self.image_to_base64 = image_to_base64
        self.parse_html = parse_html
        self.parse_markdown = parse_markdown
        self.prompt_mapping = PROMPT_MAPPING
        self.client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                             timeout=args.timeout, max_retries=0)
        self.repetition_detection_enabled = not args.disable_repetition_detection
        self.repetition_detection_lock = Lock()
        self.args = args

    def _request(self, item, temperature: float, top_p: float):
        prompt_type = item.prompt_type or "ocr_layout"
        prompt = item.prompt or self.prompt_mapping[prompt_type]
        image = self.scale_to_fit(item.image)
        image_b64 = self.image_to_base64(image)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}]

        extra_body = {
            "repetition_penalty": self.args.repetition_penalty,
            "frequency_penalty": self.args.frequency_penalty,
            "presence_penalty": self.args.presence_penalty,
        }
        if self.repetition_detection_enabled:
            extra_body["repetition_detection"] = {
                "max_pattern_size": self.args.repetition_max_pattern_size,
                "min_pattern_size": self.args.repetition_min_pattern_size,
                "min_count": self.args.repetition_min_count,
            }

        def create(body):
            return self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                max_tokens=self.args.max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=body,
            )

        try:
            return create(extra_body)
        except Exception as exc:
            # Older/custom vLLM servers may not expose the n-gram detector.
            # Fall back to repetition penalties and remember that for this run.
            if self.repetition_detection_enabled and "repetition_detection" in str(exc).lower():
                with self.repetition_detection_lock:
                    self.repetition_detection_enabled = False
                extra_body.pop("repetition_detection", None)
                return create(extra_body)
            raise

    def _to_record(self, completion) -> dict:
        choice = completion.choices[0]
        raw = choice.message.content or ""
        if isinstance(raw, list):
            raw = "\n".join(str(x.get("text", "")) for x in raw if isinstance(x, dict))
        raw = str(raw)
        html = self.parse_html(raw, include_headers_footers=True, include_images=True)
        markdown = self.parse_markdown(raw, include_headers_footers=True, include_images=True)
        prediction = chandra_html_to_text(html or "")
        finish_reason = choice.finish_reason
        tokens = getattr(getattr(completion, "usage", None), "completion_tokens", None)
        repeated = (
            finish_reason == "repetition"
            or self.detect_repeat_token(raw)
            or (len(raw) > 50 and self.detect_repeat_token(raw, cut_from_end=50))
        )
        return {
            "prediction": prediction,
            "raw": raw,
            "html": html or "",
            "markdown": markdown or "",
            "finish_reason": finish_reason,
            "tokens": tokens,
            "loop_detected": bool(repeated),
            "token_limit_reached": finish_reason == "length" or (
                isinstance(tokens, (int, float)) and tokens >= self.args.max_tokens
            ),
            "repetition_detection_enabled": self.repetition_detection_enabled,
        }

    def _generate_one(self, item, repeat_retries: int) -> dict:
        last_record = None
        initial_loop_detected = False
        for attempt in range(repeat_retries + 1):
            temperature = 0.0 if attempt == 0 else min(0.2 * attempt, 0.8)
            top_p = 0.1 if attempt == 0 else 0.95
            completion = None
            for failure_attempt in range(self.args.max_failure_retries + 1):
                try:
                    completion = self._request(item, temperature, top_p)
                    break
                except Exception as exc:
                    if failure_attempt >= self.args.max_failure_retries:
                        return {"_error": f"{type(exc).__name__}: {exc}"}
                    time.sleep(2 * (failure_attempt + 1))
            try:
                last_record = self._to_record(completion)
            except Exception as exc:
                return {"_error": f"{type(exc).__name__}: {exc}"}
            if attempt == 0:
                initial_loop_detected = last_record["loop_detected"]
            last_record["initial_loop_detected"] = initial_loop_detected
            last_record["repeat_retries_used"] = attempt
            last_record["loop_recovered"] = initial_loop_detected and not last_record["loop_detected"]
            if not last_record["loop_detected"] or attempt >= repeat_retries:
                return last_record
        return last_record or {"_error": "Chandra returned no completion"}

    def _generate_items(self, items, max_workers: int, repeat_retries: int) -> list[dict]:
        if len(items) == 1:
            return [self._generate_one(items[0], repeat_retries)]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
            return list(pool.map(lambda item: self._generate_one(item, repeat_retries), items))

    def infer_many(self, paths: list[Path], max_workers: int) -> list[dict]:
        """Send a whole batch concurrently through vLLM's OpenAI-compatible API."""
        from PIL import Image
        output: list[dict | None] = [None] * len(paths)
        items = []
        positions = []
        for position, path in enumerate(paths):
            try:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                items.append(self.BatchInputItem(image=image, prompt_type=self.args.chandra_prompt_type))
                positions.append(position)
            except Exception as exc:
                output[position] = {"_error": f"{type(exc).__name__}: {exc}"}

        if not items:
            return [r or {"_error": "Image preparation produced no batch items"} for r in output]

        try:
            results = self._generate_items(items, max_workers, self.args.chandra_repeat_retries)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            for position in positions:
                output[position] = {"_error": message}
            return [r or {"_error": message} for r in output]

        item_by_position = dict(zip(positions, items))
        for position, result in zip(positions, results):
            try:
                output[position] = result
            except Exception as exc:
                output[position] = {"_error": f"{type(exc).__name__}: {exc}"}

        # If the layout-oriented response still loops or hits its ceiling, try the
        # model's plain OCR prompt once as a bounded fallback. Keep both attempts.
        fallback_positions = [
            position for position in positions
            if output[position]
            and not output[position].get("_error")
            and (output[position].get("loop_detected") or output[position].get("token_limit_reached"))
        ]
        if self.args.chandra_prompt_type != "ocr" and fallback_positions:
            fallback_items = [
                self.BatchInputItem(image=item_by_position[position].image, prompt_type="ocr")
                for position in fallback_positions
            ]
            try:
                fallback_results = self._generate_items(fallback_items, max_workers, repeat_retries=0)
                for position, fallback_result in zip(fallback_positions, fallback_results):
                    first = output[position]
                    try:
                        fallback = fallback_result
                        if fallback.get("_error"):
                            raise RuntimeError(fallback["_error"])
                        fallback.update({
                            "initial_prediction": first.get("prediction", ""),
                            "initial_raw": first.get("raw", ""),
                            "initial_tokens": first.get("tokens"),
                            "initial_loop_detected": first.get("initial_loop_detected",
                                                                first.get("loop_detected", False)),
                            "initial_token_limit_reached": first.get("token_limit_reached", False),
                            "final_loop_detected": fallback.get("loop_detected", False),
                            "final_token_limit_reached": fallback.get("token_limit_reached", False),
                            "loop_detected": first.get("initial_loop_detected", first.get("loop_detected", False))
                                             or fallback.get("loop_detected", False),
                            "token_limit_reached": (first.get("token_limit_reached", False)
                                                     or fallback.get("token_limit_reached", False)),
                            "loop_fallback_used": True,
                            "loop_fallback_prompt_type": "ocr",
                        })
                        output[position] = fallback
                    except Exception as exc:
                        # Keep and score the original model response if fallback itself fails.
                        first["loop_fallback_used"] = True
                        first["loop_fallback_prompt_type"] = "ocr"
                        first["loop_fallback_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                for position in fallback_positions:
                    output[position]["loop_fallback_used"] = True
                    output[position]["loop_fallback_prompt_type"] = "ocr"
                    output[position]["loop_fallback_error"] = f"{type(exc).__name__}: {exc}"
        return [r or {"_error": "Chandra returned no result for this image"} for r in output]


def summarize_successes(rows: list[dict], expected_pages: int) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r for r in rows if r.get("status") != "ok"]
    scored = [r for r in rows if "char_count" in r and "word_count" in r]
    chars = sum(r["char_count"] for r in scored)
    words = sum(r["word_count"] for r in scored)
    # Batched rows repeat the batch wall time; count it once per batch.
    batch_seconds = {}
    legacy_seconds = 0.0
    for row in rows:
        batch_id = row.get("batch_id")
        if batch_id is not None:
            batch_seconds[str(batch_id)] = float(row.get("batch_seconds", 0))
        elif row.get("status") == "ok":
            legacy_seconds += float(row.get("seconds", 0))
    seconds = sum(batch_seconds.values()) + legacy_seconds
    initial_loops = sum(bool(r.get("initial_loop_detected", r.get("loop_detected", False))) for r in scored)
    final_loops = sum(bool(r.get("final_loop_detected", r.get("loop_detected", False))) for r in scored)
    fallback_pages = sum(bool(r.get("loop_fallback_used")) for r in scored)
    token_limited = sum(bool(r.get("token_limit_reached")) for r in scored)
    return {
        "expected_pages": expected_pages,
        "pages_ok": len(ok),
        "pages_failed": len(errors),
        "pages_scored": len(scored),
        "pages_pending": max(0, expected_pages - len(scored)),
        "pages_with_initial_loop": initial_loops,
        "pages_with_final_loop": final_loops,
        "pages_using_ocr_fallback": fallback_pages,
        "pages_at_token_limit": token_limited,
        "cer": sum(r["char_edits"] for r in scored) / chars if chars else None,
        "wer": sum(r["word_edits"] for r in scored) / words if words else None,
        "mean_page_cer": sum(r["cer"] for r in scored) / len(scored) if scored else None,
        "mean_page_wer": sum(r["wer"] for r in scored) / len(scored) if scored else None,
        "total_seconds": round(seconds, 2),
        "mean_seconds_per_page": round(seconds / len(scored), 2) if scored else None,
        "pages_per_second": round(len(scored) / seconds, 4) if seconds else None,
    }


def run(args):
    manifest = Path(args.manifest).expanduser().resolve()
    data = read_jsonl(manifest)
    selected = [r for r in data if args.split == "all" or r["split"] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError(f"No examples in split '{args.split}'")

    if args.model is None:
        args.model = "chandra" if args.backend == "chandra-vllm" else "Qwen/Qwen3.5-27B"
    run_dir = Path(args.output).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    config = {
        "schema_version": 3,
        "manifest_sha256": digest_file(manifest),
        "split": args.split,
        "limit": args.limit,
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "chandra_prompt_type": args.chandra_prompt_type if args.backend == "chandra-vllm" else None,
        "chandra_repeat_retries": args.chandra_repeat_retries if args.backend == "chandra-vllm" else None,
        "chandra_fallback_repeat_retries": 0 if args.backend == "chandra-vllm" else None,
        "chandra_failure_retries": args.max_failure_retries if args.backend == "chandra-vllm" else None,
        "chandra_repetition_penalty": args.repetition_penalty if args.backend == "chandra-vllm" else None,
        "chandra_frequency_penalty": args.frequency_penalty if args.backend == "chandra-vllm" else None,
        "chandra_presence_penalty": args.presence_penalty if args.backend == "chandra-vllm" else None,
        "chandra_repetition_detection": ({
            "max_pattern_size": args.repetition_max_pattern_size,
            "min_pattern_size": args.repetition_min_pattern_size,
            "min_count": args.repetition_min_count,
        } if args.backend == "chandra-vllm" and not args.disable_repetition_detection else None),
        "chandra_repetition_detection_requested": (not args.disable_repetition_detection
                                                     if args.backend == "chandra-vllm" else None),
        "chandra_loop_policy": "penalty, n-gram early stop, bounded retries, OCR fallback; score final output" if args.backend == "chandra-vllm" else None,
        "allow_thinking": args.allow_thinking if args.backend == "openai" else None,
        "prompt": PROMPT if args.backend == "openai" else "chandra SDK built-in prompt",
        "timing": "batch wall time; seconds per page is amortized throughput time",
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError(f"Run directory {run_dir} already belongs to a different configuration. Use another --output")
    else:
        write_json_atomic(config_path, config)

    result_path = run_dir / "results.jsonl"
    previous = read_jsonl(result_path) if result_path.exists() else []
    completed = {
        str(r["task_id"]) for r in previous
        if r.get("status") == "ok" or r.get("retryable") is False
    }
    selected_ids = {str(r["task_id"]) for r in selected}
    if any(r.get("status") == "ok" and str(r["task_id"]) not in selected_ids for r in previous):
        raise ValueError("Existing run contains completed tasks outside the selected manifest subset")
    if len(completed) == len(selected):
        print(f"All {len(selected)} pages are already complete or terminal; refreshing summary")
    else:
        backend = OpenAIBackend(args) if args.backend == "openai" else ChandraBackend(args)
        pending = [example for example in selected if str(example["task_id"]) not in completed]
        batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
        print(f"Backend={args.backend}; pages={len(selected)}; completed={len(completed)}; "
              f"batch_size={args.batch_size}; output={run_dir}")
        # Each batch is appended after it finishes. Failed pages are retried on resume.
        with result_path.open("a", encoding="utf-8") as f:
            for batch_number, batch in enumerate(batches, 1):
                batch_id = f"{time.time_ns()}-{batch_number}"
                started = time.perf_counter()
                valid_indices, paths = [], []
                errors = {}
                for index, example in enumerate(batch):
                    path = Path(example["image"])
                    try:
                        if not path.is_file():
                            raise FileNotFoundError(path)
                        if args.check_images and digest_file(path) != example["image_sha256"]:
                            raise RuntimeError("Image bytes changed after manifest generation")
                        valid_indices.append(index)
                        paths.append(path)
                    except Exception as exc:
                        errors[index] = f"{type(exc).__name__}: {exc}"

                predictions = []
                if paths:
                    try:
                        predictions = backend.infer_many(paths, max_workers=args.batch_size)
                    except KeyboardInterrupt:
                        print("\nInterrupted during a batch. Resume with the same command.", file=sys.stderr)
                        break
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        predictions = [{"_error": message} for _ in paths]
                elapsed = time.perf_counter() - started
                per_page_seconds = elapsed / len(batch)
                prediction_by_index = dict(zip(valid_indices, predictions))
                records = []

                for index, example in enumerate(batch):
                    path = Path(example["image"])
                    outcome = {
                        "task_id": str(example["task_id"]),
                        "split": example["split"],
                        "image": str(path),
                        "gt": example["gt"],
                        "timestamp": utc_now(),
                        "batch_id": batch_id,
                        "batch_size": len(batch),
                        "batch_seconds": round(elapsed, 3),
                        "seconds": round(per_page_seconds, 3),
                    }
                    try:
                        if index in errors:
                            raise RuntimeError(errors[index])
                        pred = prediction_by_index.get(index)
                        if pred is None:
                            raise RuntimeError("Backend returned no prediction for this page")
                        outcome.update({k: v for k, v in pred.items() if k != "_error"})
                        if pred.get("_error"):
                            outcome["inference_error"] = pred["_error"]
                            raise RuntimeError(pred["_error"])
                        if pred.get("finish_reason") == "length":
                            outcome["token_limit_reached"] = True
                        outcome.update(score_page(example["gt"], pred["prediction"]))
                        outcome["status"] = "ok"
                        completed.add(str(example["task_id"]))
                    except Exception as exc:
                        outcome["status"] = "error"
                        outcome["error"] = f"{type(exc).__name__}: {exc}"
                        outcome["retryable"] = True
                        # Keep every labeled page in the metric denominator. If inference
                        # returned no text, score it as an empty transcription (all deletions).
                        if "char_count" not in outcome:
                            outcome["prediction"] = outcome.get("prediction", "")
                            try:
                                outcome.update(score_page(example["gt"], outcome["prediction"]))
                            except Exception as metric_exc:
                                outcome["metric_error"] = f"{type(metric_exc).__name__}: {metric_exc}"
                    records.append(outcome)

                for outcome in records:
                    f.write(stable_json(outcome) + "\n")
                    f.flush()
                    if args.fsync:
                        os.fsync(f.fileno())
                ok_records = [r for r in records if r["status"] == "ok"]
                errors_count = len(records) - len(ok_records)
                print(f"[batch {batch_number}/{len(batches)}] pages={len(batch)} "
                      f"ok={len(ok_records)} errors={errors_count} wall={elapsed:.1f}s "
                      f"throughput={len(batch) / elapsed:.3f} pages/s", flush=True)
                for outcome in records:
                    if outcome["status"] == "ok":
                        flags = []
                        if outcome.get("initial_loop_detected"):
                            flags.append("loop-retried")
                        if outcome.get("loop_fallback_used"):
                            flags.append("OCR-fallback")
                        if outcome.get("loop_detected"):
                            flags.append("loop-remains")
                        if outcome.get("token_limit_reached"):
                            flags.append("token-limit")
                        print(f"  task={outcome['task_id']} CER={outcome['cer']:.2%} "
                              f"WER={outcome['wer']:.2%} "
                              f"{' '.join(flags)}", flush=True)
                    else:
                        print(f"  task={outcome['task_id']} ERROR: {outcome['error']}",
                              file=sys.stderr, flush=True)

    # Latest attempt per task, so a failed attempt followed by success is counted once.
    attempts = {str(r["task_id"]): r for r in read_jsonl(result_path)} if result_path.exists() else {}
    latest = [attempts[k] for k in selected_ids if k in attempts]
    summary = summarize_successes(latest, len(selected))
    summary.update({"config": config, "created_at": utc_now()})
    write_json_atomic(run_dir / "summary.json", summary)
    print(f"OK={summary['pages_ok']}/{summary['expected_pages']}, scored={summary['pages_scored']}, "
          f"errors={summary['pages_failed']}, "
          f"CER={summary['cer']:.2%} WER={summary['wer']:.2%}" if summary["cer"] is not None
          else f"OK=0/{summary['expected_pages']} (no successful predictions)")
    print(f"Results: {run_dir / 'results.jsonl'}")
    print(f"Loops initial/final: {summary['pages_with_initial_loop']}/"
          f"{summary['pages_with_final_loop']}; OCR fallbacks: {summary['pages_using_ocr_fallback']}; "
          f"token limit: {summary['pages_at_token_limit']}")
    if summary["pages_failed"] or summary["pages_pending"]:
        print("WARNING: request failures are scored as empty transcriptions; check pages_pending for unscored pages.",
              file=sys.stderr)


def compare(args):
    runs = {}
    for path_str in args.runs:
        path = Path(path_str).expanduser().resolve()
        cfg = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
        name = path.name
        if name in runs:
            raise ValueError(f"Duplicate run folder name: {name}")
        latest = {str(r["task_id"]): r for r in read_jsonl(path / "results.jsonl")}
        runs[name] = (cfg, latest)
    configs = [c for c, _ in runs.values()]
    if len({c["manifest_sha256"] for c in configs}) != 1:
        raise ValueError("Runs were produced from different manifest versions; cannot compare")
    common = set.intersection(*(set(r) for _, r in runs.values()))
    common = {
        task_id for task_id in common
        if all("char_count" in r[task_id] and "word_count" in r[task_id] for _, r in runs.values())
    }
    if not common:
        raise ValueError("There are no common scored documents across runs")
    print(f"Same-document comparison: {len(common)} scored documents")
    print(f"{'Run':<26} {'CER':>11} {'WER':>11} {'eff sec/page':>14}")
    compare_rows = []
    for name, (_, records) in runs.items():
        sample = [records[k] for k in common]
        summary = summarize_successes(sample, len(common))
        print(f"{name:<26} {summary['cer']:>10.2%} {summary['wer']:>10.2%} "
              f"{summary['mean_seconds_per_page']:>11.2f}")
        for task_id in common:
            r = records[task_id]
            compare_rows.append({"task_id": task_id, "run": name, "image": r["image"],
                                 "cer": r["cer"], "wer": r["wer"], "seconds": r["seconds"],
                                 "status": r["status"],
                                 "loop_detected": r.get("loop_detected", False),
                                 "initial_loop_detected": r.get("initial_loop_detected", False),
                                 "loop_recovered": r.get("loop_recovered", False),
                                 "repeat_retries_used": r.get("repeat_retries_used", 0),
                                 "token_limit_reached": r.get("token_limit_reached", False),
                                 "loop_fallback_used": r.get("loop_fallback_used", False),
                                 "repetition_detection_enabled": r.get("repetition_detection_enabled")})
    if args.csv:
        path = Path(args.csv).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "run", "image", "cer", "wer", "seconds",
                                                   "status", "loop_detected", "initial_loop_detected",
                                                   "loop_recovered", "repeat_retries_used", "token_limit_reached",
                                                   "loop_fallback_used", "repetition_detection_enabled"])
            writer.writeheader()
            writer.writerows(sorted(compare_rows, key=lambda r: (r["task_id"], r["run"])))
        print(f"Per-page comparison: {path}")


def parser_factory():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Convert Label Studio export to reproducible dev/test manifest")
    p.add_argument("-d", "--dataset", required=True, help="Folder with annotations.json and page images")
    p.add_argument("-o", "--output", default="target_manifest.jsonl")
    p.add_argument("--dev-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-field", help="Path to page group in task JSON, e.g. data.document_id")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=prepare)

    p = sub.add_parser("run", help="Run a vLLM backend in resumable batches")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--backend", choices=["openai", "chandra-vllm"], required=True,
                   help="OpenAI-compatible vLLM for Qwen, or Chandra's vLLM client")
    p.add_argument("--model", help="vLLM model name (defaults: Qwen/Qwen3.5-27B or chandra)")
    p.add_argument("--chandra-prompt-type", choices=["ocr_layout", "ocr"], default="ocr_layout")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Pages processed concurrently per batch (default: 8)")
    p.add_argument("--max-failure-retries", type=int, default=2,
                   help="Chandra vLLM retries for transient request failures (default: 2)")
    p.add_argument("--chandra-repeat-retries", type=int, default=1,
                   help="Chandra SDK retries when it detects repeated output (default: 1)")
    p.add_argument("--repetition-penalty", type=float, default=1.05,
                   help="vLLM token repetition penalty for Chandra (1.0 disables it; default: 1.05)")
    p.add_argument("--frequency-penalty", type=float, default=0.0,
                   help="vLLM frequency penalty for Chandra (default: 0)")
    p.add_argument("--presence-penalty", type=float, default=0.0,
                   help="vLLM presence penalty for Chandra (default: 0)")
    p.add_argument("--disable-repetition-detection", action="store_true",
                   help="Disable vLLM's early stop for repeated n-gram patterns")
    p.add_argument("--repetition-min-pattern-size", type=int, default=1)
    p.add_argument("--repetition-max-pattern-size", type=int, default=5)
    p.add_argument("--repetition-min-count", type=int, default=8,
                   help="Stop when an n-gram repeats this many times (default: 8)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--limit", type=int, help="Process only first N examples (smoke test)")
    p.add_argument("--max-tokens", type=int,
                   help="Output ceiling (default: 4096 for Chandra, 12288 for Qwen)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--allow-thinking", action="store_true")
    p.add_argument("--check-images", action="store_true", help="SHA256 verify images against manifest")
    p.add_argument("--fsync", action="store_true", help="Durable write after every page (slower)")
    p.set_defaults(fn=run)

    p = sub.add_parser("compare", help="Compare CER/WER across runs on exactly common successful documents")
    p.add_argument("runs", nargs="+", help="Run output folders")
    p.add_argument("--csv", help="Optional per-page comparison file")
    p.set_defaults(fn=compare)
    return parser


def main():
    parser = parser_factory()
    args = parser.parse_args()
    if args.command == "prepare" and args.dev_size < 1:
        parser.error("--dev-size must be positive")
    if args.command == "run" and args.max_tokens is None:
        args.max_tokens = 4096 if args.backend == "chandra-vllm" else 12288
    if args.command == "run" and (args.max_tokens < 1 or args.batch_size < 1
                                   or args.max_failure_retries < 0 or args.chandra_repeat_retries < 0
                                   or args.repetition_penalty <= 0 or args.repetition_min_count < 2
                                   or args.repetition_min_pattern_size < 1
                                   or args.repetition_max_pattern_size < args.repetition_min_pattern_size
                                   or (args.limit is not None and args.limit < 1)):
        parser.error("--max-tokens, --batch-size and --limit must be positive; retry count cannot be negative")
    args.fn(args)


if __name__ == "__main__":
    main()
