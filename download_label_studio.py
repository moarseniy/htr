#!/usr/bin/env python3
"""Download Label Studio project annotations and linked images."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests


def has_annotation(task):
    return any(
        not ann.get("was_cancelled", False) and bool(ann.get("result"))
        for ann in task.get("annotations", [])
    )


def main(args):
    token = os.getenv("LABEL_STUDIO_API_KEY")
    if not token:
        raise ValueError("Не задан LABEL_STUDIO_API_KEY (export LABEL_STUDIO_API_KEY='...')")

    base_url = args.base_url.rstrip("/")
    output = Path(args.output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    auth = {"Authorization": f"Token {token}"}

    # Export ALL tasks, then apply the selected filter locally.
    api_url = f"{base_url}/api/projects/{args.project_id}/export"
    response = session.get(
        api_url,
        headers=auth,
        params={"exportType": "JSON", "download_all_tasks": "true"},
        timeout=300,
    )
    response.raise_for_status()
    try:
        tasks = response.json()
    except ValueError as exc:
        raise ValueError("API вернул не JSON. Проверь URL, токен и номер проекта") from exc
    if not isinstance(tasks, list):
        raise ValueError(f"Ожидался список задач, получено: {type(tasks).__name__}")

    selected = [t for t in tasks if args.mode == "all" or has_annotation(t)]
    print(f"Всего задач: {len(tasks)}; к скачиванию: {len(selected)}", flush=True)

    result = []
    server_origin = urlparse(base_url).netloc
    for number, task in enumerate(selected, start=1):
        task_id = task["id"]
        original_image = task["data"]["image"]
        if not isinstance(original_image, str) or not original_image:
            raise ValueError(f"Task {task_id}: data.image отсутствует или пуст")

        url = urljoin(base_url + "/", original_image)
        parts = urlparse(url)
        if parts.scheme not in {"http", "https"}:
            raise ValueError(f"Task {task_id}: неподдерживаемый URL изображения: {original_image}")

        # For /data/local-files/?d=foo.png the filename is in the query string.
        remote_path = parse_qs(parts.query).get("d", [parts.path])[0]
        filename = Path(unquote(remote_path)).name
        if not filename or filename in {".", ".."}:
            raise ValueError(f"Task {task_id}: не удалось определить имя файла: {original_image}")
        output_name = f"{task_id}_{filename}"
        image_path = images_dir / output_name

        if not image_path.exists():
            # Never send the internal Label Studio token to external image hosts.
            headers = auth if parts.netloc == server_origin else {}
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "application/json" in content_type:
                    raise ValueError(
                        f"Task {task_id}: вместо изображения сервер вернул {content_type}. "
                        "Проверь доступ к изображениям в Label Studio"
                    )
                temp_path = image_path.with_name(image_path.name + ".part")
                try:
                    with temp_path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    temp_path.replace(image_path)
                finally:
                    temp_path.unlink(missing_ok=True)

        # Preserve the full Label Studio JSON but exclude drafts and predictions.
        task.pop("predictions", None)
        task.pop("drafts", None)
        if args.mode == "annotated":
            task["annotations"] = [
                a for a in task.get("annotations", [])
                if not a.get("was_cancelled", False) and a.get("result")
            ]
        task["data"]["image"] = f"images/{output_name}"
        result.append(task)
        print(f"[{number}/{len(selected)}] {output_name}", flush=True)

    with (output / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Готово: {len(result)} задач, папка: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Label Studio tasks and download images")
    parser.add_argument("--project-id", type=int, required=True, help="ID проекта Label Studio")
    parser.add_argument("-o", "--output", required=True, help="Папка для датасета")
    parser.add_argument("--mode", choices=["all", "annotated"], default="annotated")
    parser.add_argument(
        "--base-url",
        default="https://mlops-service-02.corp.ingos.ru",
        help="URL сервера Label Studio",
    )
    args = parser.parse_args()
    try:
        main(args)
    except (ValueError, KeyError, requests.RequestException, OSError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
