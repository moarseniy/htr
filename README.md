# Full-page Russian HTR benchmark

The benchmark compares full-page OCR predictions with Label Studio ground truth using CER/WER and reports processing throughput. It does not segment pages or alter image resolution.

## Backends

- `chandra-vllm`: Chandra OCR 2 through its OpenAI-compatible vLLM endpoint, with Chandra image and output helpers.
- `openai`: an OpenAI-compatible vLLM endpoint, intended for Qwen.

The vLLM server runs separately from the benchmark. Start the appropriate model server from its local checkpoint first; the benchmark connects to its `/v1` endpoint. The benchmark client does not launch a server or choose its GPU topology.

## Run from a config

Set the three local paths once. Copy the template, edit the paths, then load them into the shell:

```bash
cp configs/local.env.example configs/local.env
# Edit configs/local.env with the manifest and local model directories.
set -a
source configs/local.env
set +a
```

The run configs expand `${HTR_MANIFEST}` and resolve relative output paths from `configs/`. The model path is used only when starting its server; `model` in the run config is the alias served by vLLM.

If the Chandra server is not already running on port 8000, start vLLM directly from the local checkpoint directory:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$CHANDRA_MODEL_DIR" \
  --served-model-name chandra \
  --dtype bfloat16 \
  --max-model-len 18000 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":6291456}' \
  --host 127.0.0.1 \
  --port 8000
```

Run the 8-page pilot, then the full validation set:

```bash
python fullpage_ocr_benchmark.py run --config configs/chandra-pilot.json
python fullpage_ocr_benchmark.py run --config configs/chandra-full-rp105.json
```

Run the baseline with repetition penalty disabled, then compare:

```bash
python fullpage_ocr_benchmark.py run --config configs/chandra-full-rp100.json
python fullpage_ocr_benchmark.py compare \
  runs/chandra-full-rp105 runs/chandra-full-rp100 \
  --csv runs/chandra-repetition-comparison.csv
```

To check whether the n-gram detector is stopping too early on repeated layout tokens, run `configs/chandra-full-rp105-pattern3.json`. It changes only `repetition_min_pattern_size` from 1 to 3 and writes to a new output folder. For a penalty comparison with that detector setting held fixed, run `configs/chandra-full-rp100-pattern3.json` and compare those two output folders.

Start Qwen's local vLLM server on both GPUs and port 8001, then run its config:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$QWEN_MODEL_DIR" \
  --served-model-name Qwen/Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 8001

python fullpage_ocr_benchmark.py run --config configs/qwen-full.json
```

Each JSON file contains the endpoint, batch size, retry policy, repetition controls, and output folder. `HTR_MANIFEST` selects the same validation manifest for each run; the runner evaluates every row in it. Change values in a JSON file when you want a different setup; the model checkpoint path is used by the separate vLLM server, while the `model` field here is the server's served name.

For Chandra, the runner sends concurrent OpenAI-compatible requests to vLLM and uses Chandra's image scaling, prompt mapping, and output parsers. For the OpenAI-compatible backend, the client also submits concurrent requests so vLLM can use continuous batching. `--batch-size` controls client-side concurrency. Start with a modest value and increase it while monitoring GPU memory and server throughput.

Chandra uses a mild vLLM `repetition_penalty` of 1.05 on normal requests. If a loop is detected, only that page's retry and OCR fallback raise the penalty by 0.05, up to 1.2. This avoids applying the stronger penalty to every page. vLLM's n-gram detector is also enabled by default (`max_pattern_size=5`, `min_pattern_size=1`, `min_count=8`). It can stop genuine loops early, but may also stop legitimate repeated form/layout patterns; returned text is still scored. If the server does not support `repetition_detection`, the runner retries without that field and records that it was unavailable. The client makes one repeat-aware retry by default (`--chandra-repeat-retries 1`). If an `ocr_layout` response still repeats or reaches the token ceiling, it tries that page once with the simpler `ocr` prompt. It scores the final response even if it still repeats or is cut off, and stores the initial output when fallback runs. The `rp100` config disables penalty escalation for a clean baseline. The default Chandra ceiling is 4096 tokens; all 46 non-looping responses in the attached 50-page run were below 3000 tokens. Raise the ceiling if legitimate pages need longer output. Transient vLLM request failures receive up to two retries; if inference still fails, the page is scored with an empty transcription so it remains in the metric denominator.

Results are append-only and resumable. Successful task IDs are skipped on resume; transient errors are retried. Looping and token-limited outputs remain scored results. Use a separate output folder for each model/configuration. `run_config.json` records the manifest hash, backend, retry policy, batch size, token limit, and timing semantics. Batched `seconds` values are amortized per-page throughput time; `batch_seconds` is the measured wall time for that batch.

## Prepare a manifest from Label Studio

`prepare` converts the Label Studio export into a reproducible manifest containing every valid task ID, image path, human transcript and image hash. It does not divide the set into dev/test subsets.

```bash
python download_label_studio.py --project-id PROJECT_ID --output data/label-studio-validation --mode annotated
python fullpage_ocr_benchmark.py prepare \
  --dataset data/label-studio-validation \
  --output data/validation_174.jsonl
wc -l data/validation_174.jsonl
```

Check the `prepare` summary and `wc -l`: both should show 174 valid rows. If they show 50, the Label Studio export folder contains only 50 tasks or the rest lack valid images/annotations; use the complete export before preparing again. Then set `HTR_MANIFEST` in `configs/local.env` to the absolute path of `data/validation_174.jsonl`.

## School Notebooks RU

The local COCO export has official `train`, `val`, and `test` files. This converter builds page-level transcripts from text-bearing categories (`pupil_text`, `pupil_comment`, `teacher_comment`), ordering words by annotated line geometry. It creates separate manifests for the official validation and test splits; training pages are not included in evaluation.

```bash
python prepare_school_notebooks.py \
  --root /home/arseniy/data/school_notebooks_RU \
  --split val \
  --output data/open_manifests/school_notebooks_val.jsonl
python prepare_school_notebooks.py \
  --root /home/arseniy/data/school_notebooks_RU \
  --split test \
  --output data/open_manifests/school_notebooks_test.jsonl
```

Run both OCR models on the same official validation split, then compare:

```bash
python fullpage_ocr_benchmark.py run --config configs/chandra-school-val.json
python fullpage_ocr_benchmark.py run --config configs/qwen-school-val.json
python fullpage_ocr_benchmark.py compare \
  runs/school-notebooks-val-chandra-pattern3 runs/school-notebooks-val-qwen \
  --csv runs/school-notebooks-val-comparison.csv
```

After selecting a configuration on validation, use the corresponding `*-test.json` configs for the held-out test split. The raw Qwen test run hit its token ceiling on task `school-notebooks-test-57`; `configs/qwen-school-test-guarded.json` enables vLLM repetition detection and one bounded retry for a follow-up run. Its final output, including any output that still reaches the ceiling, remains scored.

## HWR200

HWR200 JSON annotations contain a transcript for the whole document, while each `hw_dataset/<id>/<item>/...` folder contains its pages. `prepare_hwr200.py` therefore creates document-level manifest rows with an ordered list of page images. It creates separate manifests for scans, light photos, and dark photos instead of counting the same text three times. For model development, create train/val/test manifests by source folder ID, grouped across annotation classes and modalities. The same ID receives the same split in every modality. Reuse-reference links are not grouped transitively because they connect most of the corpus and can leave validation with only one source class.

The HWR200 folder currently available on this machine is only a fragment: it has 300 annotation/image pairs per modality (20 FPR, 20 Originals, 260 Reuse) and 1,091 page images. Its manifests are named `hwr200_subset_*.jsonl`; do not use those for full-dataset results. The converter requires an explicit root path to avoid silently using this fragment.

```bash
python prepare_hwr200.py --root /absolute/path/to/full/HWR200 --modality scans \
  --split-manifests --output data/open_manifests/hwr200_scans.jsonl
python prepare_hwr200.py --root /absolute/path/to/full/HWR200 --modality light \
  --split-manifests --output data/open_manifests/hwr200_light.jsonl
python prepare_hwr200.py --root /absolute/path/to/full/HWR200 --modality dark \
  --split-manifests --output data/open_manifests/hwr200_dark.jsonl
```

Each command writes `_train.jsonl`, `_val.jsonl`, and `_test.jsonl` beside the requested output name. The split is deterministic with seed 42. First compare models on `val`; reserve `test` for a final run after choosing the model and decoding settings. The runner OCRs each document's pages in filename order, joins their outputs, and scores against that document's `full_text`.

```bash
HTR_MANIFEST="$PWD/data/open_manifests/hwr200_scans_val.jsonl" \
HTR_OUTPUT="$PWD/runs/hwr200-scans-val-chandra" \
python fullpage_ocr_benchmark.py run --config configs/chandra-hwr.json

HTR_MANIFEST="$PWD/data/open_manifests/hwr200_scans_val.jsonl" \
HTR_OUTPUT="$PWD/runs/hwr200-scans-val-qwen-guarded" \
python fullpage_ocr_benchmark.py run --config configs/qwen-hwr-guarded.json

python fullpage_ocr_benchmark.py compare \
  runs/hwr200-scans-val-chandra runs/hwr200-scans-val-qwen-guarded \
  --csv runs/hwr200-scans-val-comparison.csv
```

Repeat with `hwr200_light_val.jsonl` and `hwr200_dark_val.jsonl` if you need modality-specific comparisons. When model and decoding settings are frozen, run the chosen setup once on the matching `_test.jsonl`. Use only `_train.jsonl` for training. The repository currently contains `hwr200_subset_*.jsonl` only for pipeline smoke checks.

## Compare runs

```bash
python fullpage_ocr_benchmark.py compare runs/chandra-vllm runs/qwen-vllm
```

Runs must use the same manifest hash. The comparison uses exactly the pages with a score in every run, including pages where a request failed and was scored as an empty transcription.

## Dependencies

```bash
pip install -r requirements.txt
```
