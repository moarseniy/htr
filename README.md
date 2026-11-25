# Full-page Russian HTR benchmark

The benchmark compares full-page OCR predictions with Label Studio ground truth using CER/WER and reports processing throughput. It does not segment pages or alter image resolution.

## Backends

- `chandra-vllm`: Chandra OCR 2 through the Chandra SDK's vLLM client.
- `openai`: an OpenAI-compatible vLLM endpoint, intended for Qwen.

The vLLM server runs separately from the benchmark. Start the appropriate model server first, then point `--base-url` at its `/v1` endpoint. For Chandra, the SDK provides `chandra_vllm`; for Qwen, use the OpenAI-compatible vLLM server for that model. The benchmark client does not launch a server or choose its GPU topology.

## Run

Chandra OCR 2, using batches of eight pages by default:

```bash
python fullpage_ocr_benchmark.py run \
  --manifest target_manifest.jsonl \
  --backend chandra-vllm \
  --model chandra \
  --base-url http://localhost:8000/v1 \
  --batch-size 8 \
  --max-tokens 4096 \
  --split dev \
  --output runs/chandra-vllm
```

Qwen through an OpenAI-compatible vLLM endpoint:

```bash
python fullpage_ocr_benchmark.py run \
  --manifest target_manifest.jsonl \
  --backend openai \
  --model Qwen/Qwen3.5-27B \
  --base-url http://localhost:8000/v1 \
  --batch-size 8 \
  --split dev \
  --output runs/qwen-vllm
```

For Chandra, the runner sends concurrent OpenAI-compatible requests to vLLM and uses Chandra's image scaling, prompt mapping, and output parsers. For the OpenAI-compatible backend, the client also submits concurrent requests so vLLM can use continuous batching. `--batch-size` controls client-side concurrency. Start with a modest value and increase it while monitoring GPU memory and server throughput.

Chandra requests use a mild vLLM `repetition_penalty` of 1.05 and enable vLLM's n-gram repetition detector by default (`max_pattern_size=5`, `min_pattern_size=1`, `min_count=8`). The detector can stop a genuine loop before it consumes the whole token budget, but may also stop legitimate repeated form/layout patterns; all returned text is still scored. If the server does not support `repetition_detection`, the runner retries without that field and records that it was unavailable. The client also makes one repeat-aware retry by default (`--chandra-repeat-retries 1`). If an `ocr_layout` response still repeats or reaches the token ceiling, it tries that page once with the simpler `ocr` prompt. It scores the final response even if it still repeats or is cut off, and stores the initial output when fallback runs. Compare against `--repetition-penalty 1.0` to measure the accuracy/loop-rate tradeoff. The default Chandra ceiling is 4096 tokens; all 46 non-looping responses in the attached 50-page run were below 3000 tokens. Raise the ceiling if legitimate pages need longer output. Transient vLLM request failures receive up to two retries; if inference still fails, the page is scored with an empty transcription so it remains in the metric denominator.

Results are append-only and resumable. Successful task IDs are skipped on resume; transient errors are retried. Looping and token-limited outputs remain scored results. Use a separate output folder for each model/configuration. `run_config.json` records the manifest hash, backend, retry policy, batch size, token limit, and timing semantics. Batched `seconds` values are amortized per-page throughput time; `batch_seconds` is the measured wall time for that batch.

## Prepare a manifest from Label Studio

```bash
python download_label_studio.py --project-id PROJECT_ID --output data/label-studio
python fullpage_ocr_benchmark.py prepare \
  --dataset data/label-studio \
  --output target_manifest.jsonl \
  --dev-size 50
```

If tasks contain a document identifier, pass it as `--group-field data.document_id` so pages from one document remain in the same split.

## Compare runs

```bash
python fullpage_ocr_benchmark.py compare runs/chandra-vllm runs/qwen-vllm
```

Runs must use the same manifest hash. The comparison uses exactly the pages with a score in every run, including pages where a request failed and was scored as an empty transcription.

## Dependencies

```bash
pip install pillow rapidfuzz beautifulsoup4 openai requests chandra-ocr
```
