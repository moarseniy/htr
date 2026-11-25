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
  --max-tokens 12288 \
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

For Chandra, each batch is submitted through `InferenceManager.generate`; the SDK issues concurrent vLLM requests for the items in that batch. For the OpenAI-compatible backend, the client submits concurrent requests so vLLM can use continuous batching. `--batch-size` controls client-side concurrency. Start with a modest value and increase it while monitoring GPU memory and server throughput.

The Chandra client disables repeat-output retries to avoid spending minutes generating stochastic retries for looped pages. Repeated-token outputs and outputs that reach the configured token ceiling are saved as errors with their raw predictions retained for review, and are excluded from CER/WER. Transient vLLM request failures still receive up to two retries.

Results are append-only and resumable. Successful task IDs are skipped on resume. Use a separate output folder for each model/configuration. `run_config.json` records the manifest hash, backend, batch size, token limit, and timing semantics. Batched `seconds` values are amortized per-page throughput time; `batch_seconds` is the measured wall time for that batch.

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

Runs must use the same manifest hash. The comparison uses exactly the successful pages shared by all runs.

## Dependencies

```bash
pip install pillow rapidfuzz beautifulsoup4 openai requests chandra-ocr
```
