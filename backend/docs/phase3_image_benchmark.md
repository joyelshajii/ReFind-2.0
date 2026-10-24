# Phase 3 image-search benchmark

## What the implementation does

The normal image path uses Gemini vision description/transcription when enabled, then the configured text embedder. Local OCR remains the fallback when vision is disabled, unavailable, or returns no text. Because local OCR is skipped if Gemini returns any transcription, exact-text completeness with partial Gemini OCR still needs a real screenshot/diagram evaluation. `IMAGE_VISION_ENABLED=true` keeps vision enabled; setting it to `false` retains local OCR. Jina reranking and the public API are unchanged.

Image decoding now happens once per processor call. Gemini's resized JPEG is reused for identical image/model/config inputs within the process, local OCR receives the already-open image, identical OCR pixels share a bounded cache, and identical text/model embedding requests share a bounded cache. New image records are reused only when their SHA-256 and versioned processing signature both match. Older image records with no signature are refreshed once; unchanged legacy text records keep their existing hash-based skip behavior to avoid rebuilding the full text index. New text records also store their embedding/chunk configuration signature. Search suppresses semantic comparisons when a recorded signature identifies a different configured embedding model/provider, while keeping keyword and metadata retrieval available until reindexing.

Index logs include non-overlapping scan/hash/extraction/embedding/database stage seconds, image throughput, seconds per indexed image, bytes scanned, skipped and failed counts, plus process working-set memory at run start/end and the process high-water working set. Image processing subtimings are reported separately because they overlap. The run-level log is named `Indexing metrics` and stays out of the indexing status/API response.

## Repeatable comparison

From `backend/`, install the normal project requirements and the optional benchmark requirements:

```powershell
pip install -r requirements.txt
pip install -r requirements-image-benchmark.txt
```

Then run the generated labeled fixture set:

```powershell
python scripts/benchmark_image_search.py --output image-search-benchmark.json
```

The script writes illustrative scene drawings, a screenshot with exact text, a labeled energy diagram, and similar mountain-lake scenes with different meanings to a temporary directory. It evaluates top-1 accuracy and MRR for the current description/text-embedding path and direct EmbeddingGemma 2 image-to-text retrieval. It reports index time per image, direct model load time, query-embedding plus in-memory vector-ranking latency, memory, failure counts, and exact OCR-term lookup separately. Search latency excludes query parsing, database access, Jina reranking, and explanations; those stages are shared and unchanged. Exact OCR remains a shared indexed-text path; direct image vectors do not replace OCR.

To use a real local dataset, write a manifest with string IDs, local paths (relative to the manifest or absolute), and one or more queries per expected image:

```json
{
  "images": [
    {"id": "photo_lake", "path": "images/lake.jpg"},
    {"id": "invoice", "path": "images/invoice.png"},
    {"id": "diagram", "path": "images/system-diagram.png"}
  ],
  "queries": [
    {"query": "a canoe on a mountain lake", "expected": "photo_lake"},
    {"query": "invoice number 0427", "expected": "invoice", "ocr_exact": "0427"},
    {"query": "system diagram with MQTT", "expected": "diagram", "ocr_exact": "MQTT"}
  ]
}
```

Run it with `python scripts/benchmark_image_search.py --manifest image-search-manifest.json --output image-search-benchmark.json`. Keep the dataset and labels local; the current Gemini vision and configured embedding providers follow the same settings as ReFind. The direct EmbeddingGemma 2 embeddings run locally.

By default, the script will only load EmbeddingGemma 2 from the local model cache. To permit a model download, pass `--allow-download`. For a useful production recommendation, use a labeled, local corpus that includes genuine photographs, readable screenshots, diagrams, low-text images, and visually similar but semantically different pairs. Keep the same image set, expected target labels, and query set between runs. The built-in generated set is deterministic but is only a smoke/evaluation fixture, not a real-photo relevance benchmark.

## Measurement status at implementation time

No direct EmbeddingGemma 2 relevance, load-time, indexing-time, memory, or latency results are claimed here. The available Python runtime had no PyTorch, Transformers, Sentence Transformers, or ReFind backend dependency installation, and no cached EmbeddingGemma 2 weights. The repository's checked-in local image was a generated architecture screenshot, so the generated fixture procedure above was added rather than treating that single image as representative. Run the script in the configured backend environment to produce actual measurements.

Google's multimodal guide documents an image input path through `SentenceTransformer.encode({"image": ...})`. This is only a documented API capability; no EmbeddingGemma 2 image inference was possible in the verification environment, so image embedding quality and successful local operation remain unverified. See [Google's multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers) and [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2).

## Verification smoke runs — 2026-10-23 and 2026-10-24

Command: `python scripts/benchmark_image_search.py --output .venv/benchmark-smoke.json`, with network-backed Gemini/Jina providers disabled and `IMAGE_VISION_ENABLED=false`. The script generated its seven synthetic images and completed the current local text-embedding/OCR path. These numbers validate execution and metric reporting only; they do not establish real-world relevance or compare Gemini vision against direct image embeddings.

The first baseline-only smoke run processed 7/7 images with 0 failures at 0.809 seconds/image (5.642 seconds total image processing, 0.005 seconds total text embedding). Query-embedding plus in-memory vector-search latency was 0.564 ms mean / 0.655 ms p95. Process RSS was 62.86 MB at start and 213.06 MB at end; process high-water working set was 361.08 MB. Top-1 was 0.4286 and MRR 0.6395 on the synthetic labels. OCR found and matched both exact terms tested (`0427`, `MQTT`) using local fallback; PaddleOCR could not initialize because the PaddlePaddle runtime was absent, and Tesseract was used.

The optional benchmark requirements were then installed in the same venv, including PyTorch 2.14.1, Sentence Transformers 6.2.0, Transformers 5.20.2, and torchvision 0.29.1. An optional-dependency run kept Hugging Face offline and attempted a local-only model load. The model was not in the local cache; loading failed after a 4.465-second attempt with an `OSError` because the files were absent and Hugging Face was unreachable. No direct image inference succeeded: 0 image inferences failed because none were attempted, and all 7 images were reported as not attempted after model setup failed. That run's synthetic baseline took 1.980 seconds/image (13.840 seconds total image processing and 0.005 seconds total text embedding); query-embedding plus vector-search latency was 0.798 ms mean / 2.320 ms p95. Top-1 was 0.4286 and MRR 0.6633. Both exact OCR terms were retrieved. RSS changed from 65.28 MB to 458.46 MB during baseline processing; process high-water working set was 605.68 MB. The baseline ran between 0.809 and 3.719 seconds/image across these smokes, reflecting OCR initialization/cache behavior, so these synthetic timings are environment checks rather than a stable performance comparison.

A final repeat after the partial-status change returned `status: partial`: 7/7 baseline images, 2.023 seconds/image, 14.132 seconds image processing, 0.015 seconds text embedding, 0.801 ms mean / 2.074 ms p95 query/vector latency, top-1 0.4286, MRR 0.6633, both exact OCR terms found, RSS 65.17 MB to 457.86 MB, and 605.08 MB process high-water working set. Direct model setup again failed before inference after a 4.340-second attempt; all 7 images were not attempted.

Direct EmbeddingGemma 2 model-load time and image relevance remain unmeasured; no direct embedding result is claimed. No real-image manifest was available or evaluated.

## Recommendation

Keep Gemini vision enabled by default and make it optional through `IMAGE_VISION_ENABLED`. There is no measured result in this environment that supports replacing it. Local OCR remains enabled for exact text retrieval regardless of that setting. Consider disabling Gemini vision for a deployment only after the benchmark reports acceptable recall on that deployment's real image set and the latency/cost savings are material. The direct EmbeddingGemma 2 path remains a benchmark candidate until its vectors can be compared in an adequately representative set; adding its vectors to production search would require a separately versioned image-vector path and matching text-query embeddings.
