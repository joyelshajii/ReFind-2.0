# Phase 3 image-search benchmark

## What the implementation does

The normal image path remains Gemini vision description plus local OCR plus the configured text embedder. `IMAGE_VISION_ENABLED=true` keeps the existing vision analysis on; setting it to `false` disables that optional stage while retaining local OCR. Jina reranking and the public API are unchanged.

Image decoding now happens once per processor call. Gemini's resized JPEG is reused for identical image/model/config inputs within the process, local OCR receives the already-open image, identical OCR pixels share a bounded cache, and identical text/model embedding requests share a bounded cache. New image records are reused only when their SHA-256 and versioned processing signature both match. Older image records with no signature are refreshed once; unchanged legacy text records keep their existing hash-based skip behavior to avoid rebuilding the full text index. New text records also store their embedding/chunk configuration signature.

Index logs include stage seconds, image throughput, seconds per indexed image, bytes processed, skipped and failed counts, plus process working-set memory at run start/end and the process peak. The run-level log is named `Indexing metrics` and stays out of the indexing status/API response.

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

Google documents EmbeddingGemma 2 as supporting direct image inputs and shared 768-dimensional image/text embeddings through `SentenceTransformer.encode({"image": ...})`; it is therefore technically suitable for a local cross-modal benchmark when those dependencies and weights are available. The model card describes a 440M-parameter text-plus-image configuration and cautions against float16. See [Google's multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers) and [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2).

## Recommendation

Keep Gemini vision enabled by default and make it optional through `IMAGE_VISION_ENABLED`. There is no measured result in this environment that supports replacing it. Local OCR remains enabled for exact text retrieval regardless of that setting. Consider disabling Gemini vision for a deployment only after the benchmark reports acceptable recall on that deployment's real image set and the latency/cost savings are material. The direct EmbeddingGemma 2 path remains a benchmark candidate until its vectors can be compared in an adequately representative set; adding its vectors to production search would require a separately versioned image-vector path and matching text-query embeddings.
