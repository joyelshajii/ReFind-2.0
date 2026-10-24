# EmbeddingGemma performance changes

The default remains the pinned `google/embeddinggemma-2` model for document,
chunk, query and direct image embeddings. Existing compatible vectors are
reused; these changes do not require a model migration or a full reindex.
The local environment explicitly enables `LOCAL_FAST_MODE=true`.

## Runtime behavior

- One model runs on CUDA with bfloat16 when supported. Text batch size remains
  eight for the verified 4 GB RTX 3050 Laptop GPU; CUDA OOM splitting remains.
- Startup warms the local model before accepting searches. Disable
  `LOCAL_EMBEDDING_WARMUP` only if lower startup time is more important than
  first-search latency. Loading weights and importing the model runtime can
  still take several seconds, once per process.
- Text batches group similar lengths to reduce padding. Search queries get
  priority between indexing batches; cached queries avoid the inference queue.
- Image OCR runs on the CPU concurrently with image embedding on the GPU.
  Both must finish before the image closes. No OCR pass or image vector is
  skipped. PDF OCR consumes rendered pixels directly instead of encoding and
  immediately decoding a PNG. Raster resolution remains 150 DPI.
- Text extraction workers are reused across a folder scan. Folder discovery
  runs once per indexing request. Per-file transactions and locks remain.

## Retrieval and accuracy

SQLite startup installs an additive revision table and triggers for document
and chunk writes. A bounded in-memory snapshot stores normalized float32
vectors and searchable text. Each search checks the committed revision; inserts,
updates and deletions invalidate it, including SQL from external connections.
Uncommitted snapshots are never published. The default cache budget is 128 MB;
larger corpora use fresh snapshots rather than an unbounded persistent cache.
PostgreSQL/custom databases without the revision setup use fresh snapshots.

Every compatible vector is scored exactly with NumPy. There is no approximate
nearest-neighbor index, quantization, reduced dimension, or duplicate cosine
pass. Chunk loading is batched, and the winning passage supplies the snippet.
Ranking sorts unrounded scores; displayed scores retain two decimal places.

Neural semantic matches can now survive without a literal keyword overlap for
descriptive queries. The default admission threshold is 0.55. Explicit content
lookups and quoted phrases remain literal. This threshold is configurable;
the small smoke corpus below does not establish a universal relevance cutoff.

Jina reranking stays enabled when configured. It receives the best hybrid
candidates (40 by default, expanded for larger result requests), with scores for
the whole submitted pool. Identical successful requests are cached for five
minutes; failures for at most ten seconds. Its timeout remains eight seconds.
Therefore a first search can still incur network latency when Jina is enabled.

Generated explanations normally run only when the user opens the existing
context view. This changes no retrieval scores. Context selects relevant
passages instead of always explaining the first three chunks. Explanation
requests have an eight-second request timeout and reuse identical results.
`SEARCH_INLINE_EXPLANATIONS=true` restores the older inline behavior.

The file browser requests metadata and chunk counts instead of vector/text
blobs. It reloads files when the index changes, not every idle polling tick.
Status polling avoids overlapping requests and pauses while the tab is hidden.
The health API reports the effective EmbeddingGemma provider and runtime.

## Measurements on this machine

2026-10-24, RTX 3050 Laptop GPU (4 GB), CUDA PyTorch 2.11.0+cu128:

| Check | Result | Scope |
| --- | --- | --- |
| Old Python cosine retrieval, 400 docs / 2,400 vectors | 1,172.064 ms median | Synthetic vectors, 5 queries |
| New warm exact retrieval | 1.530 ms median | Same vectors and queries; excludes model, filesystem and network |
| New cold snapshot build | 885.741 ms | Once after index changes |
| Maximum cosine difference | 3.73e-8 | Floating-point rounding |
| Real EmbeddingGemma paraphrase queries | 6/6 expected top results | Six synthetic text cases; not a representative recall benchmark |
| Real local search latency | 101–225 ms | Six-document corpus, warmed model, remote providers disabled |
| Seven demo files, including an image | 3.581 s | Indexing after model warmup; 0 failures |
| Unchanged rescan | 0.036 s | All 7 files skipped |
| Peak GPU allocation during indexing | 1,124 MB | Same small demo corpus |
| Actual API upload/search/context/delete smoke | Passed; 102.57 ms search | Isolated two-document database, real EmbeddingGemma, remote providers disabled |

The backend regression suite passed 64 tests; the final affected-suite rerun
passed 52 tests. Frontend TypeScript checking and the production build passed.
The API smoke also verified that deletion invalidates cached retrieval
and that health reports EmbeddingGemma running on CUDA.

Image processing took 2.266 seconds in the final smoke run. An earlier run before
OCR/embedding overlap took 3.301 seconds. These are individual smoke runs, not
a controlled general throughput comparison. A representative corpus is needed
to quantify production indexing gains and recall. The application database and
source files were not modified by these benchmarks.

## Reproduce

From `backend/`, using the configured virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_search_speed.py --work-dir .venv
.\.venv\Scripts\python.exe scripts\benchmark_semantic_smoke.py
.\.venv\Scripts\python.exe scripts\benchmark_local_indexing.py --dataset demo_data --work-dir .venv/benchmark-local
```

All benchmark scripts isolate their databases. For everyday use, start one
backend process with `backend/start-local.ps1`; avoid multiple Uvicorn workers
or development reload when measuring latency. Restart an existing backend to
load these changes and install the additive cache revision migration. For the
frontend, run `npm run build` once after changes, then `npm run start` for
production use without the development compiler.
