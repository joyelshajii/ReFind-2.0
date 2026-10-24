# Local-fast indexing

Local-fast mode is the default. It uses the cached `google/embeddinggemma-2` model
for text queries, document passages, chunks, and direct image vectors. It does
not call the configured remote embedding provider while enabled. If the pinned model cannot be loaded from the
local cache, indexing fails clearly and does not fall back to a remote service
or to hash vectors.

## Start the backend in offline local-fast mode

From PowerShell:

```powershell
Set-Location "C:\Users\joyel\Documents\AJCE\Projects\ReFind 2.0\backend"
$env:LOCAL_FAST_MODE = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run one API process for local-fast mode. The model is cached once per process;
multiple Uvicorn workers would each load a separate copy and compete for VRAM.

Trigger a folder index using the existing API:

```powershell
$folder = "D:\Data\Documents"
$body = @{ folder_path = $folder } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/index" `
  -ContentType "application/json" -Body $body
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/index/status"
```

The existing upload endpoint also uses the configured mode. Gemini vision is
skipped during local-fast indexing; local OCR remains enabled. Local-fast OCR
uses the RapidOCR ONNX models bundled with `rapidocr-onnxruntime`, then tries
Tesseract only if that executable is already installed. PaddleOCR is not
initialized in this mode because Paddle may download missing OCR weights.
Jina reranking remains enabled when configured, with a bounded candidate pool and request cache. Generated explanations run on demand in the context view. See [performance details](performance.md).

Supported images, including HEIC, HEIF, PNG, WebP, GIF, BMP, and TIFF, are
decoded by Pillow and passed to EmbeddingGemma 2 as RGB `PIL.Image` objects.
HEIC and HEIF decoding uses `pillow-heif`; the opener is retried when processing
a HEIC/HEIF file if the library was installed after the backend started. The
indexed `file_type`, filename, filepath, hash, size, and source-format metadata
retain the source format. OCR uses the decoded source pixels, and no JPEG copy
is generated. Re-index existing images to refresh rows previously stored as
`jpg` by the earlier conversion policy.

Text and chunk passages use Sentence Transformers' `Document` prompt; queries
use `SearchQuery`. Images use the documented multimodal image input without a
text prompt. One pinned model instance serves the indexing process. Text inputs
are encoded in batches of eight (measured on the RTX 3050); text extraction is
bounded to two workers for TXT/Markdown, DOCX, and PPTX. PDFs and images are scheduled serially across files. Within an image, CPU OCR
and GPU embedding overlap while retaining both outputs. CUDA OOM
splits the text batch and retries smaller groups; an unrecoverable model error
leaves the file retryable.

`INDEX_HASH_METADATA_FAST_PATH_SECONDS` defaults to 60. A repeated scan within
that interval may reuse a hash already verified in this process when size,
high-resolution modification time, and file identity are unchanged. Metadata
is a bounded cache hint, not a permanent content proof. Set it to `0` for
hash-every-scan correctness when files may be edited while preserving size and
timestamps. The cache is process-local and bounded. This coordination is also
process-local: simultaneous same-path requests from separate backend processes
can still duplicate extraction, though the database's unique filepath
constraint prevents duplicate document rows and each row update is transactional.

## Existing database vectors

No schema migration is required. Stored vectors keep their actual provider,
model, revision, task, and dimension provenance. Search compares only vectors
compatible with the active query embedding, so old-provider vectors stop
contributing semantic scores while keywords and metadata remain available.

Re-embedding is resumable: run the existing folder index endpoint over the same
source folder again while local-fast mode is enabled. Compatible extracted
text/chunks are reused for ordinary documents; image files are reprocessed so
Gemini descriptions are not retained in local-fast content, local OCR runs,
and direct image vectors are generated. Existing document IDs are preserved
when replacing extracted records. Failed files are retried on a later scan.
During a partial re-embedding, search can contain a mix of stored vector types,
but semantic scoring filters incompatible provenance rather than mixing them.

## Reproducible end-to-end benchmark

The benchmark uses a temporary SQLite database and never connects to the
configured application database. For a representative run, prepare a read-only
folder with at least 100 TXT/Markdown documents, three PDFs, and at least one
each of a photograph, screenshot, and diagram. Create a JSON manifest:

```json
{
  "images": [
    {"path": "photos/desk.jpg", "kind": "photograph"},
    {"path": "screenshots/settings.png", "kind": "screenshot"},
    {"path": "diagrams/network.png", "kind": "diagram"}
  ],
  "exact_ocr": [
    {"path": "screenshots/settings.png", "text": "VISIBLE EXACT TEXT"}
  ]
}
```

Run the benchmark:

```powershell
Set-Location "C:\Users\joyel\Documents\AJCE\Projects\ReFind 2.0\backend"
.\.venv\Scripts\python.exe scripts\benchmark_local_indexing.py `
  --dataset "D:\Data\ReFindBenchmark" `
  --manifest "D:\Data\ReFindBenchmark\benchmark-manifest.json" `
  --require-representative --batch-size 8 --file-batch-size 4 --workers 2
```

The report separates model load/warm-up from indexing, reports stage and
per-file-type extraction time, first pass and unchanged rescan counts, exact
OCR-text search, process memory, GPU memory, and blocked remote-call attempts.
The script labels small fixture runs as smoke tests and does not claim they
establish semantic retrieval quality. There is no compatible historical
end-to-end baseline recorded yet; the benchmark intentionally does not call
Gemini or remote embedding APIs to manufacture one.

