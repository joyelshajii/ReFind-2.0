# Phase 4 stabilization notes

## Embedding provenance and migration

Each document vector and chunk vector stores its actual provider, model, output
dimension, implementation configuration identifier, and task. A remote failure
that falls back to the local hash/ngram embedder is recorded as `local`. Search
compares only vectors with compatible recorded provenance. Jina
`retrieval.query` and `retrieval.passage` are treated as a compatible pair for
the same model and configuration. Empty or malformed provenance on legacy rows
is unknown, so those rows remain searchable through text and metadata but do
not contribute semantic similarity.

`init_db()` adds nullable-by-default text columns to `documents` and `chunks`
after inspecting existing schemas. This is additive and idempotent; it does not
rewrite or delete existing vectors. An unchanged row with unknown provenance,
or a local fallback vector while a configured remote provider is available,
will be re-indexed on the next scan so its vector provenance can be refreshed.

## Same-path coordination

Folder scans and uploads use a process-local lock keyed by the resolved,
platform-normalized path. Windows path normalization is case-insensitive. The
lock is released in `finally`; uploads stage the new file and atomically replace
the destination before indexing. Different paths use different locks.

This lock does not coordinate multiple application processes. The unique
`documents.filepath` constraint prevents duplicate rows for the same stored
path string, and per-file database transactions prevent partial
document/chunk replacements. Case-sensitive database collations can still
admit case-variant Windows path strings from different processes. A
simultaneous insert can also lose with a database conflict and should be
retried. Atomic upload replacement prevents readers from seeing a half-written
upload, but separate processes can still race over which complete file version
is indexed. Deployments that require strict cross-process ordering need
database-backed coordination in a later change.

## OCR policy

There is no reliable completeness signal for Gemini transcription. The default
`IMAGE_OCR_POLICY=supplement` runs the existing local OCR fallback and merges
its lines with Gemini output. Set `IMAGE_OCR_POLICY=fallback` to run local OCR
only when Gemini returns no text. OCR results are cached by image pixels and an
OCR profile identifier. PaddleOCR remains the primary local engine,
pytesseract is attempted only when it returns no text, and the existing
RapidOCR fallback runs only if those produce no text. This policy change has
not been measured for accuracy on a representative corpus.

## Search-score interpretation

The offline local embedder and unavailable Jina reranker produce different
absolute composite scores from a configured remote provider. The canonical
search integration therefore asserts the expected top result and grounded
clues, while score thresholds belong in provider-specific tests with mocked
signals. Filename matches contribute to the filename score but do not satisfy
the content-keyword requirement by themselves.
