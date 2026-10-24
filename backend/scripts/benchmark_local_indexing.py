"""Isolated end-to-end throughput and exact-OCR benchmark for local-fast indexing.

Run with a user-provided real corpus and a JSON manifest. The benchmark creates
its own temporary SQLite database and never connects to the configured app DB.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md", ".png", ".jpg", ".jpeg",
    ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".jfif",
}


def _manifest(path: Path | None, dataset: Path, require_representative: bool):
    if path is None:
        if require_representative:
            raise ValueError("A labeled manifest is required for representative benchmark mode.")
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read benchmark manifest {path}: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("Manifest must be a JSON object with an 'images' array.")

    image_labels = []
    kinds = set()
    for entry in data["images"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Each images entry needs a path and kind.")
        image_path = Path(entry["path"])
        image_path = image_path if image_path.is_absolute() else dataset / image_path
        image_path = image_path.resolve()
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Manifest image is missing or unsupported: {image_path}")
        if os.path.normcase(os.path.commonpath((str(dataset.resolve()), str(image_path)))) != os.path.normcase(str(dataset.resolve())):
            raise ValueError(f"Manifest image must be inside the dataset folder: {image_path}")
        kind = entry.get("kind")
        if kind not in {"photograph", "screenshot", "diagram", "synthetic"}:
            raise ValueError(f"Invalid image kind {kind!r} for {image_path.name}.")
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                image.load()
        except Exception as error:
            raise ValueError(f"Manifest image cannot be decoded ({image_path}): {error}") from error
        kinds.add(kind)
        image_labels.append({"path": str(image_path), "kind": kind})

    exact_ocr = []
    for entry in data.get("exact_ocr", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("text"), str):
            raise ValueError("Each exact_ocr entry needs a path and non-empty text.")
        image_path = Path(entry["path"])
        image_path = image_path if image_path.is_absolute() else dataset / image_path
        image_path = image_path.resolve()
        if (
            not image_path.is_file()
            or os.path.normcase(os.path.commonpath((str(dataset.resolve()), str(image_path))))
            != os.path.normcase(str(dataset.resolve()))
        ):
            raise ValueError(f"OCR manifest image is missing or outside the dataset: {image_path}")
        if not entry["text"].strip():
            raise ValueError(f"OCR label for {image_path.name} must not be blank.")
        exact_ocr.append({"path": str(image_path), "text": entry["text"]})

    labeled_image_paths = {entry["path"] for entry in image_labels}
    if any(entry["path"] not in labeled_image_paths for entry in exact_ocr):
        raise ValueError("Every exact_ocr entry must refer to an image listed in the manifest.")

    if require_representative:
        required = {"photograph", "screenshot", "diagram"}
        if not required.issubset(kinds):
            raise ValueError(f"Representative manifest needs image kinds {sorted(required)}; found {sorted(kinds)}.")
        if not exact_ocr:
            raise ValueError("Representative manifest needs at least one exact_ocr label.")
    return image_labels, exact_ocr


def _memory_mb(peak: bool = False) -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        if not peak:
            return round(info.rss / 1048576, 2)
        return round(getattr(info, "peak_wset", info.rss) / 1048576, 2)
    except Exception:
        return None


class _MetricHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.rows = []

    def emit(self, record):
        message = record.getMessage()
        prefix = "Indexing metrics: "
        if message.startswith(prefix):
            try:
                self.rows.append(json.loads(message[len(prefix):]))
            except json.JSONDecodeError:
                pass


def run(args) -> dict[str, Any]:
    # Force a cache-only, API-free runtime before importing app modules. The
    # configured database URL is deliberately ignored by the isolated engine.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["LOCAL_FAST_MODE"] = "true"
    os.environ["DATABASE_URL"] = ""
    os.environ["SQLITE_FALLBACK_PATH"] = ":memory:"
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["JINA_API_KEY"] = ""
    os.environ["LLM_PROVIDER"] = "local"
    sys.path.insert(0, str(BACKEND_DIR))

    import torch
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.db.base import Base
    from app.db.models import Document
    from app.services import embedding_service, indexer
    from app.services.hybrid_search import search_documents
    from app.services.query_parser import parse_query
    from app.connectors.local_source import LocalFileSource

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset folder does not exist: {dataset}")
    image_labels, exact_ocr = _manifest(args.manifest, dataset, args.require_representative)
    discovered = LocalFileSource().scan(str(dataset))
    counts = Counter(item["file_type"] for item in discovered)
    text_count = sum(counts[extension] for extension in ("txt",))
    pdf_count = counts["pdf"]
    image_count = sum(counts[extension] for extension in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"))
    if args.require_representative:
        if text_count < 100:
            raise ValueError(f"Representative mode needs >=100 TXT/Markdown documents; found {text_count} TXT files.")
        if pdf_count < 3:
            raise ValueError(f"Representative mode needs at least 3 PDFs; found {pdf_count}.")
        if image_count < 3 or len(image_labels) < 3:
            raise ValueError("Representative mode needs at least 3 labeled local images.")

    settings.LOCAL_FAST_MODE = True
    settings.LOCAL_EMBEDDING_BATCH_SIZE = args.batch_size
    settings.INDEX_EMBEDDING_FILE_BATCH_SIZE = args.file_batch_size
    settings.INDEX_EXTRACTION_WORKERS = args.workers
    settings.INDEX_HASH_METADATA_FAST_PATH_SECONDS = args.hash_cache_seconds
    settings.GEMINI_API_KEY = ""
    settings.JINA_API_KEY = ""
    settings.LLM_PROVIDER = "local"

    # Guard against regressions into network providers. The calls are counted
    # and fail immediately if any indexing code attempts one.
    remote_calls = Counter()
    original_gemini = embedding_service._embed_gemini
    original_jina = embedding_service._embed_jina
    original_vision = __import__("app.processors.image_processor", fromlist=["_analyze_image_payload_with_gemini"])._analyze_image_payload_with_gemini

    def deny_remote(name):
        def denied(*_args, **_kwargs):
            remote_calls[name] += 1
            raise RuntimeError(f"Remote call blocked in offline benchmark: {name}")
        return denied

    embedding_service._embed_gemini = deny_remote("gemini_embedding")
    embedding_service._embed_jina = deny_remote("jina_embedding")
    image_processor = __import__("app.processors.image_processor", fromlist=["_analyze_image_payload_with_gemini"])
    image_processor._analyze_image_payload_with_gemini = deny_remote("gemini_vision")

    worker_times = defaultdict(list)
    worker_lock = threading.Lock()
    original_process = indexer.process_file

    def timed_process(filepath):
        started = time.perf_counter()
        result = original_process(filepath)
        elapsed = time.perf_counter() - started
        with worker_lock:
            worker_times[Path(filepath).suffix.lower().lstrip(".")].append(elapsed)
        return result

    indexer.process_file = timed_process
    work_root = (args.work_dir or BACKEND_DIR / ".benchmark-work").resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("refind.services.indexer")
    handler = _MetricHandler()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    temp_dir = None
    engine = None
    try:
        temp_dir = tempfile.TemporaryDirectory(prefix="local-index-", dir=work_root)
        sqlite_path = Path(temp_dir.name) / "benchmark.sqlite"
        engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False, "timeout": 60})
        Base.metadata.create_all(engine)
        SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        indexer.SessionLocal = SessionFactory

        gpu_before = torch.cuda.mem_get_info() if torch.cuda.is_available() else None
        process_memory_before = _memory_mb()
        load_started = time.perf_counter()
        embedding_service._get_embeddinggemma_model()
        model_load_seconds = time.perf_counter() - load_started
        # Warm text query/document and image encoder paths separately. Clear
        # the vector cache so the first indexed file still measures inference.
        warmup_started = time.perf_counter()
        embedding_service.embed_texts_with_provenance(
            ["local indexer document warm-up"], task="retrieval.passage"
        )
        text_warmup_seconds = time.perf_counter() - warmup_started
        image_warmup_seconds = None
        warmup_image = image_labels[0]["path"] if image_labels else next(
            (item["filepath"] for item in discovered if item["file_type"] in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}),
            None,
        )
        if warmup_image:
            from PIL import Image
            with Image.open(warmup_image) as image:
                image.load()
                warmup_started = time.perf_counter()
                embedding_service.embed_image_with_provenance(image)
                image_warmup_seconds = time.perf_counter() - warmup_started
        embedding_service._embeddinggemma_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        indexer.index_folder_background(str(dataset))
        first_pass_seconds = time.perf_counter() - start
        first_metrics = handler.rows[-1] if handler.rows else {}
        first_status = indexer.indexing_status.current_stage

        # Measure unchanged-rescan behavior separately; metadata hash reuse is
        # bounded by the CLI TTL and may be set to zero for strict rehashing.
        start = time.perf_counter()
        indexer.index_folder_background(str(dataset))
        rescan_seconds = time.perf_counter() - start
        rescan_metrics = handler.rows[-1] if len(handler.rows) >= 2 else {}

        gpu_after = torch.cuda.mem_get_info() if torch.cuda.is_available() else None
        gpu_peak_allocated = torch.cuda.max_memory_allocated() / 1048576 if torch.cuda.is_available() else None
        process_memory_after = _memory_mb()
        process_memory_peak = _memory_mb(peak=True)

        exact_results = []
        with SessionFactory() as db:
            for entry in exact_ocr:
                doc = db.query(Document).filter_by(filepath=entry["path"]).first()
                document_text_found = bool(doc and entry["text"].casefold() in (doc.extracted_text or "").casefold())
                query_started = time.perf_counter()
                results = search_documents(entry["text"], parse_query(entry["text"]), db, top_k=10)
                query_seconds = time.perf_counter() - query_started
                exact_results.append({
                    "file": Path(entry["path"]).name,
                    "text": entry["text"],
                    "text_present_after_ocr": document_text_found,
                    "search_returned_expected_file": any(result.filepath == entry["path"] for result in results),
                    "search_latency_seconds": round(query_seconds, 4),
                })

        file_type_timing = {
            extension: {
                "processor_calls": len(times),
                "processor_elapsed_seconds_sum": round(sum(times), 4),
                "processor_elapsed_seconds_mean": round(statistics.mean(times), 4),
            }
            for extension, times in sorted(worker_times.items())
        }
        runtime = embedding_service.local_embedding_runtime_stats()
        return {
            "dataset": str(dataset),
            "dataset_label": "user-provided local corpus" if args.require_representative else "smoke corpus; not representative evidence",
            "files_discovered_by_type": dict(sorted(counts.items())),
            "total_files_discovered": len(discovered),
            "first_pass": {
                "status": first_status,
                "seconds": round(first_pass_seconds, 4),
                "files_indexed": first_metrics.get("files_indexed"),
                "files_skipped": first_metrics.get("files_skipped"),
                "failed_files": first_metrics.get("failed_files"),
                "files_per_second": round(first_metrics.get("files_indexed", 0) / max(first_pass_seconds, 1e-9), 4),
                "stage_seconds": first_metrics.get("stage_seconds"),
                "image_substage_seconds": first_metrics.get("image_substage_seconds"),
                "pdf_substage_seconds": first_metrics.get("pdf_substage_seconds"),
                "time_by_file_type": file_type_timing,
            },
            "unchanged_rescan": {
                "seconds": round(rescan_seconds, 4),
                "files_indexed": rescan_metrics.get("files_indexed"),
                "files_skipped": rescan_metrics.get("files_skipped"),
                "failed_files": rescan_metrics.get("failed_files"),
                "hash_fast_path_seconds": args.hash_cache_seconds,
            },
            "model": {
                **runtime,
                "load_seconds": round(model_load_seconds, 4),
                "text_warmup_seconds": round(text_warmup_seconds, 4),
                "image_warmup_seconds": round(image_warmup_seconds, 4) if image_warmup_seconds is not None else None,
                "batch_size": args.batch_size,
                "extraction_workers": args.workers,
            },
            "memory": {
                "process_rss_before_mb": process_memory_before,
                "process_rss_after_mb": process_memory_after,
                "process_peak_mb": process_memory_peak,
                "gpu_free_before_mb": round(gpu_before[0] / 1048576, 1) if gpu_before else None,
                "gpu_free_after_mb": round(gpu_after[0] / 1048576, 1) if gpu_after else None,
                "gpu_total_mb": round(gpu_after[1] / 1048576, 1) if gpu_after else None,
                "gpu_peak_allocated_mb": round(gpu_peak_allocated, 1) if gpu_peak_allocated is not None else None,
            },
            "exact_ocr_search": exact_results,
            "remote_api_calls": dict(remote_calls),
            "baseline_comparison": "Not measured: no compatible historical end-to-end baseline was available; live remote calls are intentionally blocked.",
            "limitations": [
                "The benchmark does not establish semantic search quality; it only checks exact OCR labels and indexing throughput.",
                "Representative mode requires user-labeled photos, screenshots, diagrams, and OCR strings.",
                "A smoke corpus is not real-world throughput evidence.",
            ],
        }
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        indexer.process_file = original_process
        embedding_service._embed_gemini = original_gemini
        embedding_service._embed_jina = original_jina
        image_processor._analyze_image_payload_with_gemini = original_vision
        if engine is not None:
            engine.dispose()
        if temp_dir is not None:
            temp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder containing files to index; source files are read-only.")
    parser.add_argument("--manifest", type=Path, help="JSON with image kind labels and exact_ocr strings.")
    parser.add_argument("--require-representative", action="store_true", help="Require >=100 text files, >=3 PDFs, labeled photo/screenshot/diagram, and OCR terms.")
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--file-batch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--hash-cache-seconds", type=float, default=60.0, help="Metadata-matched hash reuse TTL; use 0 for hash-every-scan.")
    parser.add_argument("--work-dir", type=Path, help="Writable temporary folder for isolated benchmark SQLite data.")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
