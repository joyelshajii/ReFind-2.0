import os
import re
import hashlib
import logging
import json
import time
import sys
import threading
from contextlib import contextmanager, ExitStack, nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from app.db.base import SessionLocal
from app.db.models import Document, Chunk, Entity
from app.schemas.index import IndexProgressStatus
from app.processors.factory import process_file, get_file_type, AUDIO_TYPES, VIDEO_TYPES, MEDIA_TYPES
from app.services.audio_service import AUDIO_PROCESSOR_VERSION, SEGMENT_SECONDS, OVERLAP_SECONDS
from app.services.video_service import VIDEO_PROCESSOR_VERSION, VIDEO_SEGMENT_SECONDS, VIDEO_OVERLAP_SECONDS, VIDEO_FRAME_INTERVAL
from app.services.media_preflight import inspect_media, MEDIA_POLICY_VERSION, MediaConfirmationRequired
from app.connectors.local_source import LocalFileSource
from app.services.embedding_service import (
    get_embedding_with_provenance, expected_embedding_provenance,
    embedding_signature_is_compatible, embed_texts_with_provenance,
    embed_image_with_provenance, local_embedding_runtime_stats,
)
from app.services.query_parser import COMMON_TECH, COMMON_TOPICS
from app.processors.image_processor import IMAGE_PROCESSOR_VERSION
from app.core.config import settings

logger = logging.getLogger("refind.services.indexer")

indexing_status = IndexProgressStatus()
_indexing_run_lock = threading.Lock()
_path_locks_guard = threading.Lock()
_path_locks = {}
_hash_cache_lock = threading.Lock()
_verified_hash_cache = {}


def _display_index_error(error: Exception) -> str:
    """Return useful file errors without leaking database or provider internals."""
    if isinstance(error, SQLAlchemyError):
        return "Database write failed. Check backend logs for details."
    if isinstance(error, PermissionError):
        return "Permission denied while reading or writing the file."
    if isinstance(error, FileNotFoundError):
        return "The file was moved or removed during indexing."
    if isinstance(error, (ValueError, RuntimeError)):
        return str(error)
    return f"Processing failed ({type(error).__name__}). Check backend logs for details."


def _record_index_error(filename: str, error: Exception) -> None:
    """Keep the first few distinct failures visible while retaining every trace in logs."""
    entry = f"{filename}: {_display_index_error(error)}"
    entries = indexing_status.error_message.splitlines() if indexing_status.error_message else []
    if entry in entries or any(line.startswith("Additional failures omitted") for line in entries):
        return
    if len(entries) < 5:
        entries.append(entry)
    else:
        entries.append("Additional failures omitted; see backend logs for details.")
    indexing_status.error_message = "\n".join(entries)


@contextmanager
def file_index_lock(filepath: str):
    """Serialize indexing and upload writes for one normalized path per process."""
    path_key = os.path.normcase(str(Path(filepath).resolve()))
    with _path_locks_guard:
        entry = _path_locks.get(path_key)
        lock, users = entry if entry is not None else (threading.RLock(), 0)
        _path_locks[path_key] = (lock, users + 1)
    acquired = False
    try:
        lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            lock.release()
        with _path_locks_guard:
            current_lock, users = _path_locks[path_key]
            if current_lock is lock and users == 1:
                del _path_locks[path_key]
            else:
                _path_locks[path_key] = (current_lock, users - 1)


def _find_existing_document(db: Session, filepath: str):
    resolved = str(Path(filepath).resolve())
    query = db.query(Document)
    if os.name == "nt":
        return query.filter(func.lower(Document.filepath) == resolved.lower()).first()
    return query.filter(Document.filepath == resolved).first()


def _document_embeddings_match_configuration(doc: Document) -> bool:
    if (settings.LOCAL_FAST_MODE and doc.file_type in MEDIA_TYPES
            and inspect_media(doc.filepath).strategy == "direct"):
        required_task = "retrieval.audio" if doc.file_type in AUDIO_TYPES else "retrieval.video"
        media_chunks = [chunk for chunk in doc.chunks
                        if (chunk.get_embedding_provenance() or {}).get("task") in {"retrieval.audio", "retrieval.video"}]
        if (not any(chunk.get_embedding_provenance().get("task") == required_task for chunk in media_chunks)
                or any(not chunk.get_embedding() for chunk in media_chunks)):
            return False
    expected_doc = expected_embedding_provenance(
        "retrieval.image" if doc.file_type in IMAGE_TYPES else "retrieval.passage"
    )
    if not doc.get_embedding() or not embedding_signature_is_compatible(
        doc.get_embedding_provenance(), expected_doc
    ):
        return False
    expected_chunk = expected_embedding_provenance("retrieval.passage")
    return all(
        not chunk.get_embedding()
        or embedding_signature_is_compatible(chunk.get_embedding_provenance(), expected_chunk)
        for chunk in doc.chunks
    )


def _processing_state_matches(doc: Document, file_type: str, signature: str) -> bool:
    if doc.processing_signature == signature:
        return True
    if file_type not in IMAGE_TYPES | MEDIA_TYPES and not doc.processing_signature:
        return True
    try:
        stored = json.loads(doc.processing_signature or "{}")
        current = json.loads(signature)
    except (TypeError, ValueError):
        return False

    if (file_type in MEDIA_TYPES and current.get("strategy") == "direct"
            and "media_policy" not in stored):
        from app.services.media_preflight import companion_files
        # Existing complete direct media vectors remain valid. A new transcript
        # or any companion requires the new inspection policy to run.
        if not companion_files(doc.filepath):
            legacy_current = {key: value for key, value in current.items()
                              if key not in {"media_policy", "strategy", "media_fingerprint"}}
            stored["media_embedding_config"] = stored.get("audio_embedding_config")
            return all(stored.get(key) == value for key, value in legacy_current.items())

    # Older signatures included embedding provider settings. Only compare
    # fields that affect extracted text, chunks, or OCR so a vector-space
    # change can reuse safe cached extraction and refresh vectors alone.
    extraction_fields = set(current)
    if file_type in IMAGE_TYPES and "local_fast_mode" not in stored:
        stored["local_fast_mode"] = False
    return all(stored.get(key) == current.get(key) for key in extraction_fields)


def _refresh_document_embeddings(doc: Document, db: Session):
    """Refresh vectors from already extracted text without decoding/reprocessing the file."""
    embedding_started = time.perf_counter()
    if doc.file_type in MEDIA_TYPES:
        # Media vectors must be regenerated from samples, never timestamp labels.
        _index_single_file_locked(doc.filepath, db, force_reprocess=True)
        return time.perf_counter() - embedding_started, 0.0
    image_doc = doc.file_type in IMAGE_TYPES and settings.LOCAL_FAST_MODE
    texts = []
    if not image_doc:
        texts.append(doc.extracted_text[:1500] if doc.extracted_text else doc.filename)
    chunk_positions = [index for index, chunk in enumerate(doc.chunks) if chunk.content.strip()]
    texts.extend(doc.chunks[index].content.strip() for index in chunk_positions)
    vectors = embed_texts_with_provenance(texts, task="retrieval.passage")
    vector_index = 0
    if image_doc:
        from PIL import Image
        if Path(doc.filepath).suffix.lower() in {".heic", ".heif"}:
            from app.processors.image_processor import ensure_heif_opener
            ensure_heif_opener()
        with Image.open(doc.filepath) as image:
            image.load()
            doc_vector, doc_provenance = embed_image_with_provenance(image)
    else:
        doc_vector, doc_provenance = vectors[0]
        vector_index = 1
    chunk_vectors = [(None, None)] * len(doc.chunks)
    for chunk_index in chunk_positions:
        chunk_vectors[chunk_index] = vectors[vector_index]
        vector_index += 1
    embedding_seconds = time.perf_counter() - embedding_started

    database_started = time.perf_counter()
    try:
        doc.set_embedding(doc_vector)
        doc.set_embedding_provenance(doc_provenance)
        for chunk, (vector, provenance) in zip(doc.chunks, chunk_vectors):
            chunk.set_embedding(vector or [])
            chunk.set_embedding_provenance(provenance)
        doc.indexed_at = datetime.utcnow()
        db.commit()
        db.refresh(doc)
    except Exception:
        db.rollback()
        raise
    return embedding_seconds, time.perf_counter() - database_started


def try_reserve_indexing_run() -> bool:
    """Atomically reserve the single background indexing worker slot."""
    if not _indexing_run_lock.acquire(blocking=False):
        return False
    indexing_status.is_indexing = True
    return True


def release_indexing_run():
    indexing_status.is_indexing = False
    _indexing_run_lock.release()

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}
PIPELINE_VERSION = 3


def _processing_signature(filepath: str, file_type: str) -> str:
    """Versioned extraction config, independent of vector-provider provenance."""
    signature = {
        "file_type": file_type,
    }
    if file_type in IMAGE_TYPES:
        signature.update({
            "pipeline": PIPELINE_VERSION,
            "image_processor": IMAGE_PROCESSOR_VERSION,
            "vision_enabled": bool(
                settings.IMAGE_VISION_ENABLED and settings.GEMINI_API_KEY and not settings.LOCAL_FAST_MODE
            ),
            "vision_model": settings.GEMINI_MODEL,
            "local_ocr": (
                "rapidocr-onnxruntime-bundled+tesseract-fallback-v1"
                if settings.LOCAL_FAST_MODE
                else f"paddle-tesseract-v1:{settings.IMAGE_OCR_POLICY}"
            ),
            "local_fast_mode": bool(settings.LOCAL_FAST_MODE),
        })
        if settings.LOCAL_FAST_MODE:
            # A context-explanation model change cannot invalidate local OCR.
            signature.pop("vision_model", None)
        sidecar = Path(f"{filepath}.meta.json")
        if sidecar.exists():
            sidecar_hash = compute_file_hash(str(sidecar))
            if not sidecar_hash:
                return ""
            signature["sidecar_sha256"] = sidecar_hash
    else:
        signature["chunk_size"] = settings.CHUNK_SIZE
        signature["chunk_overlap"] = settings.CHUNK_OVERLAP
    if file_type in MEDIA_TYPES:
        plan = inspect_media(filepath)
        signature.update({"media_policy": MEDIA_POLICY_VERSION, "strategy": plan.strategy,
                          "media_fingerprint": plan.fingerprint,
                          "local_fast_mode": bool(settings.LOCAL_FAST_MODE)})
        if plan.strategy == "direct":
            signature.update({
                "audio_processor": AUDIO_PROCESSOR_VERSION,
                "segment_seconds": SEGMENT_SECONDS, "overlap_seconds": OVERLAP_SECONDS,
                "media_embedding_config": expected_embedding_provenance("retrieval.audio")["config_id"],
            })
            if file_type in VIDEO_TYPES:
                signature.update(video_processor=VIDEO_PROCESSOR_VERSION,
                                 video_segment_seconds=VIDEO_SEGMENT_SECONDS,
                                 video_overlap_seconds=VIDEO_OVERLAP_SECONDS,
                                 video_frame_interval=VIDEO_FRAME_INTERVAL)
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _process_memory_mb(peak: bool = False) -> Optional[float]:
    """Return process working-set memory where supported, without a hard dependency."""
    try:
        import psutil
        memory = psutil.Process().memory_info()
        if not peak:
            return round(memory.rss / (1024 * 1024), 2)
        peak_working_set = getattr(memory, "peak_wset", None)
        if peak_working_set is not None:
            return round(peak_working_set / (1024 * 1024), 2)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                value = counters.PeakWorkingSetSize if peak else counters.WorkingSetSize
                return round(value / (1024 * 1024), 2)
        except Exception:
            return None
    else:
        try:
            if not peak and sys.platform.startswith("linux"):
                with open("/proc/self/statm", "r", encoding="ascii") as statm:
                    resident_pages = int(statm.read().split()[1])
                return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 2)
            if peak:
                import resource
                value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # Linux reports KiB; macOS reports bytes.
                return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
        except Exception:
            return None
    return None

def compute_file_hash(filepath: str) -> str:
    """
    Computes SHA-256 hash of a file to detect whether it has changed.
    """
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Error computing hash for {filepath}: {e}")
        return ""


def _compute_index_hash(filepath: str) -> str:
    """Reuse a recently verified hash only while file identity metadata matches."""
    ttl_seconds = max(0.0, float(settings.INDEX_HASH_METADATA_FAST_PATH_SECONDS))
    if ttl_seconds <= 0:
        return compute_file_hash(filepath)
    try:
        path = Path(filepath).resolve()
        stat = path.stat()
        key = os.path.normcase(str(path))
        metadata = (
            stat.st_size,
            getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)),
            getattr(stat, "st_ino", 0),
            getattr(stat, "st_dev", 0),
        )
        now = time.monotonic()
        with _hash_cache_lock:
            cached = _verified_hash_cache.get(key)
            if cached and cached[0] == metadata and now - cached[1] <= ttl_seconds:
                return cached[2]
        file_hash = compute_file_hash(str(path))
        if file_hash:
            with _hash_cache_lock:
                _verified_hash_cache[key] = (metadata, now, file_hash)
                while len(_verified_hash_cache) > 8192:
                    _verified_hash_cache.pop(next(iter(_verified_hash_cache)))
        return file_hash
    except OSError as error:
        logger.warning("Unable to stat file for hash fast path %s: %s", filepath, error)
        return compute_file_hash(filepath)

def extract_entities_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extracts high-value domain entities (technologies, people, topics) from document text.
    """
    entities: List[Dict[str, str]] = []
    lower = text.lower()

    # Technologies
    for tech in COMMON_TECH:
        pattern = r"\b" + re.escape(tech) + r"\b"
        if re.search(pattern, lower):
            entities.append({
                "entity_type": "technology",
                "entity_value": tech.upper() if tech in ("ble", "iot", "api", "rest") else tech.title()
            })

    # Topics
    for top in COMMON_TOPICS:
        pattern = r"\b" + re.escape(top) + r"\b"
        if re.search(pattern, lower):
            entities.append({
                "entity_type": "topic",
                "entity_value": top.title()
            })

    # People detection
    name_patterns = [
        r"\b(?:Author|Sent by|Shared by|Presenter|Owner):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+)\s+sent\s+this\b",
        r"\b(?:with|from)\s+([A-Z][a-z]+)\b",
    ]
    for np in name_patterns:
        matches = re.findall(np, text)
        for m in matches:
            if m.lower() not in ("the", "find", "document", "architecture"):
                entities.append({
                    "entity_type": "person",
                    "entity_value": m.strip()
                })

    # Deduplicate
    unique = []
    seen = set()
    for e in entities:
        key = (e["entity_type"], e["entity_value"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def _process_file_timed(filepath: str):
    started = time.perf_counter()
    result = process_file(filepath)
    return result, time.perf_counter() - started


def _index_folder_local_fast(db: Session, files: list[dict], metrics: dict) -> None:
    """Reuse CPU extraction workers across the entire scan."""
    # Finish short recordings before a long movie holds an extraction batch.
    files = sorted(files, key=lambda item: item.get("file_size", 0))
    with ThreadPoolExecutor(max_workers=max(1, min(settings.INDEX_EXTRACTION_WORKERS, 4))) as pool:
        _index_folder_local_fast_batches(db, files, metrics, pool)


def _index_folder_local_fast_batches(db, files, metrics, extraction_pool):
    """Index bounded file groups with local text batches and per-file commits."""
    file_batch_size = max(1, min(int(settings.INDEX_EMBEDDING_FILE_BATCH_SIZE), 8))
    for batch_start in range(0, len(files), file_batch_size):
        file_batch = files[batch_start:batch_start + file_batch_size]
        pending = []
        to_extract = []
        # Acquire a bounded set of path locks in a consistent order. The locks
        # remain held through commit, coordinating uploads with folder scans.
        lock_paths = sorted(
            (item["filepath"] for item in file_batch),
            key=lambda value: os.path.normcase(str(Path(value).resolve())),
        )
        with ExitStack() as path_locks:
            for lock_path in lock_paths:
                path_locks.enter_context(file_index_lock(lock_path))

            for offset, file_info in enumerate(file_batch):
                file_index = batch_start + offset
                filepath = file_info["filepath"]
                filename = file_info["filename"]
                file_type = file_info["file_type"]
                is_image = file_type in IMAGE_TYPES
                file_started = time.perf_counter()
                indexing_status.current_file = filename
                if is_image:
                    metrics["image_bytes"] += file_info.get("file_size", 0)

                stage_started = time.perf_counter()
                current_hash = _compute_index_hash(filepath)
                existing_doc = _find_existing_document(db, filepath)
                try:
                    current_signature = _processing_signature(filepath, file_type)
                except Exception as error:
                    metrics["failed_files"] += 1
                    _record_index_error(filename, error)
                    indexing_status.processed_files = file_index + 1
                    continue
                metrics["hash_seconds"] += time.perf_counter() - stage_started
                unchanged = bool(
                    existing_doc
                    and existing_doc.file_hash == current_hash
                    and current_hash
                    and current_signature
                    and _processing_state_matches(existing_doc, file_type, current_signature)
                )
                if unchanged and _document_embeddings_match_configuration(existing_doc):
                    metrics["skipped_files"] = metrics.get("skipped_files", 0) + 1
                    if is_image:
                        metrics["skipped_images"] += 1
                    indexing_status.processed_files = file_index + 1
                    continue

                if unchanged:
                    try:
                        embedding_seconds, database_seconds = _refresh_document_embeddings(existing_doc, db)
                        metrics["embedding_seconds"] += embedding_seconds
                        metrics["database_seconds"] += database_seconds
                        metrics["indexed_files"] = metrics.get("indexed_files", 0) + 1
                        if is_image:
                            metrics["indexed_images"] += 1
                            metrics["image_index_seconds"] += time.perf_counter() - file_started
                    except Exception as refresh_error:
                        if isinstance(refresh_error, MediaConfirmationRequired):
                            from app.services.media_jobs import defer_media
                            defer_media(refresh_error.plan)
                            metrics["deferred_files"] = metrics.get("deferred_files", 0) + 1
                            indexing_status.processed_files = file_index + 1
                            continue
                        metrics["failed_files"] += 1
                        logger.error("Failed to refresh embeddings for %s: %s", filepath, refresh_error, exc_info=True)
                        _record_index_error(filename, refresh_error)
                    indexing_status.processed_files = file_index + 1
                    continue

                to_extract.append({
                    "file_index": file_index,
                    "file_info": file_info,
                    "file_type": file_type,
                    "filepath": filepath,
                    "filename": filename,
                    "file_started": file_started,
                    "existing_doc_id": existing_doc.id if existing_doc else None,
                    "current_hash": current_hash,
                    "current_signature": current_signature,
                })

            # End the batch's read transaction before CPU extraction, OCR, and
            # model inference. The existing row ID is enough to re-fetch just
            # before its atomic write.
            db.commit()

            if to_extract:
                indexing_status.current_stage = "Extracting"
                extraction_batch_started = time.perf_counter()
                parallel_items = [
                    item for item in to_extract
                    if item["file_type"] not in IMAGE_TYPES and item["file_type"] != "pdf"
                ]
                worker_count = min(
                    max(1, min(int(settings.INDEX_EXTRACTION_WORKERS), 4)),
                    max(1, len(parallel_items)),
                )
                futures = {}
                with nullcontext(extraction_pool) as pool:
                    if worker_count > 1:
                        futures = {id(item): pool.submit(_process_file_timed, item["filepath"]) for item in parallel_items}
                    for item in to_extract:
                        indexing_status.current_file = item["filename"]
                        try:
                            if id(item) in futures:
                                extracted_data, extraction_seconds = futures[id(item)].result()
                            else:
                                extracted_data, extraction_seconds = _process_file_timed(item["filepath"])
                            extracted_text = extracted_data.get("extracted_text", "")
                            raw_chunks = extracted_data.get("chunks", [])
                            is_image = item["file_type"] in IMAGE_TYPES
                            if is_image and not extracted_data.get("image_embedding"):
                                raise RuntimeError("Local image embedding did not produce a vector.")
                        except Exception as file_error:
                            if isinstance(file_error, MediaConfirmationRequired):
                                from app.services.media_jobs import defer_media
                                defer_media(file_error.plan)
                                metrics["deferred_files"] = metrics.get("deferred_files", 0) + 1
                                indexing_status.processed_files = item["file_index"] + 1
                                continue
                            metrics["failed_files"] += 1
                            logger.error("Failed to process %s: %s", item["filepath"], file_error, exc_info=True)
                            _record_index_error(item["filename"], file_error)
                            indexing_status.processed_files = item["file_index"] + 1
                            continue
                        if is_image:
                            metrics["image_extraction_seconds"] += extraction_seconds
                        for metric_name in (
                            "decode_metadata_seconds", "vision_preprocess_seconds", "vision_seconds",
                            "ocr_seconds", "image_embedding_seconds", "pdf_text_extraction_seconds",
                            "pdf_render_seconds", "pdf_ocr_seconds",
                        ):
                            metrics[metric_name] = metrics.get(metric_name, 0.0) + float(
                                extracted_data.get("timings", {}).get(metric_name, 0.0) or 0.0
                            )
                        entities_data = extract_entities_from_text(extracted_text)
                        for entity in extracted_data.get("entities", []):
                            if entity not in entities_data:
                                entities_data.append(entity)
                        pending.append({
                            **item,
                            "extraction_seconds": extraction_seconds,
                            "extracted_data": extracted_data,
                            "extracted_text": extracted_text,
                            "chunks": raw_chunks,
                            "entities": entities_data,
                            "image_vector": extracted_data.get("image_embedding"),
                            "image_provenance": extracted_data.get("image_embedding_provenance"),
                        })
                metrics["extraction_seconds"] += time.perf_counter() - extraction_batch_started

            if pending:
                indexing_status.current_stage = "Understanding"
                text_inputs = []
                for item in pending:
                    item["doc_text_position"] = None
                    item["chunk_positions"] = {}
                    if item["file_type"] not in IMAGE_TYPES:
                        item["doc_text_position"] = len(text_inputs)
                        text_inputs.append(item["extracted_text"][:1500] or item["filename"])
                    for chunk_index, chunk in enumerate(item["chunks"]):
                        chunk_text = chunk.get("content", "").strip()
                        if chunk_text and chunk.get("_embedding") is None:
                            item["chunk_positions"][chunk_index] = len(text_inputs)
                            text_inputs.append(chunk_text)

                stage_started = time.perf_counter()
                try:
                    text_vectors = embed_texts_with_provenance(text_inputs, task="retrieval.passage")
                except Exception as embedding_error:
                    metrics["embedding_seconds"] += time.perf_counter() - stage_started
                    metrics["failed_files"] += len(pending)
                    for item in pending:
                        logger.error("Local embedding failed for %s: %s", item["filepath"], embedding_error, exc_info=True)
                        _record_index_error(item["filename"], embedding_error)
                        indexing_status.processed_files = item["file_index"] + 1
                    pending = []
                else:
                    metrics["embedding_seconds"] += time.perf_counter() - stage_started

                for item in pending:
                    stage_started = time.perf_counter()
                    indexing_status.current_stage = "Indexing"
                    try:
                        file_info = item["file_info"]
                        stat = Path(item["filepath"]).stat()
                        if item["doc_text_position"] is None:
                            doc_vector = item["image_vector"]
                            doc_provenance = item["image_provenance"]
                        else:
                            doc_vector, doc_provenance = text_vectors[item["doc_text_position"]]
                        doc = (
                            db.query(Document).filter(Document.id == item["existing_doc_id"]).first()
                            if item["existing_doc_id"] else None
                        )
                        if doc is None:
                            doc = Document(filepath=str(Path(item["filepath"]).resolve()))
                            db.add(doc)
                        else:
                            doc.chunks.clear()
                            doc.entities.clear()
                            db.flush()
                        doc.filename = item["filename"]
                        doc.filepath = str(Path(item["filepath"]).resolve())
                        doc.file_type = item["file_type"]
                        doc.file_size = stat.st_size
                        doc.file_hash = item["current_hash"]
                        doc.processing_signature = item["current_signature"]
                        doc.created_at = datetime.fromtimestamp(stat.st_ctime)
                        doc.modified_at = datetime.fromtimestamp(stat.st_mtime)
                        doc.source = "local"
                        doc.extracted_text = item["extracted_text"]
                        doc.summary = item["extracted_text"][:300] if item["extracted_text"] else ""
                        doc.indexed_at = datetime.utcnow()
                        doc.set_embedding(doc_vector)
                        doc.set_embedding_provenance(doc_provenance)
                        db.add(doc)
                        db.flush()
                        for chunk_index, chunk_info in enumerate(item["chunks"]):
                            chunk_text = chunk_info.get("content", "").strip()
                            if not chunk_text:
                                continue
                            if chunk_info.get("_embedding") is not None:
                                chunk_vector = chunk_info["_embedding"]
                                chunk_provenance = chunk_info["_embedding_provenance"]
                            else:
                                chunk_vector, chunk_provenance = text_vectors[item["chunk_positions"][chunk_index]]
                            chunk_model = Chunk(
                                document_id=doc.id,
                                chunk_index=chunk_index,
                                content=chunk_text,
                                page_number=chunk_info.get("page_number"),
                                slide_number=chunk_info.get("slide_number"),
                            )
                            chunk_model.set_embedding(chunk_vector)
                            chunk_model.set_embedding_provenance(chunk_provenance)
                            db.add(chunk_model)
                        for entity in item["entities"]:
                            db.add(Entity(
                                document_id=doc.id,
                                entity_type=entity["entity_type"],
                                entity_value=entity["entity_value"],
                            ))
                        if item["file_type"] in MEDIA_TYPES:
                            from app.services.media_jobs import complete_media_request
                            complete_media_request(db, doc.filepath)
                        db.commit()
                        metrics["indexed_files"] = metrics.get("indexed_files", 0) + 1
                        if item["file_type"] in IMAGE_TYPES:
                            metrics["indexed_images"] += 1
                            metrics["image_index_seconds"] += time.perf_counter() - item["file_started"]
                    except Exception as database_error:
                        db.rollback()
                        metrics["failed_files"] += 1
                        logger.error("Failed to save %s: %s", item["filepath"], database_error, exc_info=True)
                        _record_index_error(item["filename"], database_error)
                    finally:
                        metrics["database_seconds"] += time.perf_counter() - stage_started
                        indexing_status.processed_files = item["file_index"] + 1


def index_folder_background(folder_path: str, _lock_reserved: bool = False, _discovered_files=None):
    """
    Background worker that runs the multi-stage indexing workflow:
    Reading -> Extracting -> Understanding -> Indexing
    Reuses unchanged content only when the processing signature also matches.
    Emits per-stage timings and throughput/memory/failure metrics to the log.
    """
    global indexing_status
    if not _lock_reserved and not _indexing_run_lock.acquire(blocking=False):
        logger.warning("Ignoring overlapping indexing request for %s; another run is active.", folder_path)
        return

    indexing_status.is_indexing = True
    indexing_status.current_stage = "Reading"
    indexing_status.total_files = 0
    indexing_status.processed_files = 0
    indexing_status.documents_count = 0
    indexing_status.images_count = 0
    indexing_status.other_count = 0
    indexing_status.error_message = None

    run_started = time.perf_counter()
    memory_start_mb = _process_memory_mb()
    metrics = {
        "scan_seconds": 0.0, "hash_seconds": 0.0, "extraction_seconds": 0.0,
        "image_extraction_seconds": 0.0, "decode_metadata_seconds": 0.0,
        "image_index_seconds": 0.0, "image_embedding_seconds": 0.0,
        "vision_preprocess_seconds": 0.0, "vision_seconds": 0.0, "ocr_seconds": 0.0,
        "pdf_text_extraction_seconds": 0.0, "pdf_render_seconds": 0.0, "pdf_ocr_seconds": 0.0,
        "embedding_seconds": 0.0, "database_seconds": 0.0,
        "indexed_images": 0, "skipped_images": 0, "failed_files": 0,
        "indexed_files": 0, "skipped_files": 0,
        "image_bytes": 0,
    }
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        source = LocalFileSource()
        stage_started = time.perf_counter()
        files = _discovered_files if _discovered_files is not None else source.scan(folder_path)
        metrics["scan_seconds"] += time.perf_counter() - stage_started
        indexing_status.total_files = len(files)

        for f in files:
            ftype = f["file_type"]
            if ftype in ("pdf", "docx", "pptx", "txt"):
                indexing_status.documents_count += 1
            elif ftype in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"):
                indexing_status.images_count += 1
            else:
                indexing_status.other_count += 1

        if settings.LOCAL_FAST_MODE:
            _index_folder_local_fast(db, files, metrics)
            indexing_status.current_stage = (
                "Error" if metrics["failed_files"] else
                "Awaiting confirmation" if metrics.get("deferred_files") else "Complete"
            )
            indexing_status.last_indexed_at = datetime.utcnow()
            indexing_status.is_indexing = False
            indexing_status.current_file = None
            return

        for idx, file_info in enumerate(files):
            filepath = file_info["filepath"]
            filename = file_info["filename"]
            file_type = file_info["file_type"]
            is_image = file_type in IMAGE_TYPES
            file_started = time.perf_counter()
            indexing_status.current_file = filename
            if is_image:
                metrics["image_bytes"] += file_info.get("file_size", 0)

            with file_index_lock(filepath):
                # Compute hash to detect if unchanged
                stage_started = time.perf_counter()
                current_hash = _compute_index_hash(filepath)
                existing_doc = _find_existing_document(db, filepath)
                try:
                    current_signature = _processing_signature(filepath, file_type)
                except Exception as error:
                    metrics["failed_files"] += 1
                    _record_index_error(filename, error)
                    indexing_status.processed_files = idx + 1
                    continue
                metrics["hash_seconds"] += time.perf_counter() - stage_started

                content_state_matches = existing_doc and _processing_state_matches(
                    existing_doc, file_type, current_signature
                )
                unchanged_file = (
                    content_state_matches and existing_doc.file_hash == current_hash
                    and current_hash and current_signature
                )
                if unchanged_file and _document_embeddings_match_configuration(existing_doc):
                    metrics["skipped_files"] += 1
                    if is_image:
                        metrics["skipped_images"] += 1
                    logger.info("Skipping unchanged document: %s (hash and processing state match)", filename)
                    indexing_status.processed_files = idx + 1
                    continue
                if unchanged_file:
                    try:
                        embedding_seconds, database_seconds = _refresh_document_embeddings(existing_doc, db)
                        metrics["embedding_seconds"] += embedding_seconds
                        metrics["database_seconds"] += database_seconds
                        metrics["indexed_files"] += 1
                        if is_image:
                            metrics["indexed_images"] += 1
                            metrics["image_index_seconds"] += time.perf_counter() - file_started
                        logger.info("Refreshed embeddings from stored text for unchanged file: %s", filename)
                    except Exception as refresh_error:
                        if isinstance(refresh_error, MediaConfirmationRequired):
                            from app.services.media_jobs import defer_media
                            defer_media(refresh_error.plan)
                            metrics["deferred_files"] = metrics.get("deferred_files", 0) + 1
                            indexing_status.processed_files = idx + 1
                            continue
                        metrics["failed_files"] += 1
                        logger.error("Failed to refresh embeddings for %s: %s", filepath, refresh_error, exc_info=True)
                        _record_index_error(filename, refresh_error)
                    indexing_status.processed_files = idx + 1
                    continue

                # Stage 2: Extracting
                indexing_status.current_stage = "Extracting"
                stage_started = time.perf_counter()
                try:
                    extracted_data = process_file(filepath)
                    extracted_text = extracted_data.get("extracted_text", "")
                    raw_chunks = extracted_data.get("chunks", [])
                except Exception as file_error:
                    metrics["extraction_seconds"] += time.perf_counter() - stage_started
                    if isinstance(file_error, MediaConfirmationRequired):
                        from app.services.media_jobs import defer_media
                        defer_media(file_error.plan)
                        metrics["deferred_files"] = metrics.get("deferred_files", 0) + 1
                        indexing_status.processed_files = idx + 1
                        continue
                    metrics["failed_files"] += 1
                    logger.error("Failed to process %s: %s", filepath, file_error, exc_info=True)
                    _record_index_error(filename, file_error)
                    indexing_status.processed_files = idx + 1
                    continue
                metrics["extraction_seconds"] += time.perf_counter() - stage_started
                if is_image:
                    metrics["image_extraction_seconds"] += time.perf_counter() - stage_started
                image_timings = extracted_data.get("timings", {})
                for metric_name in (
                    "decode_metadata_seconds", "vision_preprocess_seconds", "vision_seconds", "ocr_seconds",
                    "pdf_text_extraction_seconds", "pdf_render_seconds", "pdf_ocr_seconds",
                ):
                    metrics[metric_name] += float(image_timings.get(metric_name, 0.0) or 0.0)

                # Stage 3: Understanding
                indexing_status.current_stage = "Understanding"
                stage_started = time.perf_counter()
                try:
                    doc_embedding, doc_embedding_provenance = get_embedding_with_provenance(
                        extracted_text[:1500] if extracted_text else filename
                    )
                    entities_data = extract_entities_from_text(extracted_text)
                    if "entities" in extracted_data:
                        for ent in extracted_data["entities"]:
                            if ent not in entities_data:
                                entities_data.append(ent)

                    for chunk_info in raw_chunks:
                        chunk_text = chunk_info.get("content", "").strip()
                        if chunk_text:
                            chunk_info["_embedding"], chunk_info["_embedding_provenance"] = (
                                get_embedding_with_provenance(chunk_text)
                            )
                except Exception as embedding_error:
                    metrics["embedding_seconds"] += time.perf_counter() - stage_started
                    metrics["failed_files"] += 1
                    db.rollback()
                    logger.error("Failed to embed %s: %s", filepath, embedding_error, exc_info=True)
                    _record_index_error(filename, embedding_error)
                    indexing_status.processed_files = idx + 1
                    continue
                metrics["embedding_seconds"] += time.perf_counter() - stage_started

                # Stage 4: Indexing
                indexing_status.current_stage = "Indexing"
                stage_started = time.perf_counter()
                try:
                    if existing_doc:
                        db.delete(existing_doc)
                        db.flush()

                    doc = Document(
                        filename=filename,
                        filepath=str(Path(filepath).resolve()),
                        file_type=file_type,
                        file_size=file_info["file_size"],
                        file_hash=current_hash,
                        processing_signature=current_signature,
                        created_at=file_info["created_at"],
                        modified_at=file_info["modified_at"],
                        source="local",
                        extracted_text=extracted_text,
                        summary=extracted_text[:300] if extracted_text else "",
                        indexed_at=datetime.utcnow()
                    )
                    doc.set_embedding(doc_embedding)
                    doc.set_embedding_provenance(doc_embedding_provenance)
                    db.add(doc)
                    db.flush()

                    for chunk_idx, chunk_info in enumerate(raw_chunks):
                        chunk_text = chunk_info.get("content", "").strip()
                        if not chunk_text:
                            continue
                        chunk_model = Chunk(
                            document_id=doc.id,
                            chunk_index=chunk_idx,
                            content=chunk_text,
                            page_number=chunk_info.get("page_number"),
                            slide_number=chunk_info.get("slide_number")
                        )
                        chunk_model.set_embedding(chunk_info.pop("_embedding"))
                        chunk_model.set_embedding_provenance(chunk_info.pop("_embedding_provenance"))
                        db.add(chunk_model)

                    for ent in entities_data:
                        db.add(Entity(
                            document_id=doc.id,
                            entity_type=ent["entity_type"],
                            entity_value=ent["entity_value"]
                        ))

                    if file_type in MEDIA_TYPES:
                        from app.services.media_jobs import complete_media_request
                        complete_media_request(db, doc.filepath)
                    db.commit()
                    metrics["indexed_files"] += 1
                    if is_image:
                        metrics["indexed_images"] += 1
                        metrics["image_index_seconds"] += time.perf_counter() - file_started
                except Exception as index_error:
                    db.rollback()
                    metrics["failed_files"] += 1
                    logger.error("Failed to save %s: %s", filepath, index_error, exc_info=True)
                    _record_index_error(filename, index_error)
                finally:
                    metrics["database_seconds"] += time.perf_counter() - stage_started
                indexing_status.processed_files = idx + 1

        indexing_status.current_stage = (
            "Error" if metrics["failed_files"] else
            "Awaiting confirmation" if metrics.get("deferred_files") else "Complete"
        )
        indexing_status.last_indexed_at = datetime.utcnow()
        indexing_status.is_indexing = False
        indexing_status.current_file = None
        logger.info(
            "Indexing finished: %s files processed, %s failed.",
            len(files), metrics["failed_files"],
        )

    except Exception as e:
        metrics["failed_files"] += 1
        logger.error(f"Indexing failed: {e}", exc_info=True)
        _record_index_error(folder_path, e)
        indexing_status.is_indexing = False
        indexing_status.current_stage = "Error"
    finally:
        if db is not None:
            db.close()
        elapsed = max(time.perf_counter() - run_started, 1e-9)
        handled_images = metrics["indexed_images"] + metrics["skipped_images"]
        summary = {
            "event": "indexing_metrics",
            "files_seen": indexing_status.total_files,
            "images_indexed": metrics["indexed_images"],
            "images_skipped": metrics["skipped_images"],
            "failed_files": metrics["failed_files"],
            "files_indexed": metrics.get("indexed_files", 0),
            "files_skipped": metrics.get("skipped_files", 0),
            "image_bytes_scanned": metrics["image_bytes"],
            "files_per_second": round(metrics.get("indexed_files", 0) / elapsed, 3),
            "image_throughput_per_second": round(metrics["indexed_images"] / elapsed, 3),
            "image_handled_throughput_per_second": round(handled_images / elapsed, 3),
            "seconds_per_indexed_image": round(metrics["image_index_seconds"] / max(metrics["indexed_images"], 1), 3),
            "indexed_image_total_seconds": round(metrics["image_index_seconds"], 3),
            "elapsed_seconds": round(elapsed, 3),
            "stage_seconds": {
                key: round(metrics[key], 3)
                for key in (
                    "scan_seconds", "hash_seconds", "extraction_seconds", "image_embedding_seconds",
                    "embedding_seconds", "database_seconds",
                )
            },
            "image_substage_seconds": {
                "image_processing": round(metrics["image_extraction_seconds"], 3),
                "decode_and_metadata": round(metrics["decode_metadata_seconds"], 3),
                "vision_preprocessing": round(metrics["vision_preprocess_seconds"], 3),
                "vision_inference_or_cache": round(metrics["vision_seconds"], 3),
                "ocr_or_cache": round(metrics["ocr_seconds"], 3),
            },
            "pdf_substage_seconds": {
                "text_extraction": round(metrics["pdf_text_extraction_seconds"], 3),
                "page_rendering": round(metrics["pdf_render_seconds"], 3),
                "ocr": round(metrics["pdf_ocr_seconds"], 3),
            },
            "timing_notes": ["Image processing and its component timings overlap; do not sum image_substage_seconds fields."],
            "memory_start_mb": memory_start_mb,
            "memory_end_mb": _process_memory_mb(),
            "process_peak_memory_mb": _process_memory_mb(peak=True),
            "local_embedding_runtime": local_embedding_runtime_stats() if settings.LOCAL_FAST_MODE else None,
        }
        logger.info("Indexing metrics: %s", json.dumps(summary, sort_keys=True))
        _indexing_run_lock.release()

def index_single_file(filepath: str, db: Session) -> Document:
    """Index one file while coordinating with scans and uploads of that path."""
    with file_index_lock(filepath):
        return _index_single_file_locked(filepath, db)


def _index_single_file_locked(filepath: str, db: Session, force_reprocess: bool = False) -> Document:
    """
    Synchronously indexes a single uploaded or new file.
    Includes unchanged-file check: does not regenerate embeddings if unchanged.
    """
    total_started = time.perf_counter()
    path = Path(filepath)
    file_type = get_file_type(filepath)
    stat = path.stat()
    indexing_status.is_indexing = True
    indexing_status.current_file = path.name
    indexing_status.current_stage = "Reading"
    current_hash = _compute_index_hash(filepath)
    current_signature = _processing_signature(filepath, file_type)

    existing_doc = _find_existing_document(db, filepath)
    content_state_matches = existing_doc and _processing_state_matches(
        existing_doc, file_type, current_signature
    )
    if not force_reprocess and content_state_matches and existing_doc.file_hash == current_hash and current_hash and current_signature:
        if _document_embeddings_match_configuration(existing_doc):
            logger.info("File %s is unchanged. Returning existing indexed document.", path.name)
            return existing_doc
        indexing_status.current_stage = "Understanding"
        _refresh_document_embeddings(existing_doc, db)
        indexing_status.current_stage = "Indexing"
        logger.info("Refreshed embeddings from stored text for unchanged file %s.", path.name)
        return existing_doc

    indexing_status.current_stage = "Extracting"
    extract_started = time.perf_counter()
    extracted_data = process_file(filepath)
    extracted_text = extracted_data.get("extracted_text", "")
    raw_chunks = extracted_data.get("chunks", [])
    extraction_seconds = time.perf_counter() - extract_started

    indexing_status.current_stage = "Understanding"
    embedding_started = time.perf_counter()
    is_image = file_type in IMAGE_TYPES
    text_inputs = []
    doc_text_position = None
    if not (is_image and settings.LOCAL_FAST_MODE):
        doc_text_position = len(text_inputs)
        text_inputs.append(extracted_text[:1500] if extracted_text else path.name)
    chunk_positions = {}
    for chunk_index, chunk_data in enumerate(raw_chunks):
        chunk_text = chunk_data.get("content", "").strip()
        if chunk_text and chunk_data.get("_embedding") is None:
            chunk_positions[chunk_index] = len(text_inputs)
            text_inputs.append(chunk_text)
    text_embeddings = embed_texts_with_provenance(text_inputs, task="retrieval.passage")
    if is_image and settings.LOCAL_FAST_MODE:
        doc_embedding = extracted_data.get("image_embedding")
        doc_embedding_provenance = extracted_data.get("image_embedding_provenance")
        if not doc_embedding or not doc_embedding_provenance:
            raise RuntimeError("Local image embedding did not produce a vector.")
    else:
        doc_embedding, doc_embedding_provenance = text_embeddings[doc_text_position]
    chunk_embeddings = [
        (chunk.get("_embedding"), chunk.get("_embedding_provenance")) for chunk in raw_chunks
    ]
    for chunk_index, vector_position in chunk_positions.items():
        chunk_embeddings[chunk_index] = text_embeddings[vector_position]
    entities_data = extract_entities_from_text(extracted_text)
    if "entities" in extracted_data:
        for ent in extracted_data["entities"]:
            if ent not in entities_data:
                entities_data.append(ent)

    embedding_seconds = time.perf_counter() - embedding_started

    indexing_status.current_stage = "Indexing"
    db_started = time.perf_counter()
    try:
        doc = existing_doc
        if doc is None:
            doc = Document(filepath=str(path.resolve()))
            db.add(doc)
        else:
            doc.chunks.clear()
            doc.entities.clear()
            db.flush()
        doc.filename = path.name
        doc.filepath = str(path.resolve())
        doc.file_type = file_type
        doc.file_size = stat.st_size
        doc.file_hash = current_hash
        doc.processing_signature = current_signature
        doc.created_at = datetime.fromtimestamp(stat.st_ctime)
        doc.modified_at = datetime.fromtimestamp(stat.st_mtime)
        doc.source = "local"
        doc.extracted_text = extracted_text
        doc.summary = extracted_text[:300] if extracted_text else ""
        doc.indexed_at = datetime.utcnow()
        doc.set_embedding(doc_embedding)
        doc.set_embedding_provenance(doc_embedding_provenance)
        db.add(doc)
        db.flush()

        for chunk_idx, chunk_data in enumerate(raw_chunks):
            c_text = chunk_data.get("content", "").strip()
            if not c_text:
                continue
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=chunk_idx,
                content=c_text,
                page_number=chunk_data.get("page_number"),
                slide_number=chunk_data.get("slide_number")
            )
            chunk_embedding, chunk_provenance = chunk_embeddings[chunk_idx]
            chunk.set_embedding(chunk_embedding)
            chunk.set_embedding_provenance(chunk_provenance)
            db.add(chunk)

        for ent in entities_data:
            db.add(Entity(
                document_id=doc.id,
                entity_type=ent["entity_type"],
                entity_value=ent["entity_value"]
            ))

        if file_type in MEDIA_TYPES:
            from app.services.media_jobs import complete_media_request
            complete_media_request(db, doc.filepath)
        db.commit()
        db.refresh(doc)
    except Exception:
        db.rollback()
        raise

    logger.info(
        "Indexed %s: extraction=%.3fs embedding=%.3fs database=%.3fs total=%.3fs",
        path.name, extraction_seconds, embedding_seconds, time.perf_counter() - db_started,
        time.perf_counter() - total_started,
    )
    return doc
