import os
import re
import hashlib
import logging
import json
import time
import sys
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.base import SessionLocal
from app.db.models import Document, Chunk, Entity
from app.schemas.index import IndexProgressStatus
from app.processors.factory import process_file, get_file_type
from app.connectors.local_source import LocalFileSource
from app.services.embedding_service import get_embedding
from app.services.query_parser import COMMON_TECH, COMMON_TOPICS
from app.processors.image_processor import IMAGE_PROCESSOR_VERSION
from app.core.config import settings

logger = logging.getLogger("refind.services.indexer")

indexing_status = IndexProgressStatus()
_indexing_run_lock = threading.Lock()


def try_reserve_indexing_run() -> bool:
    """Atomically reserve the single background indexing worker slot."""
    if not _indexing_run_lock.acquire(blocking=False):
        return False
    indexing_status.is_indexing = True
    return True

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}
PIPELINE_VERSION = 3


def _processing_signature(filepath: str, file_type: str) -> str:
    """Versioned extraction/embedding config used to validate cached state."""
    signature = {
        "file_type": file_type,
        "embedding_provider": settings.EMBEDDING_PROVIDER.lower(),
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "gemini_embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        "jina_embedding_model": settings.JINA_EMBEDDING_MODEL,
        "jina_embedding_dimension": settings.JINA_EMBEDDING_DIMENSION,
    }
    if file_type in IMAGE_TYPES:
        signature.update({
            "pipeline": PIPELINE_VERSION,
            "image_processor": IMAGE_PROCESSOR_VERSION,
            "vision_enabled": bool(settings.IMAGE_VISION_ENABLED and settings.GEMINI_API_KEY),
            "vision_model": settings.GEMINI_MODEL,
            "local_ocr": "enabled-v1",
        })
        sidecar = Path(f"{filepath}.meta.json")
        if sidecar.exists():
            sidecar_hash = compute_file_hash(str(sidecar))
            if not sidecar_hash:
                return ""
            signature["sidecar_sha256"] = sidecar_hash
    else:
        signature["chunk_size"] = settings.CHUNK_SIZE
        signature["chunk_overlap"] = settings.CHUNK_OVERLAP
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _process_memory_mb(peak: bool = False) -> Optional[float]:
    """Return process working-set memory where supported, without a hard dependency."""
    if not peak:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
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
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Error computing hash for {filepath}: {e}")
        return ""

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

def index_folder_background(folder_path: str, _lock_reserved: bool = False):
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
        "image_index_seconds": 0.0,
        "vision_preprocess_seconds": 0.0, "vision_seconds": 0.0, "ocr_seconds": 0.0,
        "embedding_seconds": 0.0, "database_seconds": 0.0,
        "indexed_images": 0, "skipped_images": 0, "failed_files": 0,
        "image_bytes": 0,
    }
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        source = LocalFileSource()
        stage_started = time.perf_counter()
        files = source.scan(folder_path)
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

        for idx, file_info in enumerate(files):
            filepath = file_info["filepath"]
            filename = file_info["filename"]
            file_type = file_info["file_type"]
            is_image = file_type in IMAGE_TYPES
            file_started = time.perf_counter()
            indexing_status.current_file = filename
            if is_image:
                metrics["image_bytes"] += file_info.get("file_size", 0)

            # Compute hash to detect if unchanged
            stage_started = time.perf_counter()
            current_hash = compute_file_hash(filepath)
            existing_doc = db.query(Document).filter(Document.filepath == filepath).first()
            current_signature = _processing_signature(filepath, file_type)
            metrics["hash_seconds"] += time.perf_counter() - stage_started

            state_matches = existing_doc and (
                existing_doc.processing_signature == current_signature
                or (file_type not in IMAGE_TYPES and not existing_doc.processing_signature)
            )
            if (state_matches and existing_doc.file_hash == current_hash and current_hash and current_signature):
                if is_image:
                    metrics["skipped_images"] += 1
                logger.info("Skipping unchanged document: %s (hash and processing signature match)", filename)
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
                metrics["failed_files"] += 1
                logger.error("Failed to process %s: %s", filepath, file_error, exc_info=True)
                indexing_status.error_message = f"{filename}: {file_error}"
                indexing_status.processed_files = idx + 1
                continue
            metrics["extraction_seconds"] += time.perf_counter() - stage_started
            if is_image:
                metrics["image_extraction_seconds"] += time.perf_counter() - stage_started
            image_timings = extracted_data.get("timings", {})
            for metric_name in ("decode_metadata_seconds", "vision_preprocess_seconds", "vision_seconds", "ocr_seconds"):
                metrics[metric_name] += float(image_timings.get(metric_name, 0.0) or 0.0)

            # Stage 3: Understanding
            indexing_status.current_stage = "Understanding"
            stage_started = time.perf_counter()
            try:
                doc_embedding = get_embedding(extracted_text[:1500] if extracted_text else filename)
                entities_data = extract_entities_from_text(extracted_text)
                if "entities" in extracted_data:
                    for ent in extracted_data["entities"]:
                        if ent not in entities_data:
                            entities_data.append(ent)

                for chunk_info in raw_chunks:
                    chunk_text = chunk_info.get("content", "").strip()
                    if chunk_text:
                        chunk_info["_embedding"] = get_embedding(chunk_text)
            except Exception as embedding_error:
                metrics["embedding_seconds"] += time.perf_counter() - stage_started
                metrics["failed_files"] += 1
                db.rollback()
                logger.error("Failed to embed %s: %s", filepath, embedding_error, exc_info=True)
                indexing_status.error_message = f"{filename}: {embedding_error}"
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
                    filepath=filepath,
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
                    db.add(chunk_model)

                for ent in entities_data:
                    db.add(Entity(
                        document_id=doc.id,
                        entity_type=ent["entity_type"],
                        entity_value=ent["entity_value"]
                    ))

                db.commit()
                if is_image:
                    metrics["indexed_images"] += 1
                    metrics["image_index_seconds"] += time.perf_counter() - file_started
            except Exception as index_error:
                db.rollback()
                metrics["failed_files"] += 1
                logger.error("Failed to save %s: %s", filepath, index_error, exc_info=True)
                indexing_status.error_message = f"{filename}: {index_error}"
            finally:
                metrics["database_seconds"] += time.perf_counter() - stage_started
            indexing_status.processed_files = idx + 1

        indexing_status.current_stage = "Error" if metrics["failed_files"] else "Complete"
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
        indexing_status.error_message = str(e)
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
            "image_bytes": metrics["image_bytes"],
            "image_throughput_per_second": round(metrics["indexed_images"] / elapsed, 3),
            "image_handled_throughput_per_second": round(handled_images / elapsed, 3),
            "seconds_per_indexed_image": round(metrics["image_index_seconds"] / max(metrics["indexed_images"], 1), 3),
            "elapsed_seconds": round(elapsed, 3),
            "stage_seconds": {key: round(value, 3) for key, value in metrics.items() if key.endswith("_seconds")},
            "memory_start_mb": memory_start_mb,
            "memory_end_mb": _process_memory_mb(),
            "process_peak_memory_mb": _process_memory_mb(peak=True),
        }
        logger.info("Indexing metrics: %s", json.dumps(summary, sort_keys=True))
        _indexing_run_lock.release()

def index_single_file(filepath: str, db: Session) -> Document:
    """
    Synchronously indexes a single uploaded or new file.
    Includes unchanged-file check: does not regenerate embeddings if unchanged.
    """
    total_started = time.perf_counter()
    path = Path(filepath)
    file_type = get_file_type(filepath)
    stat = path.stat()
    current_hash = compute_file_hash(filepath)
    current_signature = _processing_signature(filepath, file_type)

    existing_doc = db.query(Document).filter(Document.filepath == str(path.resolve())).first()
    state_matches = existing_doc and (
        existing_doc.processing_signature == current_signature
        or (file_type not in IMAGE_TYPES and not existing_doc.processing_signature)
    )
    if state_matches and existing_doc.file_hash == current_hash and current_hash and current_signature:
        logger.info("File %s is unchanged. Returning existing indexed document.", path.name)
        return existing_doc

    extract_started = time.perf_counter()
    extracted_data = process_file(filepath)
    extracted_text = extracted_data.get("extracted_text", "")
    raw_chunks = extracted_data.get("chunks", [])
    extraction_seconds = time.perf_counter() - extract_started

    embedding_started = time.perf_counter()
    doc_embedding = get_embedding(extracted_text[:1500] if extracted_text else path.name)
    chunk_embeddings = []
    for chunk_data in raw_chunks:
        chunk_text = chunk_data.get("content", "").strip()
        chunk_embeddings.append(get_embedding(chunk_text) if chunk_text else None)
    entities_data = extract_entities_from_text(extracted_text)
    if "entities" in extracted_data:
        for ent in extracted_data["entities"]:
            if ent not in entities_data:
                entities_data.append(ent)

    embedding_seconds = time.perf_counter() - embedding_started

    db_started = time.perf_counter()
    try:
        if existing_doc:
            db.delete(existing_doc)
            db.flush()

        doc = Document(
            filename=path.name,
            filepath=str(path.resolve()),
            file_type=file_type,
            file_size=stat.st_size,
            file_hash=current_hash,
            processing_signature=current_signature,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            source="local",
            extracted_text=extracted_text,
            summary=extracted_text[:300] if extracted_text else "",
            indexed_at=datetime.utcnow()
        )
        doc.set_embedding(doc_embedding)
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
            chunk.set_embedding(chunk_embeddings[chunk_idx])
            db.add(chunk)

        for ent in entities_data:
            db.add(Entity(
                document_id=doc.id,
                entity_type=ent["entity_type"],
                entity_value=ent["entity_value"]
            ))

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
