"""Two-pass upload ingestion: save transcripts before inspecting their media."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

from app.core.config import settings
from app.processors.factory import SUPPORTED_EXTENSIONS
from app.services import indexer
from app.services.media_preflight import MediaConfirmationRequired, TEXT_SUFFIXES
from app.services.media_jobs import defer_media


def save_and_index_uploads(files, db):
    indexed, failed, deferred, saved = [], [], [], []
    companions = 0
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    status = indexer.indexing_status
    status.total_files, status.processed_files = len(files), 0
    status.current_stage, status.error_message = "Reading", None
    for file in files:
        name = Path((file.filename or "").replace("\\", "/")).name
        staged = None
        status.current_file = name
        try:
            suffix = Path(name).suffix.lower()
            is_companion = suffix in set(TEXT_SUFFIXES) - {".txt"} or name.lower().endswith(".meta.json")
            if not name or name in {".", ".."} or (suffix not in SUPPORTED_EXTENSIONS and not is_companion):
                raise ValueError("Unsupported file. Choose media, documents, or matching transcript/subtitle files.")
            path = upload_dir / name
            with indexer.file_index_lock(str(path.resolve())):
                with tempfile.NamedTemporaryFile(mode="wb", dir=upload_dir, prefix=".refind-upload-", suffix=".upload", delete=False) as buffer:
                    staged = Path(buffer.name)
                    shutil.copyfileobj(file.file, buffer)
                os.replace(staged, path)
                staged = None
            if is_companion:
                companions += 1
            else:
                saved.append(path)
        except Exception as error:
            failed.append({"filename": name or "(unnamed file)", "error": indexer._display_index_error(error)})
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    indexer.logger.warning("Unable to remove staged upload %s", staged)
            try:
                file.file.close()
            except OSError:
                indexer.logger.warning("Unable to close uploaded file %s", name)

    # Only explicitly uploaded recordings/documents are indexed. A companion
    # upload must never pull older recordings into this request.
    # All sibling transcript files are now present, regardless of selection order.
    status.total_files = len(saved)
    for position, path in enumerate(saved):
        status.current_file = path.name
        status.current_stage = "Checking media"
        try:
            doc = indexer.index_single_file(str(path.resolve()), db)
            indexed.append({"id": doc.id, "filename": doc.filename,
                            "file_type": doc.file_type, "file_size": doc.file_size})
        except MediaConfirmationRequired as error:
            db.rollback()
            deferred.append(defer_media(error.plan))
        except Exception as error:
            db.rollback()
            indexer.logger.exception("Failed to index upload %s", path.name)
            failed.append({"filename": path.name, "error": indexer._display_index_error(error)})
        finally:
            status.processed_files = position + 1
    status.current_file = None
    status.error_message = f"{len(failed)} file(s) failed." if failed else None
    status.current_stage = "Awaiting confirmation" if deferred else "Error" if failed else "Complete"
    if indexed:
        status.last_indexed_at = datetime.utcnow()
    parts = [f"Indexed {len(indexed)} file(s)."]
    if deferred:
        parts.append(f"{len(deferred)} long recording(s) need your choice below before full processing.")
    if companions:
        parts.append(f"Saved {companions} transcript/subtitle file(s).")
    if failed:
        parts.append(f"{len(failed)} file(s) failed.")
    return {"status": "awaiting_confirmation" if deferred else "partial" if failed and indexed else "failed" if failed else "success",
            "message": " ".join(parts), "files": indexed, "failed_files": failed,
            "deferred_files": deferred, "companions_saved": companions}
