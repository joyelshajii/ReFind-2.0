"""Remove indexed data and app-owned upload copies without deleting source files."""
from contextlib import contextmanager
import os
from pathlib import Path

from app.core.config import settings
from app.db.models import Document, Chunk, Entity, MediaIndexJob
from app.services import indexer
from app.services.media_preflight import companion_files, clear_media_cache


@contextmanager
def removal_slot():
    if not indexer.try_reserve_indexing_run():
        raise RuntimeError("Wait for active indexing to finish before removing files.")
    try:
        yield
    finally:
        indexer.release_indexing_run()


def upload_root():
    root = Path(settings.UPLOAD_DIR).absolute()
    # Never follow a redirected upload directory into another location.
    if os.path.normcase(str(root.resolve())) != os.path.normcase(str(root)):
        raise RuntimeError("Upload storage is redirected. Saved-copy deletion requires a regular upload directory.")
    return root.resolve()


def is_upload_copy(path, root):
    path = Path(path).absolute()
    return path.parent == root


def delete_copies(paths, root):
    # Validate the complete set before deleting anything. Uploads are flat;
    # directory recursion and source paths outside storage are never allowed.
    paths = sorted(set(Path(path).absolute() for path in paths))
    for path in paths:
        if path.parent != root or path.resolve().parent != root or path.is_symlink():
            raise RuntimeError("Saved-copy deletion refused a path outside upload storage.")
        if path.exists() and not path.is_file():
            raise RuntimeError("Saved-copy deletion only supports regular files.")
    removed = 0
    for path in paths:
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def clear_index(db):
    with removal_slot():
        root = upload_root()
        copies = [path for path in root.iterdir() if path.is_file() or path.is_symlink()] if root.exists() else []
        removed_count = db.query(Document).count()
        copies_deleted = delete_copies(copies, root)
        db.query(MediaIndexJob).delete()
        db.query(Entity).delete()
        db.query(Chunk).delete()
        db.query(Document).delete()
        db.commit()
        clear_media_cache()
        status = indexer.indexing_status
        status.total_files = status.processed_files = 0
        status.documents_count = status.images_count = status.other_count = 0
        status.current_file = status.error_message = status.last_indexed_at = None
        status.current_stage = "Idle"
        return removed_count, copies_deleted


def remove_document(db, document_id):
    with removal_slot():
        doc = db.get(Document, document_id)
        if doc is None:
            raise LookupError("Document not found.")
        filename, filepath = doc.filename, doc.filepath
        root = upload_root()
        copies = set()
        if is_upload_copy(filepath, root):
            path = Path(filepath).absolute()
            copies.add(path)
            companions = set(companion_files(path)) if root.exists() else set()
            # Keep companions shared by another saved recording.
            from app.processors.factory import SUPPORTED_EXTENSIONS, MEDIA_TYPES
            for other in root.iterdir() if root.exists() else []:
                if other != path and other.is_file() and SUPPORTED_EXTENSIONS.get(other.suffix.lower()) in MEDIA_TYPES:
                    companions.difference_update(companion_files(other))
            copies.update(companions)
        copies_deleted = delete_copies(copies, root)
        removed_paths = {filepath, *(str(path) for path in copies)}
        db.query(MediaIndexJob).filter(MediaIndexJob.filepath.in_(removed_paths)).delete(synchronize_session=False)
        # A saved companion may also have been indexed as a TXT document.
        for related in db.query(Document).filter(Document.filepath.in_(removed_paths)).all():
            db.delete(related)
        db.commit()
        clear_media_cache()
        status = indexer.indexing_status
        counts = [row[0] for row in db.query(Document.file_type).all()]
        status.total_files = status.processed_files = len(counts)
        status.documents_count = sum(kind in {"pdf", "docx", "pptx", "txt"} for kind in counts)
        status.images_count = sum(kind in indexer.IMAGE_TYPES for kind in counts)
        status.other_count = len(counts) - status.documents_count - status.images_count
        return filename, copies_deleted
