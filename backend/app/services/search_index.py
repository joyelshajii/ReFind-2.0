"""Exact, vectorized retrieval with a transaction-versioned SQLite cache.

Keep JSON vectors on disk for compatibility; decode/normalize them once per
index revision. SQL triggers invalidate snapshots even for external writers.
No approximate retrieval or vector quantization is used.
"""
from dataclasses import dataclass
from datetime import datetime
import json
import threading
from types import MappingProxyType
from weakref import WeakKeyDictionary

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.models import Document
from app.services.embedding_service import embedding_signature_is_compatible

_cache = WeakKeyDictionary()
_cache_lock = threading.RLock()


def install_search_revision(connection):
    """Additive migration; revisions change atomically with document/chunk writes."""
    if connection.dialect.name != "sqlite":
        return
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS search_revision "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
    ))
    connection.execute(text("INSERT OR IGNORE INTO search_revision VALUES (1, 0)"))
    for table in ("documents", "chunks"):
        for operation in ("INSERT", "UPDATE", "DELETE"):
            connection.execute(text(
                f"CREATE TRIGGER IF NOT EXISTS search_revision_{table}_{operation.lower()} "
                f"AFTER {operation} ON {table} BEGIN "
                "UPDATE search_revision SET version = version + 1 WHERE id = 1; END"
            ))


@dataclass(frozen=True)
class SearchChunk:
    content: str
    page_number: int | None
    slide_number: int | None


@dataclass(frozen=True)
class SearchDocument:
    id: str
    filename: str
    filepath: str
    file_type: str
    file_size: int
    modified_at: datetime | None
    created_at: datetime | None
    extracted_text: str
    lower_text: str
    chunks: tuple
    provenance: object

    def get_embedding_provenance(self):
        return dict(self.provenance)


@dataclass(frozen=True)
class SearchIndex:
    documents: tuple
    vectors: np.ndarray
    owners: np.ndarray
    locations: tuple
    bytes_used: int

    def score(self, query):
        """Return each document's exact cosine maximum and its best passage."""
        matches = {doc.id: (0.0, None, "") for doc in self.documents}
        vector = np.asarray(query, dtype=np.float32)
        if (not len(self.vectors) or vector.shape != (self.vectors.shape[1],)
                or not np.isfinite(vector).all()):
            return matches
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return matches
        scores = np.clip(self.vectors @ (vector / norm), 0.0, 1.0)
        best = np.zeros(len(self.documents), dtype=np.float32)
        np.maximum.at(best, self.owners, scores)
        # First winning row preserves document-before-chunk tie behavior.
        winning = np.flatnonzero((scores > 0) & (scores == best[self.owners]))
        seen = set()
        for row in winning:
            owner = int(self.owners[row])
            if owner in seen:
                continue
            seen.add(owner)
            location, content = self.locations[row]
            matches[self.documents[owner].id] = (float(scores[row]), location, content)
        return matches


def _build_index(db, provenance):
    documents, vectors, owners, locations = [], [], [], []
    dimension = int(provenance["dimension"])
    bytes_used = 0

    def add_vector(row, owner, location=None, content=""):
        if not embedding_signature_is_compatible(row.get_embedding_provenance(), provenance):
            return
        try:
            vector = np.asarray(row.get_embedding(), dtype=np.float32)
            if vector.shape != (dimension,) or not np.isfinite(vector).all():
                return
            norm = np.linalg.norm(vector)
            if norm < 1e-6:
                return
            vectors.append(vector / norm)
            owners.append(owner)
            locations.append((location, content))
        except (TypeError, ValueError):
            return

    # selectinload avoids a separate SQL query for every document's chunks.
    rows = db.query(Document).options(selectinload(Document.chunks)).populate_existing().all()
    for owner, row in enumerate(rows):
        chunks = tuple(SearchChunk(ch.content, ch.page_number, ch.slide_number)
                       for ch in sorted(row.chunks, key=lambda ch: ch.chunk_index))
        body = row.extracted_text or ""
        documents.append(SearchDocument(
            row.id, row.filename, row.filepath, row.file_type, row.file_size or 0,
            row.modified_at, row.created_at, body,
            (body + "\n" + "\n".join(chunk.content for chunk in chunks)).lower(), chunks,
            MappingProxyType(row.get_embedding_provenance() or {}),
        ))
        # Conservative UTF-32 estimate plus Python object overhead.
        bytes_used += 8 * (len(body) + sum(len(ch.content) for ch in chunks)) + 2048 + 256 * len(chunks)
        add_vector(row, owner)
        for chunk in sorted(row.chunks, key=lambda ch: ch.chunk_index):
            location = (f"Page {chunk.page_number}" if chunk.page_number
                        else f"Slide {chunk.slide_number}" if chunk.slide_number else None)
            chunk_provenance = chunk.get_embedding_provenance() or {}
            if chunk_provenance.get("task") in {"retrieval.audio", "retrieval.video"}:
                from app.services.audio_service import timestamp
                label = "Audio" if chunk_provenance["task"] == "retrieval.audio" else "Video"
                location = (
                    f"{label} {timestamp(chunk_provenance.get('start_seconds', 0))}"
                    f"–{timestamp(chunk_provenance.get('end_seconds', 0))}"
                )
            add_vector(chunk, owner, location, chunk.content)
    matrix = np.stack(vectors) if vectors else np.empty((0, dimension), dtype=np.float32)
    owner_array = np.asarray(owners, dtype=np.intp)
    matrix.flags.writeable = owner_array.flags.writeable = False
    bytes_used += matrix.nbytes + owner_array.nbytes + 128 * len(locations)
    return SearchIndex(tuple(documents), matrix, owner_array, tuple(locations), bytes_used)


def get_search_index(db, provenance):
    engine = db.get_bind()
    if db.new or db.dirty or db.deleted:
        db.flush()
    # Unit-test/custom databases and PostgreSQL use the same exact scorer with
    # fresh snapshots unless the revision migration has been installed.
    cacheable = (getattr(engine, "_refind_search_revision", False)
                 and not (db.new or db.dirty or db.deleted))
    if cacheable and db.connection().connection.driver_connection.in_transaction:
        # Never publish a snapshot containing uncommitted writes, including
        # already-flushed ORM changes and raw SQL followed by rollback.
        cacheable = False
    if not cacheable:
        return _build_index(db, provenance)
    config = json.dumps(provenance, sort_keys=True)
    with _cache_lock:
        revision = db.execute(text("SELECT version FROM search_revision WHERE id = 1")).scalar_one()
        previous = _cache.get(engine)
        if previous and previous[:2] == (revision, config):
            return previous[2]
        snapshot = _build_index(db, provenance)
        after = db.execute(text("SELECT version FROM search_revision WHERE id = 1")).scalar_one()
        if revision == after and snapshot.bytes_used <= max(0, settings.SEARCH_VECTOR_CACHE_MB) * 1024**2:
            _cache[engine] = (revision, config, snapshot)
        else:
            _cache.pop(engine, None)
        return snapshot
