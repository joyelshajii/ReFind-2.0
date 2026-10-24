"""Compare old per-row cosine retrieval with exact cached matrix retrieval.

Synthetic vectors measure retrieval overhead only, not model inference, OCR,
network calls, or real-world relevance. Uses an isolated temporary database.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from app.db.models import Document, Chunk
from app.services.embedding_service import cosine_similarity, embedding_signature_is_compatible
from app.services.search_index import get_search_index, install_search_revision


def run(count, chunks, queries, work_dir):
    rng = np.random.default_rng(42)
    provenance = {"provider": "embeddinggemma", "model": "benchmark-fixture",
                  "dimension": 768, "config_id": "fixture", "task": "retrieval.passage"}
    query_provenance = {**provenance, "task": "retrieval.query"}
    with tempfile.TemporaryDirectory(prefix="search-benchmark-", dir=work_dir) as directory:
        engine = create_engine(f"sqlite:///{Path(directory) / 'benchmark.db'}")
        sessions = sessionmaker(bind=engine)
        try:
            Base.metadata.create_all(engine)
            with engine.begin() as connection:
                install_search_revision(connection)
            engine._refind_search_revision = True
            with sessions() as db:
                for i in range(count):
                    doc = Document(filename=f"{i}.txt", filepath=f"fixture/{i}.txt", file_type="txt",
                                   extracted_text="Synthetic benchmark content.")
                    vector = rng.normal(size=768)
                    doc.set_embedding((vector / np.linalg.norm(vector)).tolist())
                    doc.set_embedding_provenance(provenance)
                    for j in range(chunks):
                        chunk = Chunk(chunk_index=j, content=f"Benchmark passage {j}.")
                        vector = rng.normal(size=768)
                        chunk.set_embedding((vector / np.linalg.norm(vector)).tolist())
                        chunk.set_embedding_provenance(provenance)
                        doc.chunks.append(chunk)
                    db.add(doc)
                db.commit()
            query_vectors = rng.normal(size=(queries, 768)).astype(np.float32)
            old_times, new_times, old_scores = [], [], []
            for query in query_vectors:
                with sessions() as db:
                    start = time.perf_counter()
                    scores = {}
                    for doc in db.query(Document).all():
                        best = 0.0
                        for row in [doc, *doc.chunks]:
                            if embedding_signature_is_compatible(row.get_embedding_provenance(), query_provenance):
                                best = max(best, cosine_similarity(query.tolist(), row.get_embedding()))
                        scores[doc.id] = best
                    old_times.append(time.perf_counter() - start)
                    old_scores.append(scores)
            with sessions() as db:
                start = time.perf_counter()
                snapshot = get_search_index(db, query_provenance)
                cold_seconds = time.perf_counter() - start
            maximum_error = 0.0
            for query, expected in zip(query_vectors, old_scores):
                with sessions() as db:
                    start = time.perf_counter()
                    actual = get_search_index(db, query_provenance).score(query)
                    new_times.append(time.perf_counter() - start)
                maximum_error = max(maximum_error, max(abs(expected[key] - actual[key][0]) for key in expected))
            return {
                "scope": "synthetic exact-vector retrieval; excludes model, OCR, filesystem and network latency",
                "documents": count, "chunks_per_document": chunks,
                "vectors": count * (chunks + 1), "queries": queries,
                "old_median_ms": round(statistics.median(old_times) * 1000, 3),
                "new_warm_median_ms": round(statistics.median(new_times) * 1000, 3),
                "new_cold_build_ms": round(cold_seconds * 1000, 3),
                "speedup": round(statistics.median(old_times) / statistics.median(new_times), 2),
                "maximum_score_error": float(maximum_error),
                "cache_estimated_mb": round(snapshot.bytes_used / 1024**2, 2),
            }
        finally:
            engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=400)
    parser.add_argument("--chunks", type=int, default=5)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.documents < 1 or args.chunks < 0 or args.queries < 1:
        parser.error("documents/queries must be positive and chunks non-negative")
    report = run(args.documents, args.chunks, args.queries, args.work_dir)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
