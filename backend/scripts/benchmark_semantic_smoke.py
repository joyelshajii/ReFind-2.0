"""Small real EmbeddingGemma paraphrase check; not a representative recall evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db.base import Base
from app.db.models import Document, Chunk
from app.services.embedding_service import embed_texts_with_provenance, local_embedding_runtime_stats
from app.services.hybrid_search import search_documents
from app.services.query_parser import parse_query
from app.services.search_index import install_search_revision, get_search_index


CASES = [
    ("doctor", "A physician diagnoses ailments and prescribes medication to patients.", "doctor treating illness"),
    ("flight", "Airline reservations, departure times and baggage allowances for international passengers.", "booking a plane journey"),
    ("cat", "Felines purr and groom their fur with their tongues.", "cats cleaning themselves"),
    ("astronomy", "The solar system contains eight planets orbiting a star.", "worlds travelling around the sun"),
    ("budget", "Monthly expenses, income, savings targets and household financial planning.", "spending plan for personal finances"),
    ("car", "Automobiles consume gasoline; engines propel them along highways.", "vehicles driving on roads"),
]


def main():
    settings.LOCAL_FAST_MODE = True
    settings.GEMINI_API_KEY = settings.JINA_API_KEY = ""
    settings.SEARCH_INLINE_EXPLANATIONS = False
    with tempfile.TemporaryDirectory(prefix="refind-semantic-", dir=Path(__file__).resolve().parents[1] / ".venv") as folder:
        engine = create_engine(f"sqlite:///{Path(folder) / 'smoke.db'}")
        sessions = sessionmaker(bind=engine)
        try:
            Base.metadata.create_all(engine)
            with engine.begin() as connection:
                install_search_revision(connection)
            engine._refind_search_revision = True
            started = time.perf_counter()
            vectors = embed_texts_with_provenance([case[1] for case in CASES])
            setup_seconds = time.perf_counter() - started
            with sessions() as db:
                for (name, body, _), (vector, provenance) in zip(CASES, vectors):
                    path = Path(folder) / f"{name}.txt"
                    path.write_text(body)
                    doc = Document(filename=path.name, filepath=str(path), file_type="txt", extracted_text=body)
                    doc.set_embedding(vector)
                    doc.set_embedding_provenance(provenance)
                    chunk = Chunk(chunk_index=0, content=body)
                    chunk.set_embedding(vector)
                    chunk.set_embedding_provenance(provenance)
                    doc.chunks.append(chunk)
                    db.add(doc)
                db.commit()
            results = []
            for name, _, query in CASES:
                with sessions() as db:
                    started = time.perf_counter()
                    found = search_documents(query, parse_query(query), db)
                    elapsed = (time.perf_counter() - started) * 1000
                    vector, provenance = embed_texts_with_provenance([query], task="retrieval.query")[0]
                    index = get_search_index(db, provenance)
                    scores = index.score(vector)
                    expected = next(doc for doc in index.documents if doc.filename == f"{name}.txt")
                    results.append({"query": query, "expected": expected.filename,
                                    "top": found[0].filename if found else None,
                                    "expected_semantic_score": scores[expected.id][0],
                                    "milliseconds": round(elapsed, 2)})
            with sessions() as db:
                literal = 'Find the file containing the word nonexistenttoken'
                absent = search_documents(literal, parse_query(literal), db)
            print(json.dumps({"scope": "six synthetic paraphrases using real cached model; remote providers disabled",
                              "model": local_embedding_runtime_stats(), "setup_seconds": setup_seconds,
                              "top1_correct": sum(row["top"] == row["expected"] for row in results),
                              "cases": results, "absent_literal_has_no_results": not absent}, indent=2))
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
