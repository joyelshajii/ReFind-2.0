import re
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from app.db.models import Document, Chunk, Entity
from app.schemas.search import QueryClues, SearchResultItem, EvidenceItem
from app.services.embedding_service import (
    get_embedding_with_provenance, cosine_similarity, embedding_signature_is_compatible,
)
from app.services.search_index import get_search_index
from app.services.reranker_service import reranker_service
from app.services.gemini_service import gemini_service
from app.core.config import settings

logger = logging.getLogger("refind.services.hybrid_search")

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"}
AUDIO_TYPES = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "wma", "opus"}
VIDEO_TYPES = {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "m4v"}

def _extract_snippet(text: str, keywords: List[str], max_len: int = 220) -> str:
    """
    Finds the most relevant snippet of text containing search keywords.
    """
    if not text:
        return "No text available."

    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean

    lower = clean.lower()
    best_pos = -1
    for kw in keywords:
        pos = lower.find(kw.lower())
        if pos != -1:
            best_pos = pos
            break

    if best_pos == -1:
        return clean[:max_len] + "..."

    start = max(0, best_pos - 60)
    end = min(len(clean), start + max_len)
    
    if start > 0:
        space_before = clean.rfind(" ", 0, start)
        if space_before != -1:
            start = space_before + 1
    if end < len(clean):
        space_after = clean.find(" ", end)
        if space_after != -1:
            end = space_after

    snippet = clean[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(clean):
        snippet = snippet + "..."
    return snippet

def search_documents(
    query: str,
    clues: QueryClues,
    db: Session,
    top_k: int = 10
) -> List[SearchResultItem]:
    """
    Full AI-Powered Hybrid Search Pipeline:
    1. Query Understanding (provided via clues)
    2. Candidate Retrieval:
       - Exact cached matrix search over compatible EmbeddingGemma vectors
       - Keyword search over indexed text
       - Filename & Metadata matching
    3. Merge candidates and deduplicate (up to RETRIEVAL_CANDIDATE_LIMIT)
    4. Compute initial 5-signal scores (semantic, keyword, filename, metadata)
    5. Jina Reranker across candidates (falls back gracefully to hybrid scores if unavailable)
    6. Optional inline explanations (normally generated in the context view)
    7. Return top ranked results
    """
    started = time.perf_counter()
    if db.query(Document.id).first() is None:
        return []
    query_vec, query_provenance = get_embedding_with_provenance(query, task="retrieval.query")
    embedded_at = time.perf_counter()
    index = get_search_index(db, query_provenance)
    semantic_matches = index.score(query_vec)
    documents = []
    seen_paths = set()
    for doc in index.documents:
        try:
            path = Path(doc.filepath)
            if not path.is_file():
                continue
            canon = str(path.resolve()).lower()
        except OSError:
            continue
        if canon not in seen_paths:
            seen_paths.add(canon)
            documents.append(doc)
    retrieved_at = time.perf_counter()
    if not documents:
        return []

    # Candidate Retrieval Pool (Deduplicated by document id)
    candidate_map: Dict[str, Document] = {}

    # Keyword tokens
    search_keywords = list(dict.fromkeys(clues.keywords + (clues.topics or [])))
    keyword_patterns = []
    for keyword in search_keywords:
        stem = keyword.lower()
        if len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss"):
            stem = stem[:-1]
        keyword_patterns.append(re.compile(r"(?<!\w)" + re.escape(stem) + r"s?(?!\w)"))

    # Semantic candidate retrieval
    scored_by_vec = sorted(
        ((semantic_matches[doc.id][0], doc) for doc in documents),
        key=lambda pair: pair[0], reverse=True,
    )
    semantic_limit = max(top_k, settings.RETRIEVAL_CANDIDATE_LIMIT)
    for score, doc in scored_by_vec[:semantic_limit]:
        candidate_map[doc.id] = doc

    # Keyword / Filename candidate retrieval
    for doc in documents:
        doc_lower = doc.lower_text
        fn_lower = doc.filename.lower()
        if any(kw.lower() in fn_lower or kw.lower() in doc_lower for kw in search_keywords):
            candidate_map[doc.id] = doc
        if clues.file_type and (
            clues.file_type.lower() in {
                doc.file_type.lower(), Path(doc.filepath).suffix.lstrip(".").lower()
            }
            or (clues.file_type.lower() == "image" and doc.file_type.lower() in IMAGE_TYPES)
            or (clues.file_type.lower() == "audio" and doc.file_type.lower() in AUDIO_TYPES)
            or (clues.file_type.lower() == "video" and doc.file_type.lower() in VIDEO_TYPES)
        ):
            candidate_map[doc.id] = doc

    # If candidates are few, include all available documents up to candidate limit
    candidate_docs = list(candidate_map.values())
    if len(candidate_docs) < settings.RETRIEVAL_CANDIDATE_LIMIT:
        for doc in documents:
            if doc.id not in candidate_map:
                candidate_docs.append(doc)
            if len(candidate_docs) >= settings.RETRIEVAL_CANDIDATE_LIMIT:
                break

    # Month mapping for date matching
    month_names = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    scored_candidates: List[Dict[str, Any]] = []
    literal_query = bool(re.search(r'\b(?:contain(?:s|ing)?|exact(?:ly)?|word|name|named)\b|["“”]', query, re.I))
    quoted_terms = re.findall(r'["“]([^"”]+)["”]', query)
    neural_query = query_provenance.get("provider") in {"embeddinggemma", "gemini", "jina"}


    for doc in candidate_docs:
        doc_text = doc.extracted_text or ""
        doc_lower = doc.lower_text
        filename_lower = doc.filename.lower()

        # Reuse the exact vectorized retrieval score and winning passage.
        best_sem, best_chunk_loc, best_chunk_text = semantic_matches[doc.id]

        # 2. Keyword Relevance (supporting singular/plural and stem matching)
        kw_matches = 0
        total_kws = len(search_keywords)
        if total_kws > 0:
            for pattern in keyword_patterns:
                # A filename hit contributes through fn_score above; it cannot
                # satisfy the separate requirement for a content keyword hit.
                if pattern.search(doc_lower):
                    kw_matches += 1
            kw_score = kw_matches / total_kws
        else:
            kw_score = 0.5

        # 3. Filename Relevance
        fn_score = 0.0
        if search_keywords:
            fn_matches = sum(1 for kw in search_keywords if kw.lower().rstrip("s") in filename_lower)
            fn_score = min(1.0, fn_matches / max(1, len(search_keywords)))

        # 4. Metadata Relevance (file_type and date/time)
        meta_score = 0.0
        meta_checks = 0

        matching_clues: List[str] = []
        detected_entities: List[str] = []
        evidence_list: List[EvidenceItem] = []

        is_image = (
            doc.file_type.lower() in IMAGE_TYPES
            or Path(doc.filepath).suffix.lstrip(".").lower() in IMAGE_TYPES
        )
        if clues.file_type:
            meta_checks += 1
            source_type = Path(doc.filepath).suffix.lstrip(".").lower()
            ft_req = clues.file_type.lower()
            if ft_req == "image":
                type_matches = doc.file_type.lower() in IMAGE_TYPES or source_type in IMAGE_TYPES
                clue_label = "IMAGE"
            elif ft_req == "audio":
                type_matches = doc.file_type.lower() in AUDIO_TYPES or source_type in AUDIO_TYPES
                clue_label = "AUDIO"
            elif ft_req == "video":
                type_matches = doc.file_type.lower() in VIDEO_TYPES or source_type in VIDEO_TYPES
                clue_label = "VIDEO"
            else:
                type_matches = ft_req in {doc.file_type.lower(), source_type}
                clue_label = ft_req.upper()

            if type_matches:
                meta_score += 1.0
                matching_clues.append(clue_label)
                evidence_list.append(EvidenceItem(
                    clue_name=clue_label,
                    clue_type="file_type",
                    matched_text=f"Document format matches requested '{clue_label}' type.",
                    location=f"Source file extension ({source_type})",
                    verified=True
                ))

        if clues.time_clues:
            meta_checks += 1
            time_matched = False
            for tc in clues.time_clues:
                tc_lower = tc.lower()
                if tc_lower in month_names:
                    target_month = month_names[tc_lower]
                    if (doc.modified_at and doc.modified_at.month == target_month) or \
                       (doc.created_at and doc.created_at.month == target_month):
                        time_matched = True
                        matching_clues.append(tc.title())
                        date_str = (doc.modified_at or datetime.utcnow()).strftime("%B %d, %Y")
                        evidence_list.append(EvidenceItem(
                            clue_name=tc.title(),
                            clue_type="date",
                            matched_text=f"Document modification timestamp matches ({date_str}).",
                            location="File Metadata: modified_at",
                            verified=True
                        ))
                        break
                if tc.isdigit() and len(tc) == 4:
                    target_year = int(tc)
                    if (doc.modified_at and doc.modified_at.year == target_year) or \
                       (doc.created_at and doc.created_at.year == target_year):
                        time_matched = True
                        break
                if tc_lower in doc_lower:
                    time_matched = True
                    break
            if time_matched:
                meta_score += 1.0

        # Entities (People, Technologies, Topics)
        for person in clues.people:
            pattern = r"\b" + re.escape(person.lower()) + r"\b"
            if re.search(pattern, doc_lower) or re.search(pattern, filename_lower):
                matching_clues.append(person)
                evidence_list.append(EvidenceItem(
                    clue_name=person,
                    clue_type="person",
                    matched_text=f"Referenced person '{person}' verified in document content.",
                    location=best_chunk_loc or "Content",
                    verified=True
                ))

        for tech in clues.technologies:
            pattern = r"\b" + re.escape(tech.lower()) + r"\b"
            if re.search(pattern, doc_lower) or re.search(pattern, filename_lower):
                matching_clues.append(tech)
                detected_entities.append(tech)
                evidence_list.append(EvidenceItem(
                    clue_name=tech,
                    clue_type="technology",
                    matched_text=f"Technology '{tech}' identified in document text/architecture.",
                    location=best_chunk_loc or "Content",
                    verified=True
                ))

        for topic in clues.topics:
            pattern = r"\b" + re.escape(topic.lower()) + r"\b"
            if re.search(pattern, doc_lower) or re.search(pattern, filename_lower):
                matching_clues.append(topic)
                evidence_list.append(EvidenceItem(
                    clue_name=topic,
                    clue_type="topic",
                    matched_text=f"Topic '{topic}' detected.",
                    location=best_chunk_loc or "Document",
                    verified=True
                ))

        has_image_content_text = (
            "visual description:" in doc_lower
            or "detected text in image:" in doc_lower
        )

        # A textless image has no OCR terms to match. Admit it only when its
        # stored vector is a compatible direct image embedding and it has a
        # positive cross-modal similarity to the query. Text embeddings of
        # image metadata must not masquerade as visual matches.
        doc_provenance = doc.get_embedding_provenance() or {}
        is_direct_visual_embedding = (
            doc_provenance.get("task") == "retrieval.image"
            and embedding_signature_is_compatible(doc_provenance, query_provenance)
        )
        visual_only_match = (
            is_image
            and not has_image_content_text
            and is_direct_visual_embedding
            and best_sem >= settings.SEARCH_SEMANTIC_MIN_SCORE
        )
        # The winning audio segment was scored in the same model space as the
        # query. Its timestamp label is not a transcript or a keyword match.
        audio_content_match = (
            doc.file_type in AUDIO_TYPES | VIDEO_TYPES
            and bool(best_chunk_loc and best_chunk_loc.startswith("Audio "))
            and best_sem >= settings.SEARCH_SEMANTIC_MIN_SCORE
        )
        video_content_match = (
            doc.file_type in VIDEO_TYPES
            and bool(best_chunk_loc and best_chunk_loc.startswith("Video "))
            and best_sem >= settings.SEARCH_SEMANTIC_MIN_SCORE
            and not literal_query
        )
        # Explicit text lookups remain literal. Natural-language searches can
        # retrieve synonyms/paraphrases using the neural embedding signal.
        semantic_match = (
            neural_query and not literal_query
            and best_sem >= settings.SEARCH_SEMANTIC_MIN_SCORE
            and (len(search_keywords) > 1 or bool(re.search(r"\b(?:about|related|describ)\w*\b", query, re.I)))
        )
        if quoted_terms and not audio_content_match and not all(term.lower() in doc_lower for term in quoted_terms):
            continue
        if total_kws > 0 and kw_matches == 0 and not (semantic_match or audio_content_match or video_content_match or (visual_only_match and not literal_query)):
            continue
        if semantic_match and kw_matches == 0 and not (visual_only_match or audio_content_match or video_content_match):
            matching_clues.append("Semantic similarity")

        if audio_content_match:
            matching_clues.append("Audio content")
            evidence_list.append(EvidenceItem(
                clue_name="Audio content", clue_type="audio",
                matched_text="The sound in this segment is similar to your search. A verbatim transcript is not available for this match.",
                location=best_chunk_loc, verified=False,
            ))
        if video_content_match:
            matching_clues.append("Video content")
            evidence_list.append(EvidenceItem(
                clue_name="Video content", clue_type="video",
                matched_text="The sampled visuals in this segment are similar to your search.",
                location=best_chunk_loc, verified=False,
            ))

        if visual_only_match and kw_matches == 0:
            matching_clues.append("Visual content")
            evidence_list.append(EvidenceItem(
                clue_name="Visual content",
                clue_type="visual",
                matched_text="Matched the query using the image's visual embedding.",
                location="Image content",
                verified=True,
            ))

        final_meta = (meta_score / meta_checks) if meta_checks > 0 else 0.5
        snippet = _extract_snippet(best_chunk_text or doc_text, search_keywords)

        # Baseline hybrid score (weights configurable in settings)
        # initial weights: semantic 0.45, reranker 0.20, keyword 0.15, metadata 0.10, filename 0.10
        scored_candidates.append({
            "doc": doc,
            "best_sem": best_sem,
            "kw_score": kw_score,
            "fn_score": fn_score,
            "meta_score": final_meta,
            "reranker_score": 0.0,
            "best_chunk_loc": best_chunk_loc,
            "snippet": snippet,
            "matching_clues": list(dict.fromkeys(matching_clues)),
            "detected_entities": list(dict.fromkeys(detected_entities)),
            "evidence": evidence_list,
            "repr_text": (best_chunk_text or doc_text)[:800] or doc.filename,
            # A text reranker cannot inspect pixels. Only send an image to it
            # when OCR or a genuine visual description was extracted.
            "text_rerankable": (
                not is_image
                or has_image_content_text
            ) and not (audio_content_match or video_content_match),
        })

    scored_at = time.perf_counter()
    # Bound remote work by hybrid relevance, keeping all local candidates in
    # the final ranking. Ask for every selected score to avoid mixing Jina's
    # top-N scores with incomparable fallback scores inside the reranked pool.
    def initial_score(candidate):
        return ((settings.WEIGHT_SEMANTIC + settings.WEIGHT_RERANKER) * candidate["best_sem"]
                + settings.WEIGHT_KEYWORD * candidate["kw_score"]
                + settings.WEIGHT_METADATA * candidate["meta_score"]
                + settings.WEIGHT_FILENAME * candidate["fn_score"])
    ranked_indices = sorted(range(len(scored_candidates)),
                            key=lambda i: initial_score(scored_candidates[i]), reverse=True)
    # 5. Jina Reranking Stage
    rerank_indices = [
        index for index in ranked_indices
        if scored_candidates[index]["text_rerankable"]
    ][:max(top_k, settings.RETRIEVAL_CANDIDATE_LIMIT, settings.RERANKER_TOP_K)]
    candidate_texts = [scored_candidates[index]["repr_text"] for index in rerank_indices]
    jina_results = None
    if settings.JINA_API_KEY and candidate_texts:
        try:
            jina_results = reranker_service.rerank(
                query=query,
                documents=candidate_texts,
                top_n=len(candidate_texts)
            )
        except Exception as je:
            logger.warning(f"Jina reranker failed: {je}. Falling back to standard hybrid.")

    reranked_at = time.perf_counter()

    # Apply reranker score if returned
    # Semantic scores are the correct reranking signal for textless images.
    # They remain intact even when the text reranker returns scores for other
    # candidates.
    for candidate in scored_candidates:
        candidate["reranker_score"] = candidate["best_sem"]
    if jina_results:
        for item in jina_results:
            idx = item.get("index")
            rel = item.get("relevance_score", 0.0)
            if idx is not None and 0 <= idx < len(rerank_indices):
                scored_candidates[rerank_indices[idx]]["reranker_score"] = float(rel)

    # Compute final composite score
    w_sem = settings.WEIGHT_SEMANTIC
    w_rerank = settings.WEIGHT_RERANKER
    w_kw = settings.WEIGHT_KEYWORD
    w_meta = settings.WEIGHT_METADATA
    w_fn = settings.WEIGHT_FILENAME

    final_results: List[Tuple[float, SearchResultItem]] = []

    for c in scored_candidates:
        doc = c["doc"]
        composite = (
            w_sem * c["best_sem"] +
            w_rerank * c["reranker_score"] +
            w_kw * c["kw_score"] +
            w_meta * c["meta_score"] +
            w_fn * c["fn_score"]
        )
        clamped_score = round(max(0.05, min(0.98, composite)), 2)

        unique_clues = c["matching_clues"]
        if clamped_score >= 0.70 or len(unique_clues) >= 4:
            score_label = "Strong match"
        elif clamped_score >= 0.50 or len(unique_clues) >= 2:
            score_label = "Good match"
        elif clamped_score >= 0.30 or len(unique_clues) >= 1:
            score_label = "Relevant"
        else:
            score_label = "Partial match"

        item = SearchResultItem(
            document_id=doc.id,
            filename=doc.filename,
            filepath=doc.filepath,
            file_type=doc.file_type,
            file_size=doc.file_size or 0,
            modified_at=doc.modified_at or datetime.utcnow(),
            score=clamped_score,
            score_label=score_label,
            snippet=c["snippet"],
            page_or_slide=c["best_chunk_loc"],
            matching_clues=unique_clues,
            detected_entities=c["detected_entities"],
            evidence=c["evidence"],
            ai_explanation=None # Will enrich top results below
        )
        final_results.append((composite, item))

    final_results.sort(key=lambda x: x[0], reverse=True)
    top_items = [r[1] for r in final_results[:top_k]]

    # 6. Generate grounded Gemini result explanations for top 3 results
    if settings.SEARCH_INLINE_EXPLANATIONS and settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        for item in top_items[:3]:
            try:
                # Find matching doc chunks
                matched_doc = next((d for d in candidate_docs if d.id == item.document_id), None)
                if matched_doc:
                    chunks_text = [ch.content for ch in matched_doc.chunks[:3]]
                    explanation = gemini_service.explain_match(
                        query=query,
                        filename=item.filename,
                        file_type=item.file_type,
                        relevant_chunks=chunks_text
                    )
                    item.ai_explanation = explanation
            except Exception as ee:
                logger.debug(f"AI explanation generation skipped: {ee}")

    logger.info(
        "Search timings: embedding=%.1fms retrieval=%.1fms scoring=%.1fms rerank=%.1fms total=%.1fms documents=%d candidates=%d",
        (embedded_at-started)*1000, (retrieved_at-embedded_at)*1000,
        (scored_at-retrieved_at)*1000, (reranked_at-scored_at)*1000,
        (time.perf_counter()-started)*1000, len(documents), len(scored_candidates),
    )
    return top_items

# Backward-compatible alias
search_hybrid = search_documents
