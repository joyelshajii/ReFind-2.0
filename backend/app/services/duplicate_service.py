import os
import re
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import Document, Chunk, Entity
from app.schemas.duplicate import (
    DuplicateFileItem,
    DuplicateGroup,
    DuplicateOverviewResponse,
    ResolveGroupResponse,
    ResolveAllResponse,
)

logger = logging.getLogger("refind.services.duplicate")

COPY_SUFFIX_REGEX = re.compile(
    r"(\s*[-_]?\s*copy(\s*\d+)?|\s*\(\d+\)|[-_]v\d+|[-_]draft|[-_]final\d*)$",
    re.IGNORECASE,
)

def _clean_basename(filename: str) -> str:
    """Strip extension and common copy suffixes like (1), _copy, - copy 2, v2."""
    stem = Path(filename).stem
    cleaned = COPY_SUFFIX_REGEX.sub("", stem).strip()
    return cleaned.lower() if cleaned else stem.lower()

def _compute_sha256(filepath: str) -> Optional[str]:
    """Compute SHA-256 for a file on disk."""
    try:
        p = Path(filepath)
        if not p.is_file():
            return None
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning("Could not compute hash for %s: %s", filepath, e)
        return None

def _get_effective_mtime(doc: Document) -> datetime:
    """
    Get file modification time from database modified_at, falling back to disk st_mtime or created_at.
    """
    if doc.modified_at:
        return doc.modified_at
    try:
        if doc.filepath:
            p = Path(doc.filepath)
            if p.exists():
                ts = p.stat().st_mtime
                return datetime.fromtimestamp(ts)
    except Exception:
        pass
    if doc.created_at:
        return doc.created_at
    return datetime.min

def _format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count} B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count / (1024 * 1024):.1f} MB"

def _format_time_diff(dt1: datetime, dt2: datetime) -> str:
    """Format human-readable time difference between two dates."""
    diff = abs(dt1 - dt2)
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return "less than a minute"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''}"
    months = days // 30
    return f"{months} month{'s' if months != 1 else ''}"

def _generate_group_explanation(
    match_type: str,
    docs: List[Document],
    recommended_doc: Document,
    mtimes: Dict[str, datetime],
    confidence: float = 1.0,
) -> Tuple[str, str]:
    """Generate clear 'why duplicate' and 'differences' context."""
    # Why duplicate
    if match_type == "exact_hash":
        sample_hash = docs[0].file_hash[:12] if docs[0].file_hash else ""
        why = f"Binary identical: Files produce identical SHA-256 checksums ({sample_hash}...) and matching byte size."
    elif match_type == "exact_text":
        why = "Identical text content: Both documents contain 100% identical extracted textual content."
    elif match_type == "filename_and_content":
        sample_names = ", ".join(f"'{d.filename}'" for d in docs[:2])
        why = f"Filename variant with matching content: Similar name pattern ({sample_names}) with confirmed content match."
    elif match_type == "similar_content":
        why = f"High content similarity: Documents share ~{int(confidence * 100)}% overlapping textual content."
    else:
        why = "Detected as duplicate based on matching file attributes and content."

    # What is different
    diff_parts = []

    # 1. Directory differences
    dirs = {d.id: str(Path(d.filepath).parent) for d in docs}
    unique_dirs = set(dirs.values())
    if len(unique_dirs) > 1:
        dir_samples = [f"'{p}'" for p in list(unique_dirs)[:2]]
        diff_parts.append(f"Located in different folders ({' vs '.join(dir_samples)})")
    else:
        diff_parts.append("Located in the same folder")

    # 2. Timestamp difference
    rec_time = mtimes.get(recommended_doc.id, datetime.min)
    other_times = [
        (d, mtimes.get(d.id, datetime.min))
        for d in docs if d.id != recommended_doc.id
    ]
    if other_times:
        other_times.sort(key=lambda x: x[1])
        oldest_doc, oldest_time = other_times[0]
        if rec_time > oldest_time and (rec_time - oldest_time).total_seconds() > 60:
            time_diff_str = _format_time_diff(rec_time, oldest_time)
            diff_parts.append(
                f"Recommended file '{recommended_doc.filename}' is newer by {time_diff_str} "
                f"(modified {rec_time.strftime('%b %d, %Y')}) compared to '{oldest_doc.filename}' "
                f"(modified {oldest_time.strftime('%b %d, %Y')})"
            )
        elif rec_time == oldest_time:
            diff_parts.append("Files have identical modification timestamps")

    # 3. Size difference
    sizes = [d.file_size or 0 for d in docs]
    if max(sizes) != min(sizes) and min(sizes) > 0:
        diff_bytes = max(sizes) - min(sizes)
        diff_parts.append(f"File sizes differ by {_format_size(diff_bytes)}")
    else:
        diff_parts.append(f"Exact same file size ({_format_size(sizes[0])})")

    # 4. Content length differences
    texts = [len((d.extracted_text or "").split()) for d in docs if d.extracted_text]
    if len(texts) == len(docs) and max(texts) != min(texts) and min(texts) > 0:
        diff_words = max(texts) - min(texts)
        diff_parts.append(f"Content differs by ~{diff_words} words")

    differences = ". ".join(diff_parts) + "."
    return why, differences

def find_duplicates(db: Session, min_content_similarity: float = 0.85) -> DuplicateOverviewResponse:
    """
    Scan all indexed documents and identify duplicate groups using:
    1. Exact SHA-256 hash matching (binary identical)
    2. Exact extracted text hash matching (same content across format conversions)
    3. Filename similarity + content match (e.g. 'file.pdf' vs 'file (1).pdf')
    4. Near-duplicate text similarity (>= 85%)

    Also determines the 'best one to keep' (the latest updated file).
    """
    docs = db.query(Document).all()
    if not docs or len(docs) < 2:
        return DuplicateOverviewResponse(
            total_groups=0,
            total_duplicate_files=0,
            total_reclaimable_bytes=0,
            groups=[],
        )

    assigned_doc_ids: Set[str] = set()
    raw_groups: List[Dict[str, Any]] = []

    # ----------------------------------------------------
    # TIER 1: Exact Hash Matching (Binary Identical)
    # ----------------------------------------------------
    hash_map = defaultdict(list)
    for doc in docs:
        if doc.file_hash and doc.file_hash.strip():
            hash_map[doc.file_hash.strip()].append(doc)

    for fhash, group_docs in hash_map.items():
        if len(group_docs) > 1:
            raw_groups.append({
                "match_type": "exact_hash",
                "match_label": "Exact Content Match",
                "confidence": 1.0,
                "docs": group_docs,
            })
            for d in group_docs:
                assigned_doc_ids.add(d.id)

    # ----------------------------------------------------
    # TIER 2: Exact Extracted Text Hash Matching
    # ----------------------------------------------------
    text_hash_map = defaultdict(list)
    for doc in docs:
        if doc.id in assigned_doc_ids:
            continue
        text = (doc.extracted_text or "").strip()
        if len(text) >= 50:
            norm_text = " ".join(text.split()).lower()
            thash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
            text_hash_map[thash].append(doc)

    for thash, group_docs in text_hash_map.items():
        if len(group_docs) > 1:
            raw_groups.append({
                "match_type": "exact_text",
                "match_label": "Identical Text Content",
                "confidence": 1.0,
                "docs": group_docs,
            })
            for d in group_docs:
                assigned_doc_ids.add(d.id)

    # ----------------------------------------------------
    # TIER 3: Filename Similarity + Content Checking
    # ----------------------------------------------------
    remaining_docs = [d for d in docs if d.id not in assigned_doc_ids]
    name_cluster_map = defaultdict(list)
    for doc in remaining_docs:
        base_clean = _clean_basename(doc.filename)
        if len(base_clean) >= 2:
            name_cluster_map[base_clean].append(doc)

    # Pre-tokenize sample text for fast token comparison
    token_cache: Dict[str, Set[str]] = {}
    def get_tokens(d: Document) -> Set[str]:
        if d.id not in token_cache:
            raw = (d.extracted_text or "")[:15000].lower()
            token_cache[d.id] = set(re.findall(r"\w+", raw))
        return token_cache[d.id]

    for base_clean, candidates in name_cluster_map.items():
        if len(candidates) > 1:
            matched = [candidates[0]]
            t0 = get_tokens(candidates[0])
            s0 = candidates[0].file_size or 0

            for other in candidates[1:]:
                t_other = get_tokens(other)
                s_other = other.file_size or 0
                if t0 and t_other:
                    union = t0 | t_other
                    sim = len(t0 & t_other) / len(union) if union else 0.0
                    if sim >= 0.70:
                        matched.append(other)
                elif s0 > 0 and s_other > 0 and abs(s0 - s_other) / max(s0, s_other) <= 0.05:
                    matched.append(other)

            if len(matched) > 1:
                raw_groups.append({
                    "match_type": "filename_and_content",
                    "match_label": "Filename Variant with Matching Content",
                    "confidence": 0.95,
                    "docs": matched,
                })
                for d in matched:
                    assigned_doc_ids.add(d.id)

    # ----------------------------------------------------
    # TIER 4: High Content Similarity (Near-Duplicates)
    # ----------------------------------------------------
    still_remaining = [
        d for d in docs
        if d.id not in assigned_doc_ids and d.extracted_text and len(d.extracted_text.strip()) >= 80
    ]
    for i in range(len(still_remaining)):
        d1 = still_remaining[i]
        if d1.id in assigned_doc_ids:
            continue
        tok1 = get_tokens(d1)
        if not tok1:
            continue
        cluster = [d1]
        for j in range(i + 1, len(still_remaining)):
            d2 = still_remaining[j]
            if d2.id in assigned_doc_ids:
                continue
            tok2 = get_tokens(d2)
            if not tok2:
                continue
            len_ratio = min(len(tok1), len(tok2)) / max(len(tok1), len(tok2))
            if len_ratio < min_content_similarity:
                continue
            union = tok1 | tok2
            sim = len(tok1 & tok2) / len(union) if union else 0.0
            if sim >= min_content_similarity:
                cluster.append(d2)

        if len(cluster) > 1:
            raw_groups.append({
                "match_type": "similar_content",
                "match_label": "Near-Duplicate Text",
                "confidence": min_content_similarity,
                "docs": cluster,
            })
            for d in cluster:
                assigned_doc_ids.add(d.id)

    # ----------------------------------------------------
    # Build Structured Output & Identify Best One to Keep
    # ----------------------------------------------------
    final_groups: List[DuplicateGroup] = []
    total_reclaimable_bytes = 0
    total_duplicate_files = 0

    for idx, rg in enumerate(raw_groups):
        gdocs: List[Document] = rg["docs"]
        match_type = rg["match_type"]
        match_label = rg["match_label"]
        confidence = rg["confidence"]

        # Determine effective mtimes for each file
        mtimes: Dict[str, datetime] = {d.id: _get_effective_mtime(d) for d in gdocs}

        # Select the best one to keep:
        # Criterion 1: latest updated one (highest mtime)
        # Criterion 2: cleaner filename (does not contain copy suffix)
        # Criterion 3: larger size / more text
        def _sort_key(d: Document):
            mtime = mtimes.get(d.id, datetime.min)
            has_copy_pattern = bool(COPY_SUFFIX_REGEX.search(Path(d.filename).stem))
            clean_score = 0 if has_copy_pattern else 1
            size = d.file_size or 0
            return (mtime, clean_score, size)

        sorted_docs = sorted(gdocs, key=_sort_key, reverse=True)
        recommended_doc = sorted_docs[0]

        # Calculate potential savings: sum of sizes of all duplicate files except recommended
        savings = sum((d.file_size or 0) for d in sorted_docs[1:])
        total_reclaimable_bytes += savings
        total_duplicate_files += (len(sorted_docs) - 1)

        why, diffs = _generate_group_explanation(
            match_type, gdocs, recommended_doc, mtimes, confidence
        )

        rec_mtime = mtimes.get(recommended_doc.id, datetime.min)

        file_items: List[DuplicateFileItem] = []
        for d in sorted_docs:
            is_rec = (d.id == recommended_doc.id)
            d_mtime = mtimes.get(d.id, datetime.min)
            if is_rec:
                other_mtimes = [mtimes.get(o.id, datetime.min) for o in sorted_docs if o.id != d.id]
                oldest_other = min(other_mtimes) if other_mtimes else d_mtime
                if (d_mtime - oldest_other).total_seconds() > 60:
                    diff_str = _format_time_diff(d_mtime, oldest_other)
                    reason = f"Latest updated file (newer by {diff_str})"
                else:
                    reason = "Latest updated / canonical version to keep"
            else:
                reason = "Redundant duplicate copy"

            file_exists = Path(d.filepath).is_file() if d.filepath else False

            file_items.append(
                DuplicateFileItem(
                    id=d.id,
                    filename=d.filename,
                    filepath=d.filepath,
                    file_type=d.file_type,
                    file_size=d.file_size or 0,
                    modified_at=d_mtime,
                    created_at=d.created_at,
                    is_recommended=is_rec,
                    recommendation_reason=reason,
                    exists_on_disk=file_exists,
                )
            )

        group_id = f"dup_{idx + 1}_{recommended_doc.id[:8]}"
        final_groups.append(
            DuplicateGroup(
                group_id=group_id,
                match_type=match_type,
                match_label=match_label,
                confidence=confidence,
                why_duplicate=why,
                differences=diffs,
                potential_savings_bytes=savings,
                recommended_document_id=recommended_doc.id,
                files=file_items,
            )
        )

    # Sort groups so highest potential savings are shown first
    final_groups.sort(key=lambda g: g.potential_savings_bytes, reverse=True)

    return DuplicateOverviewResponse(
        total_groups=len(final_groups),
        total_duplicate_files=total_duplicate_files,
        total_reclaimable_bytes=total_reclaimable_bytes,
        groups=final_groups,
    )

def delete_single_duplicate(
    db: Session, document_id: str, delete_from_disk: bool = True
) -> Dict[str, Any]:
    """
    Delete a single duplicate document:
    1. If delete_from_disk is True, delete the file from the filesystem.
    2. Delete document and all associated chunks and entities from the database.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise ValueError(f"Document with ID {document_id} not found.")

    filename = doc.filename
    filepath = doc.filepath
    disk_deleted = False

    if delete_from_disk and filepath:
        file_path_obj = Path(filepath)
        if file_path_obj.exists() and file_path_obj.is_file():
            try:
                file_path_obj.unlink()
                disk_deleted = True
                logger.info("Deleted file from disk: %s", filepath)
            except Exception as e:
                logger.error("Failed to delete file from disk %s: %s", filepath, e)
                raise RuntimeError(f"Could not delete physical file on disk ({e}).")

    # Cascade delete in DB
    db.delete(doc)
    db.commit()

    from app.services.indexer import indexing_status
    total_docs = db.query(Document).count()
    indexing_status.processed_files = total_docs
    indexing_status.total_files = total_docs

    msg = f"Successfully removed '{filename}'"
    if disk_deleted:
        msg += " from disk and index."
    else:
        msg += " from index (file preserved on disk)."

    return {
        "status": "success",
        "message": msg,
        "document_id": document_id,
        "filename": filename,
        "disk_deleted": disk_deleted,
    }

def resolve_duplicate_group(
    db: Session,
    keep_document_id: str,
    delete_document_ids: Optional[List[str]] = None,
    delete_from_disk: bool = True,
) -> ResolveGroupResponse:
    """
    Keep the designated best document and delete the other duplicate copies from disk and index.
    """
    keep_doc = db.query(Document).filter(Document.id == keep_document_id).first()
    if not keep_doc:
        raise ValueError(f"Target document to keep ({keep_document_id}) not found.")

    if not delete_document_ids:
        overview = find_duplicates(db)
        target_group = None
        for grp in overview.groups:
            if any(f.id == keep_document_id for f in grp.files):
                target_group = grp
                break
        if not target_group:
            raise ValueError(f"No duplicate group found containing document {keep_document_id}.")
        delete_document_ids = [f.id for f in target_group.files if f.id != keep_document_id]

    deleted_count = 0
    freed_bytes = 0

    for doc_id in delete_document_ids:
        if doc_id == keep_document_id:
            continue
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            continue
        freed_bytes += (doc.file_size or 0)
        if delete_from_disk and doc.filepath:
            p = Path(doc.filepath)
            if p.exists() and p.is_file():
                try:
                    p.unlink()
                except Exception as e:
                    logger.warning("Could not delete %s from disk: %s", doc.filepath, e)
        db.delete(doc)
        deleted_count += 1

    db.commit()

    from app.services.indexer import indexing_status
    total_docs = db.query(Document).count()
    indexing_status.processed_files = total_docs
    indexing_status.total_files = total_docs

    return ResolveGroupResponse(
        status="success",
        message=f"Kept '{keep_doc.filename}' and deleted {deleted_count} duplicate copy/copies ({_format_size(freed_bytes)} freed).",
        kept_document_id=keep_document_id,
        deleted_count=deleted_count,
        freed_bytes=freed_bytes,
    )

def resolve_all_duplicates(db: Session, delete_from_disk: bool = True) -> ResolveAllResponse:
    """
    Batch resolve ALL duplicate groups across the system:
    Keeps the recommended (latest updated) file in each group and deletes all other duplicate copies.
    """
    overview = find_duplicates(db)
    if not overview.groups:
        return ResolveAllResponse(
            status="success",
            message="No duplicates found to resolve.",
            groups_resolved=0,
            total_deleted=0,
            freed_bytes=0,
        )

    groups_resolved = 0
    total_deleted = 0
    freed_bytes = 0

    for group in overview.groups:
        keep_id = group.recommended_document_id
        delete_ids = [f.id for f in group.files if f.id != keep_id]
        if not delete_ids:
            continue

        for doc_id in delete_ids:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                continue
            freed_bytes += (doc.file_size or 0)
            if delete_from_disk and doc.filepath:
                p = Path(doc.filepath)
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                    except Exception as e:
                        logger.warning("Could not delete %s: %s", doc.filepath, e)
            db.delete(doc)
            total_deleted += 1

        groups_resolved += 1

    db.commit()

    from app.services.indexer import indexing_status
    total_docs = db.query(Document).count()
    indexing_status.processed_files = total_docs
    indexing_status.total_files = total_docs

    return ResolveAllResponse(
        status="success",
        message=f"Resolved {groups_resolved} duplicate group(s), removed {total_deleted} redundant file(s), and freed {_format_size(freed_bytes)} of storage.",
        groups_resolved=groups_resolved,
        total_deleted=total_deleted,
        freed_bytes=freed_bytes,
    )
