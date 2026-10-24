"""Durable user choices for expensive media indexing; executed by the backend."""
from datetime import datetime, timezone
from pathlib import Path
import threading

from app.db.base import SessionLocal
from app.db.models import MediaIndexJob
from app.services.media_preflight import inspect_media, allow_media_processing

_jobs_lock = threading.RLock()


def job_dict(job):
    return {
        "id": job.id, "filename": job.filename, "duration_seconds": job.duration_seconds,
        "status": job.status,
        "scheduled_at": job.scheduled_at.replace(tzinfo=timezone.utc).isoformat() if job.scheduled_at else None,
        "error_message": job.error_message,
    }


def defer_media(plan):
    with _jobs_lock, SessionLocal() as db:
        job = db.query(MediaIndexJob).filter_by(filepath=plan.filepath).first()
        if job is None:
            job = MediaIndexJob(filepath=plan.filepath, filename=Path(plan.filepath).name)
            db.add(job)
        if job.fingerprint != plan.fingerprint or job.status in {None, "completed", "failed", "skipped"}:
            job.fingerprint = plan.fingerprint
            job.duration_seconds = plan.duration_seconds
            job.status = "awaiting"
            job.scheduled_at = None
            job.error_message = None
            job.updated_at = datetime.utcnow()
        db.commit()
        return job_dict(job)


def list_media_jobs():
    with SessionLocal() as db:
        rows = db.query(MediaIndexJob).filter(MediaIndexJob.status.notin_(["completed", "skipped"]))
        return [job_dict(job) for job in rows.order_by(MediaIndexJob.created_at).all()]


def complete_media_request(db, filepath):
    """Commit an obsolete pending request atomically with the new document index."""
    db.query(MediaIndexJob).filter(
        MediaIndexJob.filepath == filepath,
        MediaIndexJob.status.in_(["awaiting", "queued", "scheduled", "failed"]),
    ).update({"status": "completed", "error_message": None, "updated_at": datetime.utcnow()})


def decide_media_job(job_id, action, scheduled_at=None):
    with _jobs_lock, SessionLocal() as db:
        job = db.get(MediaIndexJob, job_id)
        if job is None:
            raise LookupError("Media indexing request not found.")
        if job.status == "running":
            raise RuntimeError("This file is already being indexed.")
        if job.status == "completed":
            raise RuntimeError("This file has already been indexed.")
        if action == "schedule":
            if scheduled_at is None or scheduled_at.utcoffset() is None:
                raise ValueError("Choose a date and time with a timezone.")
            scheduled_at = scheduled_at.astimezone(timezone.utc).replace(tzinfo=None)
            if scheduled_at <= datetime.utcnow():
                raise ValueError("Choose a future date and time.")
        if action != "skip":
            # A choice is tied to the inspected media and its companion files.
            plan = inspect_media(job.filepath)
            if plan.fingerprint != job.fingerprint:
                job.fingerprint = plan.fingerprint
                job.duration_seconds = plan.duration_seconds
                job.status = "awaiting"
                job.scheduled_at = None
                job.error_message = "The file or its transcript changed. Please review and choose again."
                db.commit()
                raise RuntimeError(job.error_message)
        job.status = {"continue": "queued", "schedule": "scheduled", "skip": "skipped"}[action]
        job.scheduled_at = scheduled_at if action == "schedule" else None
        job.error_message = None
        job.updated_at = datetime.utcnow()
        db.commit()
        return job_dict(job)


def recover_media_jobs():
    # A process interruption is resumable. Fingerprints are checked again at execution.
    with SessionLocal() as db:
        db.query(MediaIndexJob).filter_by(status="running").update({"status": "queued"})
        db.commit()


def run_next_media_job():
    from sqlalchemy import or_, and_
    from app.services import indexer

    with _jobs_lock, SessionLocal() as db:
        job = db.query(MediaIndexJob).filter(or_(
            MediaIndexJob.status == "queued",
            and_(MediaIndexJob.status == "scheduled", MediaIndexJob.scheduled_at <= datetime.utcnow()),
        )).order_by(MediaIndexJob.created_at).first()
        if job is None or not indexer.try_reserve_indexing_run():
            return
        job.status = "running"
        job.updated_at = datetime.utcnow()
        try:
            db.commit()
        except Exception:
            indexer.release_indexing_run()
            raise
        job_id, filepath, fingerprint = job.id, job.filepath, job.fingerprint

    status = indexer.indexing_status
    status.current_file = Path(filepath).name
    status.current_stage = "Checking media"
    status.total_files, status.processed_files = 1, 0
    status.error_message = None
    try:
        with indexer.file_index_lock(filepath):
            plan = inspect_media(filepath)
            if plan.fingerprint != fingerprint:
                with _jobs_lock, SessionLocal() as db:
                    job = db.get(MediaIndexJob, job_id)
                    job.fingerprint, job.duration_seconds = plan.fingerprint, plan.duration_seconds
                    job.status, job.scheduled_at = "awaiting", None
                    job.error_message = "The file changed since your approval. Please choose again."
                    db.commit()
                status.current_stage = "Awaiting confirmation"
                return
            with allow_media_processing(plan), indexer.SessionLocal() as db:
                indexer.index_single_file(filepath, db)
        with _jobs_lock, SessionLocal() as db:
            job = db.get(MediaIndexJob, job_id)
            job.status, job.error_message = "completed", None
            job.updated_at = datetime.utcnow()
            db.commit()
        status.processed_files = 1
        status.last_indexed_at = datetime.utcnow()
        status.current_stage = "Complete"
    except Exception as error:
        message = indexer._display_index_error(error)
        with _jobs_lock, SessionLocal() as db:
            job = db.get(MediaIndexJob, job_id)
            job.status, job.error_message = "failed", message
            job.updated_at = datetime.utcnow()
            db.commit()
        status.current_stage, status.error_message = "Error", message
        indexer.logger.exception("Scheduled media indexing failed for %s", filepath)
    finally:
        status.current_file = None
        indexer.release_indexing_run()
