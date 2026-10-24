import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pathlib import Path
from app.core.config import settings
from app.demo_data.generate_demo_data import generate_demo_files
from app.services.indexer import index_folder_background

router = APIRouter(prefix="/api/demo", tags=["Demo"])
logger = logging.getLogger("refind.api.demo")

@router.post("/seed")
def seed_demo_dataset(background_tasks: BackgroundTasks):
    """
    Generates realistic demo files (final2.pdf, IMG_2384.png, etc.)
    and immediately queues them into the indexing pipeline.
    """
    try:
        demo_dir = Path(settings.DEMO_DATA_DIR)
        generate_demo_files(str(demo_dir.resolve()))

        from app.db.base import SessionLocal
        from app.db.models import Document

        # Reset database for fresh demo
        db = SessionLocal()
        try:
            db.query(Document).delete()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        # Trigger background indexing
        background_tasks.add_task(index_folder_background, str(demo_dir.resolve()))

        return {
            "status": "success",
            "message": "Demo files generated and indexing started.",
            "demo_scenario": "Find the architecture PDF Rahul sent me around February. It had PostgreSQL and BLE."
        }
    except Exception as e:
        logger.exception("Failed to seed demo data")
        raise HTTPException(status_code=500, detail="Failed to prepare demo data. Check the backend logs for details.") from e
