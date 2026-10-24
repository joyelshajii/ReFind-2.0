import logging
import asyncio
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.db.base import init_db
from app.api import search, documents, index, demo, health, duplicates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("refind")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at startup rather than accept uploads in a Python environment that
    # cannot inspect or decode media.
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            f"Media dependency PyAV is unavailable in {sys.executable}. "
            "On Windows, start ReFind with backend/start-local.ps1. "
            "Install dependencies into that environment with "
            "backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt."
        ) from error
    logger.info("Initializing ReFind database schema...")
    try:
        init_db()
    except Exception:
        logger.exception("Backend startup failed while initializing the database")
        raise
    if settings.LOCAL_FAST_MODE and settings.LOCAL_EMBEDDING_WARMUP:
        from app.services.embedding_service import get_embedding_with_provenance
        logger.info("Warming the cached EmbeddingGemma model before accepting searches...")
        try:
            await asyncio.to_thread(get_embedding_with_provenance, "ReFind search warmup", "retrieval.query")
        except Exception:
            logger.exception("Local embedding warmup failed; check cached weights and the backend environment")
    logger.info("ReFind backend started successfully.")
    from app.services.media_jobs import recover_media_jobs, run_next_media_job
    await asyncio.to_thread(recover_media_jobs)

    async def media_scheduler():
        while True:
            try:
                await asyncio.to_thread(run_next_media_job)
            except Exception:
                logger.exception("Unable to process queued media indexing")
            await asyncio.sleep(5)

    scheduler = asyncio.create_task(media_scheduler())
    try:
        yield
    finally:
        scheduler.cancel()
        try:
            await scheduler
        except asyncio.CancelledError:
            pass
        logger.info("ReFind backend shutting down.")

app = FastAPI(
    title="ReFind API",
    description="Context-Aware Digital Memory Search System by BitByBit",
    version=settings.VERSION,
    lifespan=lifespan
)


@app.exception_handler(SQLAlchemyError)
async def handle_database_error(request: Request, exc: SQLAlchemyError):
    logger.exception("Database request failed for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "The database is temporarily unavailable. Please try again."},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled request failure for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Please try again."},
    )

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(index.router)
app.include_router(duplicates.router)
app.include_router(demo.router)

@app.get("/")
def root():
    return {
        "product": settings.PROJECT_NAME,
        "tagline": settings.TAGLINE,
        "team": settings.TEAM_NAME,
        "docs": "/docs",
        "health": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT)
