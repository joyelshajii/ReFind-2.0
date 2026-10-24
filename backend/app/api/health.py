from fastapi import APIRouter
from app.core.config import settings
from app.db.base import is_postgres_active, is_pgvector_active
from app.services.embedding_service import local_embedding_runtime_stats

router = APIRouter(prefix="/api/health", tags=["Health"])

@router.get("")
def health_check():
    return {
        "status": "healthy",
        "product": settings.PROJECT_NAME,
        "tagline": settings.TAGLINE,
        "team": settings.TEAM_NAME,
        "version": settings.VERSION,
        "database": {
            "postgres_connected": is_postgres_active,
            "pgvector_enabled": is_pgvector_active,
            "engine": "PostgreSQL with pgvector" if is_pgvector_active else "Local SQLite + Cosine Vector Engine"
        },
        "providers": {
            "embedding": "embeddinggemma" if settings.LOCAL_FAST_MODE else settings.EMBEDDING_PROVIDER,
            "llm": settings.LLM_PROVIDER,
            "search_explanations": "inline" if settings.SEARCH_INLINE_EXPLANATIONS else "on_demand",
        },
        "local_embedding": local_embedding_runtime_stats() if settings.LOCAL_FAST_MODE else None,
    }
