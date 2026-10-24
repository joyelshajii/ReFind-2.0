import logging
import math
import hashlib
import json
import requests
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.request_cache import RequestCache

logger = logging.getLogger("refind.services.reranker")

class RerankerService:
    def __init__(self):
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.model = "jina-reranker-v2-base-multilingual"
        self._cache = RequestCache()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Calls Jina Reranker API to rerank retrieved candidate document texts against the query.
        Returns a list of dicts: [{"index": int, "relevance_score": float, "document": str}, ...]
        Returns None if Jina is unavailable or fails, allowing seamless fallback.
        """
        if not settings.JINA_API_KEY:
            logger.debug("Jina API key not set, skipping external reranker.")
            return None

        if not documents:
            return []

        top_n = min(max(1, top_n or settings.RERANKER_TOP_K), len(documents))

        headers = {
            "Authorization": f"Bearer {settings.JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": settings.JINA_RERANKER_MODEL,
            "query": query,
            "documents": [d[:1000] for d in documents], # Truncate long chunks for speed and token limits
            "top_n": top_n
        }

        key = hashlib.sha256((settings.JINA_API_KEY + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        return self._cache.get_or_call(
            key, settings.RERANK_CACHE_SECONDS,
            lambda: self._request(headers, payload, len(documents)),
        )

    def _request(self, headers, payload, document_count):
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload,
                                 timeout=settings.RERANK_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results") if isinstance(data, dict) else None
                if not isinstance(results, list):
                    logger.warning("Jina reranker returned an unexpected response shape.")
                    return None

                valid_results = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("index")
                    score = item.get("relevance_score")
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < document_count:
                        continue
                    try:
                        score = float(score)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not math.isfinite(score):
                        continue
                    valid_results.append({
                        **item,
                        "index": index,
                        "relevance_score": max(0.0, min(1.0, score)),
                    })

                logger.info("Jina reranked %s candidates successfully.", document_count)
                return valid_results
            else:
                logger.warning("Jina reranker returned status %s: %s", resp.status_code, resp.text[:500])
                return None
        except Exception as e:
            logger.warning(f"Jina reranker request failed: {e}. Falling back to hybrid score.")
            return None

reranker_service = RerankerService()
