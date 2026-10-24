import math
import hashlib
import re
import logging
import json
import os
import threading
from contextlib import contextmanager
from collections import OrderedDict
from typing import List, Optional
import numpy as np
import httpx
from app.core.config import settings

logger = logging.getLogger("refind.services.embedding")

DIMENSION = settings.EMBEDDING_DIMENSION  # Default 768
JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
_EMBEDDING_CACHE_LIMIT = 1024
_embedding_cache = OrderedDict()
_local_embedding_cache = OrderedDict()
_embedding_cache_lock = threading.RLock()
_embedding_key_locks = {}
_local_model_lock = threading.RLock()
_inference_condition = threading.Condition()
_inference_busy = False
_waiting_queries = 0
_local_model = None
_local_model_config = None
_local_model_load_seconds = None
_local_model_device = None
_local_model_precision = None
_embeddinggemma_cache = OrderedDict()
_embeddinggemma_cache_limit = 1024


def _embeddinggemma_runtime_config():
    """Resolve a supported inference dtype without ever selecting float16."""
    try:
        import torch
    except Exception as error:
        raise RuntimeError("Local-fast indexing requires the installed PyTorch runtime.") from error

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, dtype, "bfloat16" if use_bf16 else "float32"


def _embeddinggemma_provenance(task: str) -> dict:
    device, _, precision = _embeddinggemma_runtime_config()
    model = settings.LOCAL_EMBEDDING_MODEL_ID
    revision = settings.LOCAL_EMBEDDING_MODEL_REVISION
    dimension = settings.EMBEDDING_DIMENSION
    configuration = {
        "provider": "embeddinggemma",
        "model": model,
        "revision": revision,
        "dimension": dimension,
        "precision": precision,
        "normalization": "l2",
        "implementation": "sentence-transformers-multimodal-v1",
    }
    config_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**configuration, "config_id": config_id, "task": task, "device": device}


def _get_embeddinggemma_model():
    """Load one pinned model instance from the local HF cache, never the network."""
    global _local_model, _local_model_config, _local_model_load_seconds
    global _local_model_device, _local_model_precision
    import time

    device, dtype, precision = _embeddinggemma_runtime_config()
    config = (settings.LOCAL_EMBEDDING_MODEL_ID, settings.LOCAL_EMBEDDING_MODEL_REVISION, device, precision)
    with _local_model_lock:
        if _local_model is not None and _local_model_config == config:
            return _local_model
        try:
            # Hugging Face and Transformers must not attempt network access as
            # a side effect of local-fast indexing.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            model = SentenceTransformer(
                settings.LOCAL_EMBEDDING_MODEL_ID,
                revision=settings.LOCAL_EMBEDDING_MODEL_REVISION,
                device=device,
                local_files_only=True,
                model_kwargs={"torch_dtype": dtype},
            )
            if device == "cuda":
                torch.cuda.synchronize()
            _local_model_load_seconds = time.perf_counter() - started
            _local_model = model
            _local_model_config = config
            _local_model_device = device
            _local_model_precision = precision
            logger.info(
                "Loaded local embedding model %s@%s on %s (%s) in %.3fs",
                config[0], config[1], device, precision, _local_model_load_seconds,
            )
            return model
        except Exception as error:
            raise RuntimeError(
                f"Local-fast mode could not load {config[0]}@{config[1]} from the local Hugging Face cache. "
                "No remote embedding fallback was attempted. Verify the cached snapshot and model dependencies."
            ) from error


def _embeddinggemma_cache_key(text: str, task: str):
    provenance = _embeddinggemma_provenance(task)
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return (provenance["config_id"], task, digest)


def _encode_gemma_text_batch(texts: list[str], task: str):
    """Encode a bounded text batch, splitting and retrying CUDA OOM failures."""
    import torch

    model = _get_embeddinggemma_model()
    prompt_name = "SearchQuery" if task == "retrieval.query" else "Document"
    batch_size = max(1, min(int(settings.LOCAL_EMBEDDING_BATCH_SIZE), len(texts)))
    try:
        vectors = model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(texts) == 1 and vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape != (len(texts), settings.EMBEDDING_DIMENSION) or not np.isfinite(vectors).all():
            raise ValueError(
                f"EmbeddingGemma returned invalid text vectors with shape {vectors.shape}; "
                f"expected ({len(texts)}, {settings.EMBEDDING_DIMENSION}) finite values."
            )
        return vectors
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        if not is_oom or len(texts) <= 1:
            raise
        logger.warning("EmbeddingGemma batch OOM at size %d; retrying in smaller batches.", len(texts))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        midpoint = len(texts) // 2
        return np.concatenate((
            _encode_gemma_text_batch(texts[:midpoint], task),
            _encode_gemma_text_batch(texts[midpoint:], task),
        ), axis=0)


@contextmanager
def _inference_slot(is_query=False):
    """Keep one model invocation active; let searches run between index batches."""
    global _inference_busy, _waiting_queries
    with _inference_condition:
        if is_query:
            _waiting_queries += 1
            _inference_condition.notify_all()
        try:
            _inference_condition.wait_for(
                lambda: not _inference_busy and (is_query or _waiting_queries == 0)
            )
            _inference_busy = True
        finally:
            if is_query:
                _waiting_queries -= 1
    try:
        yield
    finally:
        with _inference_condition:
            _inference_busy = False
            _inference_condition.notify_all()


def embed_texts_with_provenance(texts: List[str], task: str = "retrieval.passage"):
    """Length-grouped EmbeddingGemma batches with bounded GPU use and caching."""
    if not texts:
        return []
    if not settings.LOCAL_FAST_MODE:
        return [embed_text_with_provenance(text, task=task) for text in texts]

    provenance = _embeddinggemma_provenance(task)
    keys = [(provenance["config_id"], task, hashlib.sha256((text or "").encode()).hexdigest())
            for text in texts]
    unique_missing, results = {}, {}
    # Cache hits never queue behind a long model invocation.
    with _embedding_cache_lock:
        for text, key in zip(texts, keys):
            cached = _embeddinggemma_cache.get(key)
            if cached is not None:
                _embeddinggemma_cache.move_to_end(key)
                results[key] = cached
            else:
                unique_missing.setdefault(key, text or "")
    # Similar lengths reduce padding without changing inputs, prompts, or order
    # in the returned result. Model batch size stays safe for the 4GB GPU.
    missing_keys = sorted(unique_missing, key=lambda key: len(unique_missing[key]))
    batch_limit = max(1, int(settings.LOCAL_EMBEDDING_BATCH_SIZE))
    for start in range(0, len(missing_keys), batch_limit):
        with _inference_slot(is_query=task == "retrieval.query"):
            batch_keys = []
            with _embedding_cache_lock:
                for key in missing_keys[start:start + batch_limit]:
                    cached = _embeddinggemma_cache.get(key)
                    if cached is not None:
                        results[key] = cached
                    else:
                        batch_keys.append(key)
            if not batch_keys:
                continue
            vectors = _encode_gemma_text_batch([unique_missing[key] for key in batch_keys], task)
            with _embedding_cache_lock:
                for key, vector in zip(batch_keys, vectors):
                    stored = (tuple(float(value) for value in vector), provenance)
                    _embeddinggemma_cache[key] = stored
                    _embeddinggemma_cache.move_to_end(key)
                    results[key] = stored
                while len(_embeddinggemma_cache) > _embeddinggemma_cache_limit:
                    _embeddinggemma_cache.popitem(last=False)
    return [(list(results[key][0]), dict(results[key][1])) for key in keys]


def embed_image_with_provenance(image):
    """Encode one PIL image with the documented shared multimodal API."""
    if not settings.LOCAL_FAST_MODE:
        raise RuntimeError("Direct local image embeddings require LOCAL_FAST_MODE.")
    import torch
    task = "retrieval.image"
    provenance = _embeddinggemma_provenance(task)
    image_rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256(
        f"{image_rgb.width}x{image_rgb.height}:RGB:".encode("ascii") + image_rgb.tobytes()
    ).hexdigest()
    key = (provenance["config_id"], task, digest)
    with _inference_slot():
        with _embedding_cache_lock:
            cached = _embeddinggemma_cache.get(key)
            if cached is not None:
                _embeddinggemma_cache.move_to_end(key)
                return list(cached[0]), dict(cached[1])
        model = _get_embeddinggemma_model()
        try:
            vector = model.encode(
                {"image": image_rgb},
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector.shape != (settings.EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"EmbeddingGemma returned invalid image vector with shape {vector.shape}; "
                    f"expected ({settings.EMBEDDING_DIMENSION},) finite values."
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if torch.cuda.is_available() and (
                isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            ):
                torch.cuda.empty_cache()
            raise
        stored = (tuple(float(value) for value in vector), provenance)
        with _embedding_cache_lock:
            _embeddinggemma_cache[key] = stored
            _embeddinggemma_cache.move_to_end(key)
            while len(_embeddinggemma_cache) > _embeddinggemma_cache_limit:
                _embeddinggemma_cache.popitem(last=False)
        return list(stored[0]), dict(stored[1])


def embed_audio_with_provenance(waveform):
    """Embed a decoded mono 16 kHz segment in the shared text/audio space."""
    if not settings.LOCAL_FAST_MODE:
        raise RuntimeError("Direct audio embeddings require LOCAL_FAST_MODE.")
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite mono samples at 16 kHz.")
    provenance = _embeddinggemma_provenance("retrieval.audio")
    with _inference_slot():
        model = _get_embeddinggemma_model()
        vector = model.encode(
            {"audio": samples},
            processing_kwargs={"audio": {"sampling_rate": 16000}},
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if (vector.shape != (settings.EMBEDDING_DIMENSION,)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6):
            raise ValueError("EmbeddingGemma returned an invalid audio vector.")
    return vector.tolist(), provenance


def local_embedding_runtime_stats():
    """Return local model warm-up and runtime details for logging/benchmarks."""
    return {
        "model": settings.LOCAL_EMBEDDING_MODEL_ID,
        "revision": settings.LOCAL_EMBEDDING_MODEL_REVISION,
        "loaded": _local_model is not None,
        "load_seconds": _local_model_load_seconds,
        "device": _local_model_device,
        "precision": _local_model_precision,
    }


def embed_video_with_provenance(frames):
    """Encode a short sequence of sampled frames without a video codec backend."""
    if not settings.LOCAL_FAST_MODE:
        raise RuntimeError("Direct video embeddings require LOCAL_FAST_MODE.")
    if not frames:
        raise ValueError("A video segment must contain frames.")
    provenance = _embeddinggemma_provenance("retrieval.video")
    with _inference_slot():
        model = _get_embeddinggemma_model()
        vector = model.encode(
            {"video": frames},
            processing_kwargs={"video": {
                "fps": None, "max_frames": len(frames), "max_soft_tokens": 70,
                "add_timestamps": False,
            }},
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
        )
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if (vector.shape != (settings.EMBEDDING_DIMENSION,)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-6):
            raise ValueError("EmbeddingGemma returned an invalid video vector.")
    return vector.tolist(), provenance


@contextmanager
def _shared_embedding_key_lock(key):
    """Share per-key locks safely and discard them after the last waiter exits."""
    with _embedding_cache_lock:
        entry = _embedding_key_locks.get(key)
        lock, users = entry if entry is not None else (threading.Lock(), 0)
        _embedding_key_locks[key] = (lock, users + 1)
    acquired = False
    try:
        lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            lock.release()
        with _embedding_cache_lock:
            current_lock, users = _embedding_key_locks[key]
            if current_lock is lock and users == 1:
                del _embedding_key_locks[key]
            else:
                _embedding_key_locks[key] = (current_lock, users - 1)

def _local_semantic_vector(text: str, dim: int = DIMENSION) -> List[float]:
    """
    Zero-dependency deterministic semantic embedding vectorizer.
    Uses token hashing, character n-grams, and TF-IDF weighting projected to L2 unit sphere.
    Produces stable dense vectors with genuine cosine distance behavior.
    """
    vec = np.zeros(dim, dtype=np.float32)
    clean = text.lower().strip()
    if not clean:
        return vec.tolist()

    words = re.findall(r"\w+", clean)
    
    # 1. Word-level hashing with positional damping
    for idx, word in enumerate(words):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        pos = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        weight = (1.0 + math.log(1.0 + len(word))) * (1.0 / (1.0 + 0.05 * min(idx, 20)))
        vec[pos] += sign * weight

    # 2. Substring 3-gram hashing for morphological & typo resilience
    for i in range(len(clean) - 2):
        trigram = clean[i:i+3]
        h = int(hashlib.sha256(trigram.encode("utf-8")).hexdigest(), 16)
        pos = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[pos] += sign * 0.35

    # L2 Normalization
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm

    return vec.tolist()

def _embed_gemini(text: str, dim: int = DIMENSION) -> Optional[List[float]]:
    """
    Generates embedding via Gemini Embedding 2 with configurable output dimensionality.
    """
    if not settings.GEMINI_API_KEY:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        # Truncate content if too long for a single embedding call
        clean_text = text[:8000].strip()
        if not clean_text:
            return None

        response = client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=clean_text,
            config={"output_dimensionality": dim}
        )
        if response and response.embeddings and len(response.embeddings) > 0:
            return list(response.embeddings[0].values)
    except Exception as e:
        logger.warning(f"Gemini embedding API call failed: {e}. Falling back to local vectorizer.")
        return None

    return None

def _embed_jina(text: str, task: str = "retrieval.passage") -> Optional[List[float]]:
    """
    Generates embedding via Jina embeddings API if configured.
    """
    if not settings.JINA_API_KEY:
        return None
    try:
        response = httpx.post(
            JINA_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.JINA_EMBEDDING_MODEL,
                "input": [text or " "],
                "task": task,
                "dimensions": settings.JINA_EMBEDDING_DIMENSION,
            },
            timeout=settings.JINA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if data and data[0].get("embedding"):
            return data[0]["embedding"]
    except Exception as e:
        logger.warning(f"Jina embedding API call failed: {e}")
        return None
    return None

def _embedding_provenance(provider: str, model: str, dimension: int, task: str) -> dict:
    """Describe the vector space that actually produced an embedding."""
    implementation = "local-hash-ngram-v1" if provider == "local" else "provider-api-v1"
    configuration = {
        "provider": provider,
        "model": model,
        "dimension": dimension,
        "implementation": implementation,
    }
    config_id = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**configuration, "config_id": config_id, "task": task}


def embed_text_with_provenance(text: str, task: str = "retrieval.passage"):
    """
    Core embedding interface: embed_text(text) -> vector (768-d).
    Uses Gemini Embedding 2 when configured, with robust local fallback.
    """
    if settings.LOCAL_FAST_MODE:
        return embed_texts_with_provenance([text], task=task)[0]

    provider = settings.EMBEDDING_PROVIDER.lower()
    dimension = settings.EMBEDDING_DIMENSION
    # Hash input text so bounded cache keys do not retain document contents.
    gemini_key_id = hashlib.sha256(settings.GEMINI_API_KEY.encode()).hexdigest()[:16] if settings.GEMINI_API_KEY else ""
    jina_key_id = hashlib.sha256(settings.JINA_API_KEY.encode()).hexdigest()[:16] if settings.JINA_API_KEY else ""
    text_hash = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    # Jina's query and passage task prompts are part of the embedding
    # configuration. Keep them separate in the cache and honor the task passed
    # by search_documents; Gemini uses its shared text embedding interface.
    effective_task = task if provider == "jina" else ""
    key = (
        provider, settings.GEMINI_EMBEDDING_MODEL, settings.JINA_EMBEDDING_MODEL,
        dimension, settings.JINA_EMBEDDING_DIMENSION, effective_task,
        gemini_key_id, jina_key_id, text_hash,
    )

    with _embedding_cache_lock:
        cached = _embedding_cache.get(key)
        if cached is not None:
            _embedding_cache.move_to_end(key)
            if provider == "jina":
                model, actual_dimension = settings.JINA_EMBEDDING_MODEL, settings.JINA_EMBEDDING_DIMENSION
            else:
                model, actual_dimension = settings.GEMINI_EMBEDDING_MODEL, dimension
            return list(cached), _embedding_provenance(provider, model, actual_dimension, task)
        remote_enabled = (provider == "gemini" and bool(settings.GEMINI_API_KEY)) or (
            provider == "jina" and bool(settings.JINA_API_KEY)
        )

    if remote_enabled:
        with _shared_embedding_key_lock(key):
            with _embedding_cache_lock:
                cached = _embedding_cache.get(key)
                if cached is not None:
                    _embedding_cache.move_to_end(key)
                    if provider == "jina":
                        model, actual_dimension = settings.JINA_EMBEDDING_MODEL, settings.JINA_EMBEDDING_DIMENSION
                    else:
                        model, actual_dimension = settings.GEMINI_EMBEDDING_MODEL, dimension
                    return list(cached), _embedding_provenance(provider, model, actual_dimension, task)

            if provider == "gemini":
                vector = _embed_gemini(text or "", dim=dimension)
                valid = vector is not None and len(vector) == dimension
                actual_provider = "gemini"
                actual_model = settings.GEMINI_EMBEDDING_MODEL
                actual_dimension = dimension
            else:
                vector = _embed_jina(text or "", task=effective_task)
                valid = vector is not None and len(vector) == settings.JINA_EMBEDDING_DIMENSION
                actual_provider = "jina"
                actual_model = settings.JINA_EMBEDDING_MODEL
                actual_dimension = settings.JINA_EMBEDDING_DIMENSION
            with _embedding_cache_lock:
                if valid:
                    _embedding_cache[key] = tuple(vector)
                    _embedding_cache.move_to_end(key)
                    while len(_embedding_cache) > _EMBEDDING_CACHE_LIMIT:
                        _embedding_cache.popitem(last=False)
                    return list(vector), _embedding_provenance(
                        actual_provider, actual_model, actual_dimension, task
                    )

    # Keep remote failures retryable and cache only the deterministic local
    # fallback, with a text hash as its key rather than plaintext content.
    local_key = (text_hash, dimension)
    local_lock_key = ("local",) + local_key
    with _shared_embedding_key_lock(local_lock_key):
        with _embedding_cache_lock:
            cached_local = _local_embedding_cache.get(local_key)
            if cached_local is not None:
                _local_embedding_cache.move_to_end(local_key)
                return list(cached_local), _embedding_provenance(
                    "local", "local-hash-ngram-v1", dimension, task
                )
        vector = tuple(_local_semantic_vector(text or "", dim=dimension))
        with _embedding_cache_lock:
            _local_embedding_cache[local_key] = vector
            _local_embedding_cache.move_to_end(local_key)
            while len(_local_embedding_cache) > _EMBEDDING_CACHE_LIMIT:
                _local_embedding_cache.popitem(last=False)
    return list(vector), _embedding_provenance(
        "local", "local-hash-ngram-v1", dimension, task
    )


def embed_text(text: str, task: str = "retrieval.passage") -> List[float]:
    """Backward-compatible vector-only embedding API."""
    return embed_text_with_provenance(text, task=task)[0]

def get_embedding(text: str, task: str = "retrieval.passage") -> List[float]:
    """
    Backward-compatible alias for existing codebase callers.
    """
    return embed_text(text, task=task)


def get_embedding_with_provenance(text: str, task: str = "retrieval.passage"):
    """Return a vector and provenance without changing the legacy get_embedding API."""
    return embed_text_with_provenance(text, task=task)


def expected_embedding_provenance(task: str = "retrieval.passage") -> dict:
    """Describe the configured provider if credentials exist, else local fallback."""
    if settings.LOCAL_FAST_MODE:
        return _embeddinggemma_provenance(task)
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "gemini" and settings.GEMINI_API_KEY:
        return _embedding_provenance(
            provider, settings.GEMINI_EMBEDDING_MODEL, settings.EMBEDDING_DIMENSION, task
        )
    if provider == "jina" and settings.JINA_API_KEY:
        return _embedding_provenance(
            provider, settings.JINA_EMBEDDING_MODEL, settings.JINA_EMBEDDING_DIMENSION, task
        )
    return _embedding_provenance(
        "local", "local-hash-ngram-v1", settings.EMBEDDING_DIMENSION, task
    )


def embedding_signature_is_compatible(stored_provenance, query_provenance=None) -> bool:
    """Only compare vectors whose actual recorded embedding spaces are compatible.

    Empty or malformed legacy provenance is unknown and is therefore excluded
    from semantic similarity. Jina's retrieval.query and retrieval.passage tasks
    are the documented paired modes for one model; task names are retained in
    provenance while other task mismatches are rejected.
    """
    if isinstance(stored_provenance, str):
        if not stored_provenance:
            return False
        try:
            stored = json.loads(stored_provenance)
        except (TypeError, ValueError):
            return False
    else:
        stored = stored_provenance
    if not isinstance(stored, dict):
        return False
    if query_provenance is None:
        query_provenance = expected_embedding_provenance("retrieval.query")
    if isinstance(query_provenance, str):
        try:
            query_provenance = json.loads(query_provenance)
        except (TypeError, ValueError):
            return False
    if not isinstance(query_provenance, dict):
        return False

    fields = ("provider", "model", "dimension", "config_id")
    if any(stored.get(field) != query_provenance.get(field) for field in fields):
        return False
    if stored.get("provider") == "jina":
        return {stored.get("task"), query_provenance.get("task")} <= {
            "retrieval.query", "retrieval.passage"
        }
    if stored.get("provider") == "embeddinggemma":
        return {stored.get("task"), query_provenance.get("task")} <= {
            "retrieval.query", "retrieval.passage", "retrieval.image", "retrieval.audio", "retrieval.video"
        }
    return True

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    Returns float in range [0.0, 1.0].
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
        
    dot = float(np.dot(a, b))
    sim = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, sim))

def rerank_with_jina(query: str, documents: List[str]) -> List[float]:
    """Return provider scores aligned to documents, or [] when unavailable."""
    if not settings.JINA_API_KEY or not documents:
        return []

    try:
        response = httpx.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.JINA_RERANKER_MODEL,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "return_documents": False,
            },
            timeout=settings.JINA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        scores = [0.0] * len(documents)
        for item in response.json().get("results", []):
            index = item.get("index")
            score = item.get("relevance_score")
            if isinstance(index, int) and 0 <= index < len(scores) and isinstance(score, (int, float)):
                scores[index] = max(0.0, min(1.0, float(score)))
        return scores
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return []
