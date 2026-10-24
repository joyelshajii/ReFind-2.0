import io
import json
import logging
import re
import hashlib
import time
import threading
from collections import OrderedDict
from contextlib import nullcontext, ExitStack
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from app.core.config import settings

logger = logging.getLogger("refind.processors.image")
_image_embedding_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="refind-image")


def _embed_image_timed(image):
    from app.services.embedding_service import embed_image_with_provenance
    started = time.perf_counter()
    vector, provenance = embed_image_with_provenance(image)
    return vector, provenance, time.perf_counter() - started

# Bump this when image extraction, OCR, or the Gemini prompt changes. The
# indexer stores this alongside the content hash so a stale extraction is
# refreshed once without relying on text markers in its output.
IMAGE_PROCESSOR_VERSION = 4
_RESIZED_IMAGE_CACHE_LIMIT = 16
_resized_image_cache = OrderedDict()
_resized_image_cache_lock = threading.RLock()
_resized_image_key_locks = {}
_VISION_RESULT_CACHE_LIMIT = 64
_vision_result_cache = OrderedDict()
_vision_result_cache_lock = threading.RLock()
_vision_key_locks = {}

# Register HEIF/HEIC support without decoding a small embedded thumbnail.
_heif_registration_error = None
_heif_registered = False
_heif_registration_lock = threading.Lock()
try:
    import pillow_heif
    pillow_heif.register_heif_opener(thumbnails=False)
    _heif_registered = True
except Exception as error:
    _heif_registration_error = error


def ensure_heif_opener() -> None:
    """Retry registration if pillow-heif was installed after this worker started."""
    global _heif_registered, _heif_registration_error
    if _heif_registered:
        return
    with _heif_registration_lock:
        if _heif_registered:
            return
        try:
            import pillow_heif
            pillow_heif.register_heif_opener(thumbnails=False)
        except Exception as error:
            _heif_registration_error = error
            raise RuntimeError(
                "HEIC/HEIF decoding requires pillow-heif in the backend virtual environment."
            ) from error
        _heif_registered = True
        _heif_registration_error = None

_rapid_ocr = None

def _get_rapid_ocr():
    global _rapid_ocr
    if _rapid_ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr = RapidOCR()
        except Exception as error:
            logger.warning("Optional RapidOCR initialization failed; continuing without this OCR engine: %s", error)
            _rapid_ocr = False
    return _rapid_ocr if _rapid_ocr else None

def _clean_str(val: Any) -> str:
    """Removes null bytes and non-printable noise from strings (e.g. corrupted EXIF tags)."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        try:
            val = val.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    s = str(val).replace("\x00", "").strip()
    return s

def _parse_gemini_vision_output(raw_text: str) -> Dict[str, Any]:
    """Parses structured fields from Gemini Vision response."""
    result = {
        "visual_description": "",
        "detected_text": [],
        "entities": []
    }
    if not raw_text:
        return result

    # Extract Visual Description
    vd_match = re.search(r"Visual Description:\s*(.*?)(?=\n\s*(?:Detected Text|Key Entities):|$)", raw_text, re.DOTALL | re.IGNORECASE)
    if vd_match:
        result["visual_description"] = vd_match.group(1).strip()

    # Extract Detected Text
    dt_match = re.search(r"Detected Text:\s*(.*?)(?=\n\s*Key Entities:|$)", raw_text, re.DOTALL | re.IGNORECASE)
    if dt_match:
        dt_raw = dt_match.group(1).strip()
        if dt_raw.lower() not in ("none", "none.", "n/a", "no text", "no text detected"):
            for line in dt_raw.splitlines():
                clean_line = line.strip().lstrip("-*• ")
                if clean_line and clean_line.lower() != "none":
                    result["detected_text"].append(clean_line)

    # Extract Key Entities
    ke_match = re.search(r"Key Entities:\s*(.*?)$", raw_text, re.DOTALL | re.IGNORECASE)
    if ke_match:
        ke_raw = ke_match.group(1).strip()
        tags = [t.strip().lstrip("-*• ") for t in re.split(r"[,;\n]", ke_raw) if t.strip()]
        for tag in tags:
            if tag and len(tag) < 60 and tag.lower() not in ("none", "n/a"):
                result["entities"].append(tag)

    # Fallback if structure was slightly unaligned
    if not result["visual_description"] and not result["detected_text"]:
        result["visual_description"] = raw_text.strip()

    return result

def _analyze_image_payload_with_gemini(image_bytes: bytes, model_name: str) -> Dict[str, Any]:
    """
    Uses Gemini Vision to produce an in-depth visual description, transcribe all text,
    and identify key scene entities from the prepared JPEG payload.
    """
    if not settings.GEMINI_API_KEY:
        return {}

    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        prompt = """Analyze this image thoroughly for a personal search and memory indexing engine called ReFind.
Provide the output in exactly this format:

Visual Description:
[A detailed 2-4 sentence description of what is depicted in the image: the scene, subjects, people, actions, physical setting, appearance, clothing, colors, objects, diagrams, charts, UI elements, or documents]

Detected Text:
[Verbatim transcription of ALL text visible anywhere in the image, including headers, labels, captions, signs, handwriting, code, or UI text. If none, write: None]

Key Entities:
[Comma-separated list of 3 to 10 specific tags: topics, technologies, people, places, object categories, e.g. Rahul, BLE Beacon, Python, Architecture Diagram]
"""
        models_to_try = list(dict.fromkeys([model_name, "gemini-3.5-flash-lite", "gemini-3.8-flash"]))
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    ]
                )
                if response and response.text:
                    return _parse_gemini_vision_output(response.text)
            except Exception as me:
                logger.warning(f"Gemini vision call failed with model {model_name}: {me}")
                continue

    except Exception as e:
        logger.warning(f"Gemini vision analysis failed: {e}")

    return {}


def _cached_gemini_vision(cache_key: str, model_name: str, image_bytes: bytes) -> str:
    """Reuse identical resized image/model requests without retaining pixels."""
    with _vision_result_cache_lock:
        cached = _vision_result_cache.get(cache_key)
        if cached is not None:
            _vision_result_cache.move_to_end(cache_key)
            return cached
        key_lock = _vision_key_locks.setdefault(cache_key, threading.Lock())

    with key_lock:
        with _vision_result_cache_lock:
            cached = _vision_result_cache.get(cache_key)
            if cached is not None:
                _vision_result_cache.move_to_end(cache_key)
                _vision_key_locks.pop(cache_key, None)
                return cached

        analysis = _analyze_image_payload_with_gemini(image_bytes, model_name)
        result = json.dumps(analysis, ensure_ascii=False)
        with _vision_result_cache_lock:
            # Empty results can indicate a transient provider failure; leave
            # those uncached so a later indexing attempt retries the request.
            if analysis:
                _vision_result_cache[cache_key] = result
                _vision_result_cache.move_to_end(cache_key)
                while len(_vision_result_cache) > _VISION_RESULT_CACHE_LIMIT:
                    _vision_result_cache.popitem(last=False)
            _vision_key_locks.pop(cache_key, None)
        return result


def _analyze_image_with_gemini(img: Image.Image) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Resize/encode once, and reuse a response for identical model inputs."""
    if settings.LOCAL_FAST_MODE or not settings.IMAGE_VISION_ENABLED or not settings.GEMINI_API_KEY:
        return {}, {"vision_preprocess_seconds": 0.0, "vision_seconds": 0.0}
    try:
        preprocess_started = time.perf_counter()
        rgb_image = img.convert("RGB")
        pixel_hash = hashlib.sha256(
            f"{rgb_image.width}x{rgb_image.height}:RGB:".encode("ascii") + rgb_image.tobytes()
        ).hexdigest()
        cache_key = (pixel_hash, rgb_image.width, rgb_image.height, "jpeg-1280-q85-v1")
        with _resized_image_cache_lock:
            image_bytes = _resized_image_cache.get(cache_key)
            if image_bytes is not None:
                _resized_image_cache.move_to_end(cache_key)
            else:
                resize_lock = _resized_image_key_locks.setdefault(cache_key, threading.Lock())
        if image_bytes is None:
            with resize_lock:
                with _resized_image_cache_lock:
                    image_bytes = _resized_image_cache.get(cache_key)
                if image_bytes is None:
                    resized_image = rgb_image.copy()
                    resized_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    resized_image.save(buf, format="JPEG", quality=85)
                    image_bytes = buf.getvalue()
                    with _resized_image_cache_lock:
                        _resized_image_cache[cache_key] = image_bytes
                        _resized_image_cache.move_to_end(cache_key)
                        while len(_resized_image_cache) > _RESIZED_IMAGE_CACHE_LIMIT:
                            _resized_image_cache.popitem(last=False)
                with _resized_image_cache_lock:
                    _resized_image_key_locks.pop(cache_key, None)
        payload_hash = hashlib.sha256(image_bytes).hexdigest()
        vision_preprocess_seconds = time.perf_counter() - preprocess_started
        credential_id = hashlib.sha256(settings.GEMINI_API_KEY.encode()).hexdigest()[:16]
        cache_key = hashlib.sha256(
            f"{payload_hash}:{settings.GEMINI_MODEL}:{credential_id}:prompt-v{IMAGE_PROCESSOR_VERSION}".encode()
        ).hexdigest()
        vision_started = time.perf_counter()
        raw_result = _cached_gemini_vision(cache_key, settings.GEMINI_MODEL, image_bytes)
        return json.loads(raw_result), {
            "vision_preprocess_seconds": vision_preprocess_seconds,
            "vision_seconds": time.perf_counter() - vision_started,
        }
    except Exception as e:
        logger.warning("Gemini vision analysis failed: %s", e)
        return {}, {"vision_preprocess_seconds": 0.0, "vision_seconds": 0.0}

def process_image(file_path: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
    """
    Extracts OCR text, visual description, and image metadata (dimensions, EXIF) from an image.
    Supports PNG, JPG, JPEG, WEBP, GIF, BMP, TIFF, HEIC, and HEIF.
    Integrates Gemini Multimodal Vision with local OCR fallback.
    """
    path = Path(file_path)
    metadata: Dict[str, Any] = {}
    ocr_lines: List[str] = []
    visual_description: str = ""
    discovered_entities: List[Dict[str, str]] = []
    image_embedding = None
    image_embedding_provenance = None
    timings = {
        "decode_metadata_seconds": 0.0,
        "vision_preprocess_seconds": 0.0,
        "vision_seconds": 0.0,
        "ocr_seconds": 0.0,
        "image_embedding_seconds": 0.0,
    }

    decode_started = time.perf_counter()
    try:
        if path.suffix.lower() in {".heic", ".heif"}:
            ensure_heif_opener()
        image_context = Image.open(str(path)) if image is None else nullcontext(image)
        with image_context as img, ExitStack() as pending_work:
            img.load()
            ocr_image = img if img.mode == "RGB" else img.convert("RGB")
            metadata["dimensions"] = f"{img.width}x{img.height}"
            metadata["format"] = img.format or path.suffix.lstrip(".").upper()

            # Extract EXIF metadata if available (sanitized of null bytes)
            try:
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, val in exif_data.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if tag_name in ("DateTime", "DateTimeOriginal", "Make", "Model", "Software", "ImageDescription"):
                            clean_val = _clean_str(val)
                            if clean_val:
                                metadata[tag_name.lower()] = clean_val
                                if tag_name == "ImageDescription" and not visual_description:
                                    visual_description = clean_val
            except Exception as exif_err:
                logger.debug(f"EXIF extraction error for {path.name}: {exif_err}")

            # Check PNG info text chunks if present
            if hasattr(img, "info") and isinstance(img.info, dict):
                for k, v in img.info.items():
                    if isinstance(v, str) and len(v) < 1000:
                        clean_v = _clean_str(v)
                        if clean_v:
                            metadata[f"png_{k.lower()}"] = clean_v
                            if k.lower() in ("description", "comment", "title") and not visual_description:
                                visual_description = clean_v

            timings["decode_metadata_seconds"] = time.perf_counter() - decode_started

            # CPU OCR and GPU vision embedding consume the same read-only
            # decoded pixels independently. Join before closing the image,
            # including on errors; only one image embedding worker exists.
            image_future = None
            if settings.LOCAL_FAST_MODE:
                image_future = _image_embedding_pool.submit(_embed_image_timed, ocr_image)
                pending_work.callback(image_future.result)

            # 1. AI Multimodal Vision Analysis (visual description + entities).
            ai_result, vision_timings = _analyze_image_with_gemini(img)
            timings.update(vision_timings)
            if ai_result:
                if ai_result.get("visual_description"):
                    visual_description = ai_result["visual_description"]
                if ai_result.get("detected_text"):
                    for line in ai_result["detected_text"]:
                        if line not in ocr_lines:
                            ocr_lines.append(line)
                if ai_result.get("entities"):
                    for tag in ai_result["entities"]:
                        discovered_entities.append({
                            "entity_type": "topic",
                            "entity_value": tag
                        })

            # There is no reliable signal that a vision transcription is
            # complete. Supplement mode (default) merges local OCR lines;
            # fallback mode preserves the faster prior behavior.
            run_local_ocr = (
                not ocr_lines or settings.IMAGE_OCR_POLICY == "supplement"
            )
            if run_local_ocr:
                ocr_started = time.perf_counter()
                try:
                    from app.services.ocr_service import ocr_service
                    local_lines = ocr_service.extract_text_from_image(ocr_image)
                    for line in local_lines:
                        clean_line = _clean_str(line)
                        if clean_line and clean_line not in ocr_lines:
                            ocr_lines.append(clean_line)
                except Exception as local_ocr_err:
                    logger.debug("Local OCR skipped for %s: %s", path.name, local_ocr_err)
                finally:
                    timings["ocr_seconds"] = time.perf_counter() - ocr_started

                # RapidOCR fallback also consumes the already-decoded pixels.
                if not ocr_lines and not settings.LOCAL_FAST_MODE:
                    rapid_ocr = _get_rapid_ocr()
                    if rapid_ocr:
                        try:
                            import numpy as np
                            ocr_rgb = np.asarray(img.convert("RGB"))
                            ocr_result, _ = rapid_ocr(ocr_rgb[:, :, ::-1])
                            for item in ocr_result or []:
                                if len(item) >= 2 and str(item[1]).strip():
                                    clean_line = _clean_str(item[1])
                                    if clean_line and clean_line not in ocr_lines:
                                        ocr_lines.append(clean_line)
                        except Exception as rapid_err:
                            logger.debug("RapidOCR skipped for %s: %s", path.name, rapid_err)

            # In local-fast mode, reuse the decoded image for its direct
            # multimodal vector. Gemini remains available in normal mode.
            if settings.LOCAL_FAST_MODE:
                image_embedding, image_embedding_provenance, image_seconds = image_future.result()
                timings["image_embedding_seconds"] = image_seconds

    except Exception as e:
        logger.error(f"Error opening image {file_path}: {e}")
        raise

    # Check companion metadata file if exists (e.g. for demo dataset: IMG_2384.png.meta.json)
    companion_meta = path.with_suffix(path.suffix + ".meta.json")
    if companion_meta.exists():
        try:
            with open(companion_meta, "r", encoding="utf-8") as f:
                comp = json.load(f)
                if "detected_text" in comp:
                    for item in comp["detected_text"]:
                        clean_item = _clean_str(item)
                        if clean_item and clean_item not in ocr_lines:
                            ocr_lines.append(clean_item)
                if "visual_description" in comp and not visual_description:
                    visual_description = _clean_str(comp["visual_description"])
                if "metadata" in comp:
                    for k, v in comp["metadata"].items():
                        clean_v = _clean_str(v)
                        if clean_v:
                            metadata[k] = clean_v
        except Exception as ce:
            logger.debug(f"Companion metadata read error: {ce}")

    # Include the filename/path
    metadata["filename"] = path.name
    metadata["source_path"] = str(path.parent)

    # Construct the rich searchable text representation
    text_blocks = []
    if visual_description:
        text_blocks.append(f"Visual Description:\n{visual_description}")
    if ocr_lines:
        text_blocks.append("Detected Text in Image:\n" + "\n".join(ocr_lines))
    if metadata:
        metadata_lines = [
            f"{key.replace('_', ' ').title()}: {value}"
            for key, value in metadata.items()
            if value is not None and str(value).strip()
        ]
        if metadata_lines:
            text_blocks.append("Image Metadata:\n" + "\n".join(metadata_lines))

    extracted_text = "\n\n".join(text_blocks)
    if not extracted_text:
        extracted_text = f"Image file: {path.name}, Format: {metadata.get('format', 'Image')}, Dimensions: {metadata.get('dimensions', 'unknown')}"

    chunk = {
        "content": extracted_text,
        "page_number": 1,
        "slide_number": None
    }

    return {
        "extracted_text": extracted_text,
        "chunks": [chunk],
        "metadata": metadata,
        "entities": discovered_entities,
        "timings": timings,
        "ocr_text": "\n".join(ocr_lines),
        "image_embedding": image_embedding,
        "image_embedding_provenance": image_embedding_provenance,
    }
