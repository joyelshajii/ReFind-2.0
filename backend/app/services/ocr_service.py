import logging
import hashlib
import threading
from collections import OrderedDict
from typing import List, Union
from pathlib import Path
from PIL import Image
from app.core.config import settings

logger = logging.getLogger("refind.services.ocr")

_OCR_CACHE_LIMIT = 128
_OCR_PROFILE = "paddle-angle-en+tesseract-fallback-v1"
_LOCAL_FAST_OCR_PROFILE = "rapidocr-onnxruntime-bundled+tesseract-fallback-v1"

class OCRService:
    def __init__(self):
        self._paddleocr = None
        self._paddleocr_failed = False
        self._rapidocr = None
        self._rapidocr_failed = False
        self._result_cache = OrderedDict()
        self._cache_lock = threading.RLock()
        self._key_locks = {}
        self._model_lock = threading.Lock()

    def _get_paddleocr(self):
        if self._paddleocr is None and not self._paddleocr_failed:
            with self._model_lock:
                if self._paddleocr is None and not self._paddleocr_failed:
                    try:
                        from paddleocr import PaddleOCR
                        # Initialize local PaddleOCR once for English OCR.
                        self._paddleocr = PaddleOCR(use_angle_cls=True, lang="en")
                        logger.info("PaddleOCR initialized successfully.")
                    except Exception as e:
                        logger.warning("PaddleOCR initialization failed: %s. Will fall back to pytesseract / other methods.", e)
                        self._paddleocr_failed = True
        return self._paddleocr

    def _get_rapidocr(self):
        if self._rapidocr is None and not self._rapidocr_failed:
            with self._model_lock:
                if self._rapidocr is None and not self._rapidocr_failed:
                    try:
                        from rapidocr_onnxruntime import RapidOCR
                        self._rapidocr = RapidOCR()
                        logger.info("Bundled local RapidOCR initialized successfully.")
                    except Exception as error:
                        self._rapidocr_failed = True
                        logger.warning("Local RapidOCR initialization failed: %s", error)
        return self._rapidocr

    def extract_text_from_image(self, image_path: Union[str, Image.Image]) -> List[str]:
        """
        Extracts text locally from an image file (PNG, JPG, JPEG, screenshots).
        Uses PaddleOCR first, then falls back to pytesseract if needed.
        Never sends user documents to Gemini for text extraction.
        """
        lines: List[str] = []
        path = Path(image_path) if isinstance(image_path, (str, Path)) else None
        if path is not None and not path.exists():
            return lines

        try:
            if isinstance(image_path, Image.Image):
                # The image processor passes its already-open image so OCR does
                # not decode the same file a second time.
                image = image_path if image_path.mode == "RGB" else image_path.convert("RGB")
            else:
                image = Image.open(str(path)).convert("RGB")
            image_bytes = image.tobytes()
            profile = _LOCAL_FAST_OCR_PROFILE if settings.LOCAL_FAST_MODE else _OCR_PROFILE
            cache_key = hashlib.sha256(
                f"{profile}:{image.width}x{image.height}:RGB:".encode("ascii") + image_bytes
            ).hexdigest()
        except Exception as error:
            logger.debug("Unable to prepare image for OCR: %s", error)
            return lines

        with self._cache_lock:
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)
                return list(cached)
            key_lock = self._key_locks.setdefault(cache_key, threading.Lock())

        # Calls for a duplicate image share work. Distinct images can still be
        # processed concurrently instead of holding a global lock over OCR.
        with key_lock:
            with self._cache_lock:
                cached = self._result_cache.get(cache_key)
                if cached is not None:
                    self._result_cache.move_to_end(cache_key)
                    return list(cached)

            # Local-fast mode uses the ONNX weights bundled with the installed
            # RapidOCR package. PaddleOCR may download weights on first use, so
            # it is intentionally excluded from offline indexing.
            if settings.LOCAL_FAST_MODE:
                ocr = self._get_rapidocr()
                if ocr is not None:
                    try:
                        import numpy as np
                        image_bgr = np.asarray(image)[:, :, ::-1]
                        result, _ = ocr(image_bgr)
                        for item in result or []:
                            if len(item) > 1 and str(item[1]).strip():
                                text = str(item[1]).strip()
                                if text not in lines:
                                    lines.append(text)
                    except Exception as error:
                        logger.warning("Bundled RapidOCR failed on %s: %s", path.name if path else "image", error)
            # 1. Legacy mode tries PaddleOCR on the decoded pixels.
            else:
                ocr = self._get_paddleocr()
            if not settings.LOCAL_FAST_MODE and ocr is not None:
                try:
                    import numpy as np
                    # Match PaddleOCR's previous path-based OpenCV decode
                    # (BGR array order) while reusing the already-open pixels.
                    image_bgr = np.asarray(image)[:, :, ::-1]
                    result = ocr.ocr(image_bgr, cls=True)
                    if result:
                        for line_group in result:
                            if line_group:
                                for line_info in line_group:
                                    if len(line_info) > 1 and len(line_info[1]) > 0:
                                        text = line_info[1][0].strip()
                                        if text and text not in lines:
                                            lines.append(text)
                except Exception as pe:
                    logger.warning("PaddleOCR extraction failed on %s: %s", path.name if path else "image", pe)

            # Local Tesseract is a fallback only when the executable is
            # installed. It is never downloaded or configured automatically.
            if not lines:
                try:
                    import pytesseract
                    raw_text = pytesseract.image_to_string(image)
                    if raw_text and raw_text.strip():
                        lines = list(dict.fromkeys(
                            line.strip() for line in raw_text.splitlines() if line.strip()
                        ))
                except Exception as te:
                    logger.debug("Pytesseract fallback skipped/failed: %s", te)

            with self._cache_lock:
                self._result_cache[cache_key] = tuple(lines)
                self._result_cache.move_to_end(cache_key)
                while len(self._result_cache) > _OCR_CACHE_LIMIT:
                    self._result_cache.popitem(last=False)
                self._key_locks.pop(cache_key, None)

        return lines

    def extract_text_from_scanned_pdf_page(self, page_pixmap) -> str:
        """
        Extracts OCR text from a rendered PyMuPDF Pixmap.
        """
        try:
            # PyMuPDF already provides lossless raster pixels. Avoid PNG
            # compression followed immediately by PNG decoding for every page.
            if hasattr(page_pixmap, "samples"):
                mode = "RGBA" if page_pixmap.alpha else ("L" if page_pixmap.n == 1 else "RGB")
                img = Image.frombytes(mode, (page_pixmap.width, page_pixmap.height), page_pixmap.samples)
            else:
                # Compatibility with pixmap adapters exposing only encoded bytes.
                import io
                img = Image.open(io.BytesIO(page_pixmap.tobytes("png")))
            return "\n".join(self.extract_text_from_image(img))
        except Exception as e:
            logger.debug(f"Pixmap OCR failed: {e}")
            return ""

ocr_service = OCRService()
