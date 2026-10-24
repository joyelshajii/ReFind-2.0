from pathlib import Path
from typing import Dict, Any, Callable
from app.processors.pdf_processor import process_pdf
from app.processors.docx_processor import process_docx
from app.processors.pptx_processor import process_pptx
from app.processors.txt_processor import process_txt
from app.processors.image_processor import process_image
from app.processors.audio_processor import process_audio
from app.processors.video_processor import process_video

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "txt",
    # Images
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".heic": "heic",
    ".heif": "heif",
    ".jfif": "jpeg",
    # Audio
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".flac": "flac",
    ".aac": "aac",
    ".ogg": "ogg",
    ".wma": "wma",
    ".opus": "opus",
    # Video
    ".mp4": "mp4",
    ".mkv": "mkv",
    ".avi": "avi",
    ".mov": "mov",
    ".wmv": "wmv",
    ".webm": "webm",
    ".flv": "flv",
    ".m4v": "m4v",
}

PROCESSORS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "pdf": process_pdf,
    "docx": process_docx,
    "pptx": process_pptx,
    "txt": process_txt,
    "png": process_image,
    "jpg": process_image,
    "jpeg": process_image,
    "webp": process_image,
    "gif": process_image,
    "bmp": process_image,
    "tiff": process_image,
    "heic": process_image,
    "heif": process_image,
    "mp3": process_audio,
    "wav": process_audio,
    "m4a": process_audio,
    "flac": process_audio,
    "aac": process_audio,
    "ogg": process_audio,
    "wma": process_audio,
    "opus": process_audio,
    "mp4": process_video,
    "mkv": process_video,
    "avi": process_video,
    "mov": process_video,
    "wmv": process_video,
    "webm": process_video,
    "flv": process_video,
    "m4v": process_video,
}

AUDIO_TYPES = {"mp3", "wav", "m4a", "flac", "aac", "ogg", "wma", "opus"}
VIDEO_TYPES = {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "m4v"}
MEDIA_TYPES = AUDIO_TYPES | VIDEO_TYPES

def get_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, "unknown")

def process_file(file_path: str) -> Dict[str, Any]:
    file_type = get_file_type(file_path)
    processor = PROCESSORS.get(file_type)
    if not processor:
        raise ValueError(f"Unsupported file type: {Path(file_path).suffix or '(no extension)'}")
    return processor(file_path)
