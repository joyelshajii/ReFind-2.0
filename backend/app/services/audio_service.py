"""Stream recordings into bounded, overlapping segments for local retrieval."""
from itertools import chain

import numpy as np

AUDIO_PROCESSOR_VERSION = 1
SAMPLE_RATE = 16000
SEGMENT_SECONDS = 25
OVERLAP_SECONDS = 5


def iter_audio_segments(file_path):
    """Decode the complete first audio stream, including the final partial window."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "Audio decoding is unavailable. Install backend requirements (PyAV) and reindex."
        ) from error

    window = SEGMENT_SECONDS * SAMPLE_RATE
    step = (SEGMENT_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE
    pending = np.empty(0, dtype=np.float32)
    offset = 0
    last_end = 0
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.audio:
                raise ValueError("The file has no decodable audio stream.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            for frame in chain(container.decode(audio=0), [None]):
                for decoded in resampler.resample(frame):
                    samples = decoded.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    pending = np.concatenate((pending, samples))
                    while pending.size >= window:
                        last_end = offset + window
                        yield offset / SAMPLE_RATE, last_end / SAMPLE_RATE, pending[:window].copy()
                        pending = pending[step:]
                        offset += step
            # Do not emit an overlap-only duplicate when the file ends on a window.
            if pending.size and offset + pending.size > last_end:
                yield offset / SAMPLE_RATE, (offset + pending.size) / SAMPLE_RATE, pending.copy()
    except av.FFmpegError as error:
        raise RuntimeError(f"Audio decoding failed for {file_path.name}: {error}") from error
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        raise RuntimeError(f"Audio decoding failed for {file_path.name}: {error}") from error


def timestamp(seconds):
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def build_audio_chunks(file_path):
    from app.services.embedding_service import embed_audio_with_provenance

    chunks = []
    for start, end, waveform in iter_audio_segments(file_path):
        vector, provenance = embed_audio_with_provenance(waveform)
        chunks.append({
            "content": f"Audio {timestamp(start)}–{timestamp(end)}",
            "page_number": None,
            "slide_number": None,
            "_embedding": vector,
            "_embedding_provenance": {
                **provenance, "start_seconds": start, "end_seconds": end,
            },
        })
    if not chunks:
        raise ValueError("The recording contains no audio samples to index.")
    return chunks
