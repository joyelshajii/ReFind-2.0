"""Local visual and sound retrieval for videos, including those without subtitles."""
from app.services.audio_service import build_audio_chunks, timestamp

VIDEO_PROCESSOR_VERSION = 1
VIDEO_SEGMENT_SECONDS = 8
VIDEO_OVERLAP_SECONDS = 2
VIDEO_FRAME_INTERVAL = 1.0


def iter_video_segments(file_path):
    """Sample one frame per second throughout the first video stream."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError("Video decoding is unavailable. Install backend requirements (PyAV) and reindex.") from error

    frames = []
    start = 0.0
    next_sample = 0.0
    last_end = 0.0
    last_emitted_end = 0.0
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.video:
                raise ValueError("The file has no decodable video stream.")
            stream = container.streams.video[0]
            rate = float(stream.average_rate or 25)
            origin = None
            for index, frame in enumerate(container.decode(video=0)):
                frame_time = float(frame.time) if frame.time is not None else index / rate
                if origin is None:
                    origin = frame_time
                elapsed = max(0.0, frame_time - origin)
                last_end = elapsed + 1.0 / rate
                if elapsed < next_sample:
                    continue
                # Keep frame copies small; the encoder handles final resizing.
                image = frame.to_image().convert("RGB")
                image.thumbnail((448, 448))
                frames.append((elapsed, image))
                next_sample = elapsed + VIDEO_FRAME_INTERVAL
                if len(frames) >= VIDEO_SEGMENT_SECONDS:
                    last_emitted_end = elapsed + VIDEO_FRAME_INTERVAL
                    yield start, last_emitted_end, [image for _, image in frames]
                    frames = frames[-VIDEO_OVERLAP_SECONDS:]
                    start = frames[0][0]
            # Preserve new frames in the final partial segment, but do not
            # re-embed the overlap by itself after a complete segment.
            if frames and (not last_emitted_end or frames[-1][0] >= last_emitted_end):
                yield start, last_end, [image for _, image in frames]
    except av.FFmpegError as error:
        raise RuntimeError(f"Video decoding failed for {file_path.name}: {error}") from error
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        raise RuntimeError(f"Video decoding failed for {file_path.name}: {error}") from error


def build_video_chunks(file_path):
    from app.services.embedding_service import embed_video_with_provenance

    chunks = []
    for start, end, frames in iter_video_segments(file_path):
        vector, provenance = embed_video_with_provenance(frames)
        chunks.append({
            "content": f"Video {timestamp(start)}–{timestamp(end)}",
            "page_number": None, "slide_number": None,
            "_embedding": vector,
            "_embedding_provenance": {
                **provenance, "start_seconds": start, "end_seconds": end,
            },
        })
    if not chunks:
        raise ValueError("The video contains no frames to index.")
    import av
    with av.open(str(file_path)) as container:
        has_audio = bool(container.streams.audio)
    if has_audio:
        # Missing subtitles never prevent speech retrieval. Silent videos
        # simply use their visual chunks; audio decode failures remain visible.
        chunks.extend(build_audio_chunks(file_path))
    return chunks
