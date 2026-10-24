"""Media metadata and subtitle decoding using the bundled FFmpeg libraries."""
import re


def probe_local_media(file_path):
    import av

    with av.open(str(file_path)) as container:
        streams = []
        for stream in container.streams:
            codec = stream.codec_context
            info = {
                "codec_type": stream.type, "codec_name": getattr(codec, "name", ""),
                "tags": dict(stream.metadata),
            }
            if stream.duration is not None and stream.time_base is not None:
                info["duration"] = float(stream.duration * stream.time_base)
            if stream.type == "video":
                info.update(width=codec.width, height=codec.height)
            elif stream.type == "audio":
                info.update(sample_rate=codec.sample_rate, channels=codec.channels)
            streams.append(info)
        return {"format": {
            "tags": dict(container.metadata),
            "duration": container.duration / av.time_base if container.duration else 0,
            "bit_rate": container.bit_rate,
        }, "streams": streams}


def read_local_subtitles(file_path):
    """Yield language, start time, text for text subtitle packets (not bitmap OCR)."""
    import av

    with av.open(str(file_path)) as container:
        streams = [stream for stream in container.streams.subtitles
                   if stream.codec_context.name not in {
                       "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub",
                   }]
        if not streams:
            return
        for packet in container.demux(streams):
            if packet.pts is None:
                continue
            start = float(packet.pts * packet.time_base)
            language = packet.stream.metadata.get("language", "UND").upper()
            for subtitle in packet.decode():
                # PyAV 16's dialogue getter can reject non-ASCII ASS bytes.
                # FFmpeg's ASS event has eight comma-separated header fields;
                # split only those fields so commas in the dialogue survive.
                ass = getattr(subtitle, "ass", None)
                if ass:
                    if isinstance(ass, bytes):
                        ass = ass.decode("utf-8", errors="replace")
                    fields = ass.split(",", 8)
                    content = fields[8] if len(fields) == 9 else ""
                else:
                    content = getattr(subtitle, "text", None)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                if content:
                    content = re.sub(r"\{[^}]*\}|<[^>]+>", "", content)
                    content = content.replace("\\N", " ").replace("\\n", " ").strip()
                    if content:
                        yield language, start, content
