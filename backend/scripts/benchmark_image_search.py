"""Repeatable, small image-search benchmark for Gemini text and EmbeddingGemma 2.

By default the script uses generated fixtures and only loads EmbeddingGemma 2
from the local Hugging Face cache. Pass --allow-download to opt into fetching
the model weights. The built-in pictures are deliberately illustrative test
art, not a substitute for a representative real-photo corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

LOGGER = logging.getLogger("refind.image_benchmark")


def _fixture_set(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create six small, labelled fixtures with text and visual hard negatives."""
    root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    def save_scene(name: str, background: tuple[int, int, int], scene: str) -> Path:
        image = Image.new("RGB", (640, 420), background)
        draw = ImageDraw.Draw(image)
        # Shared mountain/lake composition makes the first pair visually
        # similar while their salient object and meaning differ.
        draw.polygon([(0, 210), (130, 35), (260, 210)], fill=(97, 116, 139))
        draw.polygon([(180, 210), (360, 55), (540, 210)], fill=(74, 93, 119))
        draw.rectangle((0, 210, 640, 420), fill=(61, 122, 151))
        draw.line((0, 224, 640, 224), fill=(211, 225, 219), width=4)
        if scene == "canoe":
            draw.polygon([(210, 305), (400, 305), (365, 332), (245, 332)], fill=(182, 51, 39))
            draw.ellipse((292, 276, 320, 310), fill=(219, 181, 143))
            draw.line((275, 302, 345, 351), fill=(39, 37, 32), width=5)
        elif scene == "tent":
            draw.polygon([(225, 337), (330, 255), (435, 337)], fill=(213, 127, 43))
            draw.polygon([(330, 255), (330, 337), (435, 337)], fill=(165, 78, 33))
            draw.polygon([(305, 337), (330, 296), (354, 337)], fill=(50, 45, 37))
        elif scene == "forest":
            for x, height in [(55, 170), (115, 210), (510, 185), (575, 230)]:
                draw.rectangle((x - 7, 215, x + 7, 300), fill=(83, 63, 43))
                draw.polygon([(x, 215 - height // 2), (x - 46, 245), (x + 46, 245)], fill=(36, 84, 57))
                draw.polygon([(x, 225 - height // 3), (x - 36, 275), (x + 36, 275)], fill=(47, 103, 65))
        path = root / name
        image.save(path)
        return path

    canoe = save_scene("red_canoe_lake.png", (173, 205, 226), "canoe")
    tent = save_scene("orange_tent_lake.png", (173, 205, 226), "tent")
    forest = save_scene("pine_forest_lake.png", (183, 208, 225), "forest")

    invoice = Image.new("RGB", (640, 420), (248, 248, 244))
    draw = ImageDraw.Draw(invoice)
    draw.rectangle((28, 24, 612, 396), outline=(45, 52, 61), width=3)
    draw.text((56, 54), "NORTHWIND SUPPLY", fill=(28, 50, 76), font=font)
    draw.text((56, 104), "INVOICE 0427", fill=(28, 50, 76), font=font)
    draw.text((56, 155), "DATE: APRIL 18 2026", fill=(35, 35, 35), font=font)
    draw.line((55, 203, 585, 203), fill=(120, 130, 140), width=2)
    draw.text((56, 235), "Camera equipment     1,280.00", fill=(35, 35, 35), font=font)
    draw.text((56, 292), "TOTAL USD 1,480.00", fill=(35, 35, 35), font=font)
    invoice_path = root / "invoice_screenshot.png"
    invoice.save(invoice_path)

    diagram = Image.new("RGB", (640, 420), (28, 34, 44))
    draw = ImageDraw.Draw(diagram)
    draw.text((35, 30), "SOLAR HOME ENERGY", fill=(242, 244, 247), font=font)
    boxes = [(40, "SOLAR PANELS"), (230, "INVERTER"), (420, "BATTERY")]
    for x, label in boxes:
        draw.rounded_rectangle((x, 125, x + 160, 225), radius=9, fill=(47, 58, 72), outline=(71, 194, 155), width=3)
        draw.text((x + 22, 166), label, fill=(244, 246, 248), font=font)
    draw.line((200, 175, 228, 175), fill=(237, 207, 95), width=5)
    draw.line((390, 175, 418, 175), fill=(237, 207, 95), width=5)
    draw.text((42, 285), "GRID METER  |  MQTT TELEMETRY", fill=(188, 199, 209), font=font)
    diagram_path = root / "solar_energy_diagram.png"
    diagram.save(diagram_path)

    cat = Image.new("RGB", (640, 420), (204, 190, 174))
    draw = ImageDraw.Draw(cat)
    draw.rectangle((0, 280, 640, 420), fill=(102, 79, 66))
    draw.rounded_rectangle((190, 165, 440, 350), radius=75, fill=(130, 132, 137))
    draw.polygon([(220, 190), (235, 105), (290, 176)], fill=(130, 132, 137))
    draw.polygon([(340, 176), (397, 104), (413, 198)], fill=(130, 132, 137))
    draw.ellipse((265, 226, 278, 240), fill=(25, 25, 25))
    draw.ellipse((346, 226, 359, 240), fill=(25, 25, 25))
    draw.polygon([(305, 251), (326, 251), (315, 263)], fill=(171, 112, 118))
    cat_path = root / "gray_cat_sofa.png"
    cat.save(cat_path)

    dog = Image.new("RGB", (640, 420), (173, 208, 231))
    draw = ImageDraw.Draw(dog)
    draw.rectangle((0, 260, 640, 420), fill=(92, 139, 74))
    draw.ellipse((230, 160, 435, 315), fill=(39, 43, 47))
    draw.ellipse((180, 154, 290, 245), fill=(39, 43, 47))
    draw.ellipse((172, 145, 206, 212), fill=(28, 30, 34))
    draw.rectangle((258, 280, 284, 360), fill=(39, 43, 47))
    draw.rectangle((375, 280, 401, 360), fill=(39, 43, 47))
    dog_path = root / "black_dog_field.png"
    dog.save(dog_path)

    records = [
        {"path": canoe, "expected": "red_canoe_lake.png"},
        {"path": tent, "expected": "orange_tent_lake.png"},
        {"path": forest, "expected": "pine_forest_lake.png"},
        {"path": invoice_path, "expected": "invoice_screenshot.png"},
        {"path": diagram_path, "expected": "solar_energy_diagram.png"},
        {"path": cat_path, "expected": "gray_cat_sofa.png"},
        {"path": dog_path, "expected": "black_dog_field.png"},
    ]
    queries = [
        {"query": "a red canoe floating on a mountain lake", "expected": "red_canoe_lake.png"},
        {"query": "an orange camping tent beside a mountain lake", "expected": "orange_tent_lake.png"},
        {"query": "pine trees beside a mountain lake", "expected": "pine_forest_lake.png"},
        {"query": "invoice number 0427 for camera equipment", "expected": "invoice_screenshot.png", "ocr_exact": "0427"},
        {"query": "solar home energy diagram with battery and MQTT telemetry", "expected": "solar_energy_diagram.png", "ocr_exact": "MQTT"},
        {"query": "a gray cat sitting on a sofa", "expected": "gray_cat_sofa.png"},
        {"query": "a black dog standing in a grassy field", "expected": "black_dog_field.png"},
    ]
    return records, queries


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load local image IDs and query/target labels from a JSON manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read benchmark manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object.")
    image_items = manifest.get("images")
    queries = manifest.get("queries")
    if not isinstance(image_items, list) or not isinstance(queries, list):
        raise ValueError("Manifest must include 'images' and 'queries' arrays.")

    records = []
    seen_ids = set()
    for index, item in enumerate(image_items):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
            raise ValueError(f"Manifest image {index + 1} must include a non-empty path.")
        image_path = Path(item["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise ValueError(f"Manifest image does not exist or is not a file: {image_path}")
        image_id = item.get("id") or image_path.name
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"Manifest image {index + 1} has an invalid id.")
        if image_id in seen_ids:
            raise ValueError(f"Manifest image id is duplicated: {image_id}")
        try:
            with Image.open(image_path) as source:
                source.load()
        except Exception as error:
            raise ValueError(f"Manifest image is not a readable image ({image_path}): {error}") from error
        seen_ids.add(image_id)
        records.append({"path": image_path.resolve(), "expected": image_id})

    image_ids = {record["expected"] for record in records}
    if not records or not queries:
        raise ValueError("Manifest must include at least one image and one query.")
    for index, item in enumerate(queries):
        if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"Manifest query {index + 1} must include a non-empty query string.")
        if item.get("expected") not in image_ids:
            raise ValueError(f"Manifest query {index + 1} has an expected id that does not match an image id.")
        if item.get("ocr_exact") is not None and not isinstance(item["ocr_exact"], str):
            raise ValueError(f"Manifest query {index + 1} has an invalid ocr_exact value.")
    return records, queries


def _memory_mb(peak: bool = False) -> float | None:
    if not peak:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass
    try:
        if sys.platform == "win32":
            import ctypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong),
                            ("peak", ctypes.c_size_t), ("working", ctypes.c_size_t),
                            ("other", ctypes.c_size_t * 6)]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                value = counters.peak if peak else counters.working
                return round(value / (1024 * 1024), 2)
        elif peak:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return round(value / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
        elif sys.platform.startswith("linux"):
            with open("/proc/self/statm", "r", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return round(resident_pages * __import__("os").sysconf("SC_PAGE_SIZE") / (1024 * 1024), 2)
    except Exception:
        return None
    return None


def _rank_metrics(vectors: dict[str, Any], queries: list[dict[str, str]], encode_query,
                  ocr_texts: dict[str, str]):
    import numpy as np

    ranks = []
    exact_ocr = []
    latencies = []
    for item in queries:
        started = time.perf_counter()
        query_vector = np.asarray(encode_query(item["query"]), dtype=np.float32)
        scored = []
        for name, vector in vectors.items():
            other = np.asarray(vector, dtype=np.float32)
            denom = float(np.linalg.norm(query_vector) * np.linalg.norm(other))
            score = float(np.dot(query_vector, other) / denom) if denom else 0.0
            scored.append((score, name))
        scored.sort(reverse=True)
        latencies.append(time.perf_counter() - started)
        rank = next((i + 1 for i, (_, name) in enumerate(scored) if name == item["expected"]), None)
        ranks.append(rank)
        exact = item.get("ocr_exact")
        if exact:
            matches = [name for name, text in ocr_texts.items() if exact.casefold() in text.casefold()]
            exact_ocr.append({
                "query": item["query"], "target": item["expected"], "term": exact,
                "found": exact.casefold() in ocr_texts.get(item["expected"], "").casefold(),
                "matching_files": matches,
                "retrieved_expected": item["expected"] in matches,
            })
    valid_ranks = [rank for rank in ranks if rank is not None]
    query_count = len(queries)
    return {
        "top1_accuracy": round(sum(rank == 1 for rank in valid_ranks) / max(query_count, 1), 4),
        "mrr": round(sum(1 / rank for rank in valid_ranks) / max(query_count, 1), 4),
        "expected_ranks": ranks,
        "missing_expected_items": sum(rank is None for rank in ranks),
        "query_embedding_plus_vector_search_latency_ms_mean": round(statistics.mean(latencies) * 1000, 3) if latencies else 0,
        "query_embedding_plus_vector_search_latency_ms_p95": round(sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] * 1000, 3) if latencies else 0,
        "exact_ocr": exact_ocr,
    }


def run_benchmark(allow_download: bool, manifest_path: Path | None = None) -> dict[str, Any]:
    try:
        import numpy as np
        from app.core.config import settings
        from app.processors.image_processor import process_image
        from app.services.embedding_service import get_embedding
    except Exception as error:
        return {"status": "unavailable", "reason": f"ReFind backend dependencies unavailable: {error}"}

    with tempfile.TemporaryDirectory(prefix="refind_image_benchmark_") as fixture_dir:
        if manifest_path:
            records, queries = _load_manifest(manifest_path.resolve())
            dataset_name = f"labelled local image set, {len(records)} images"
        else:
            records, queries = _fixture_set(Path(fixture_dir))
            dataset_name = "synthetic generated illustrations and text images, 7 images; not a real-image relevance benchmark"
        image_paths = {record["expected"]: record["path"] for record in records}
        images = {}
        for record in records:
            with Image.open(record["path"]) as source:
                images[record["expected"]] = source.convert("RGB")
        result: dict[str, Any] = {
            "status": "measured",
            "dataset": dataset_name,
            "query_count": len(queries),
            "methods": {},
            "limitations": [],
        }
        texts: dict[str, str] = {}
        ocr_texts: dict[str, str] = {}

        # The deployed path: Gemini vision/OCR, then the configured ReFind text embedder.
        legacy_started = time.perf_counter()
        legacy_vectors = {}
        legacy_stages = {"image_processing_seconds": 0.0, "embedding_seconds": 0.0,
                         "vision_seconds": 0.0, "vision_preprocess_seconds": 0.0, "ocr_seconds": 0.0}
        legacy_failures = 0
        memory_before = _memory_mb()
        for name, image in images.items():
            try:
                stage_started = time.perf_counter()
                processed = process_image(str(image_paths[name]), image=image)
                legacy_stages["image_processing_seconds"] += time.perf_counter() - stage_started
                for stage in ("vision_seconds", "vision_preprocess_seconds", "ocr_seconds"):
                    legacy_stages[stage] += float(processed.get("timings", {}).get(stage, 0.0))
                text = processed.get("extracted_text", "")
                texts[name] = text
                ocr_texts[name] = processed.get("ocr_text", "")
                stage_started = time.perf_counter()
                legacy_vectors[name] = get_embedding(text[:1500] if text else name)
                legacy_stages["embedding_seconds"] += time.perf_counter() - stage_started
            except Exception:
                LOGGER.exception("Legacy pipeline failed for %s", name)
                legacy_failures += 1
        legacy_elapsed = time.perf_counter() - legacy_started
        if legacy_vectors:
            legacy_metrics = _rank_metrics(
                legacy_vectors, queries,
                lambda query: get_embedding(query, task="retrieval.query"),
                ocr_texts,
            )
        else:
            legacy_metrics = {"status": "unavailable", "reason": "No legacy image vectors were produced."}
        result["methods"]["gemini_description_plus_text_embedding"] = {
            "status": "measured" if len(legacy_vectors) == len(images) else "partial" if legacy_vectors else "failed",
            "configured_vision_model": settings.GEMINI_MODEL,
            "configured_embedding_provider": settings.EMBEDDING_PROVIDER,
            "vision_enabled": bool(settings.IMAGE_VISION_ENABLED and settings.GEMINI_API_KEY),
            "images_indexed": len(legacy_vectors), "failures": legacy_failures,
            "index_seconds_total": round(legacy_elapsed, 3),
            "index_seconds_per_image": round(legacy_elapsed / max(len(legacy_vectors), 1), 3),
            "stage_seconds": {key: round(value, 3) for key, value in legacy_stages.items()},
            "memory_start_mb": memory_before, "memory_end_mb": _memory_mb(),
            "process_peak_memory_mb": _memory_mb(peak=True),
            **legacy_metrics,
        }

        # Direct image -> local EmbeddingGemma 2, with matching text query vectors.
        direct_started = time.perf_counter()
        direct_vectors = {}
        model_load_seconds = None
        memory_after_model_load = None
        direct_failure = None
        direct_stages = {"image_embedding_seconds": 0.0}
        direct_memory_before = _memory_mb()
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            model_load_started = time.perf_counter()
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            model = SentenceTransformer(
                "google/embeddinggemma-2",
                config_kwargs={"audio_config": None},
                model_kwargs={"torch_dtype": dtype},
                local_files_only=not allow_download,
            )
            model_load_seconds = time.perf_counter() - model_load_started
            memory_after_model_load = _memory_mb()
            for name, image in images.items():
                stage_started = time.perf_counter()
                direct_vectors[name] = model.encode(
                    {"image": image}, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                direct_stages["image_embedding_seconds"] += time.perf_counter() - stage_started

            direct_index_elapsed = time.perf_counter() - direct_started
            direct_metrics = _rank_metrics(
                direct_vectors, queries,
                lambda query: model.encode(
                    query, prompt_name="SearchQuery", convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                ),
                ocr_texts,
            )
        except Exception as error:
            direct_failure = f"{type(error).__name__}: {error}"
            direct_index_elapsed = time.perf_counter() - direct_started
            direct_metrics = {"status": "unavailable", "reason": direct_failure}
        result["methods"]["embeddinggemma_2_direct_image"] = {
            "status": "measured" if len(direct_vectors) == len(images) else "partial" if direct_vectors else "unavailable",
            "model": "google/embeddinggemma-2",
            "model_load_seconds": round(model_load_seconds, 3) if model_load_seconds is not None else None,
            "memory_after_model_load_mb": memory_after_model_load,
            "model_load_memory_delta_mb": round(memory_after_model_load - direct_memory_before, 2)
            if memory_after_model_load is not None and direct_memory_before is not None else None,
            "images_indexed": len(direct_vectors),
            "failures": (len(images) - len(direct_vectors)) if direct_failure else 0,
            "index_seconds_total_including_load": round(direct_index_elapsed, 3),
            "index_seconds_per_image_excluding_load": round(
                direct_stages["image_embedding_seconds"] / max(len(direct_vectors), 1), 3
            ) if direct_vectors else None,
            "stage_seconds": {key: round(value, 3) for key, value in direct_stages.items()},
            "memory_start_mb": direct_memory_before, "memory_end_mb": _memory_mb(),
            "process_peak_memory_mb": _memory_mb(peak=True),
            **direct_metrics,
        }

        if not settings.GEMINI_API_KEY:
            result["limitations"].append("Gemini vision has no configured key, so description latency and relevance use OCR/metadata fallback.")
        if not direct_vectors:
            result["limitations"].append("Direct EmbeddingGemma 2 results were not measured; install the optional dependencies and cache the model or pass --allow-download.")
        result["limitations"].append("Fixtures are generated illustrations. Repeat with a labelled local set containing real photographs, screenshots, diagrams, and lookalikes before choosing a production embedding path.")
        result["exact_ocr_shared_results"] = [
            {"expected": item["expected"], "term": item["ocr_exact"],
             "found": item["ocr_exact"].casefold() in ocr_texts.get(item["expected"], "").casefold(),
             "matching_files": [name for name, text in ocr_texts.items()
                                if item["ocr_exact"].casefold() in text.casefold()]}
            for item in queries if item.get("ocr_exact")
        ]
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", help="Allow the local EmbeddingGemma 2 model to download from Hugging Face.")
    parser.add_argument("--manifest", type=Path, help="JSON manifest describing local images, search queries, and expected image IDs.")
    parser.add_argument("--output", type=Path, help="Write JSON results to this path; stdout is always printed too.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        result = run_benchmark(args.allow_download, args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if result.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
