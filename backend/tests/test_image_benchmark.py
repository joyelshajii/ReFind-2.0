import sys
from pathlib import Path

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_image_search as benchmark


def test_rank_metrics_handles_empty_and_missing_expected_items():
    assert benchmark._rank_metrics({}, [], lambda query: [], {}) == {
        "top1_accuracy": 0.0,
        "mrr": 0.0,
        "expected_ranks": [],
        "missing_expected_items": 0,
        "query_embedding_plus_vector_search_latency_ms_mean": 0,
        "query_embedding_plus_vector_search_latency_ms_p95": 0,
        "exact_ocr": [],
    }

    metrics = benchmark._rank_metrics(
        {"available": [1.0, 0.0]},
        [{"query": "missing target", "expected": "missing"}],
        lambda query: [1.0, 0.0],
        {},
    )
    assert metrics["expected_ranks"] == [None]
    assert metrics["missing_expected_items"] == 1
    assert metrics["top1_accuracy"] == 0.0
    assert metrics["mrr"] == 0.0


def test_manifest_validates_paths_labels_and_image_content(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"images":[{"id":"image-a","path":"valid.png"}],'
        '"queries":[{"query":"a white square","expected":"image-a"}]}',
        encoding="utf-8",
    )
    records, queries = benchmark._load_manifest(manifest)
    assert records[0]["path"] == image_path.resolve()
    assert queries[0]["expected"] == "image-a"

    manifest.write_text(
        '{"images":[{"id":"missing","path":"absent.png"}],'
        '"queries":[{"query":"missing","expected":"missing"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not exist"):
        benchmark._load_manifest(manifest)

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_bytes(b"not an image")
    manifest.write_text(
        '{"images":[{"id":"invalid","path":"invalid.png"}],'
        '"queries":[{"query":"invalid","expected":"invalid"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a readable image"):
        benchmark._load_manifest(manifest)


def test_manifest_rejects_unknown_targets_and_duplicate_ids(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"images":[{"id":"image-a","path":"valid.png"}],'
        '"queries":[{"query":"no target","expected":"missing"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match an image id"):
        benchmark._load_manifest(manifest)

    manifest.write_text(
        '{"images":[{"id":"same","path":"valid.png"},'
        '{"id":"same","path":"valid.png"}],'
        '"queries":[{"query":"same","expected":"same"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicated"):
        benchmark._load_manifest(manifest)
