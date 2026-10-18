"""Behavioral tests for the external DocLayNet candidate conversion."""

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pyarrow as arrow
import pyarrow.parquet as parquet
from PIL import Image

from mdprep.doclaynet import (
    DATASET_REVISION,
    LABEL_CRITERIA,
    LABEL_NAMES,
    LICENSE,
    LICENSE_URL,
    SOURCE,
    SPLIT,
    STATE,
    SourceFile,
    _metadata,
    _record,
    convert_doclaynet,
    convert_files,
    validate_doclaynet_record,
)
from mdprep.records import validate_record


class _Length:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCompiler:
    def __init__(self):
        self.compiled = 0

    def compile(self, items):
        self.compiled += len(items)
        return SimpleNamespace(expanded_lengths=[_Length(137) for _ in items])


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


def source_row(index: int = 0) -> dict:
    page_hash = hashlib.sha256(f"page-{index}".encode()).hexdigest()
    metadata = {
        "augmentation": {
            "type": "region_highlight",
            "box_xyxy": [1.0, 1.0, 7.0, 7.0],
            "color": [255, 0, 0],
            "encoding": "PNG",
            "image_size": [8, 8],
            "line_width_fraction": 0.004,
            "mask_opacity": 0.15,
            "original_image_sha256": "a" * 64,
            "pillow_version": "12.3.0",
            "resize": None,
        },
        "bbox_xywh": [1.0, 1.0, 6.0, 6.0],
        "coco_height": 8,
        "coco_width": 8,
        "collection": "test_collection",
        "doc_category": "financial_reports",
        "image_group_id": f"doclaynet:{page_hash}",
        "image_id": index,
        "num_pages": 1,
        "original_filename": f"report-{index}.pdf",
        "original_height": 8.0,
        "original_width": 8.0,
        "page_hash": page_hash,
        "page_no": 0,
        "source_category_id": 0,
        "source_dataset": "docling-project/DocLayNet-v1.1",
        "source_license": "CDLA-Permissive-1.0",
        "source_license_url": LICENSE_URL,
        "source_revision": "upstream-revision",
        "source_row": f"{page_hash}:0",
    }
    return {
        "images": [{"bytes": png_bytes(), "path": None}],
        "inputs": "Identify the region highlighted in red.",
        "labels": index % len(LABEL_NAMES),
        "metadata": json.dumps(metadata),
    }


def write_source(path: Path, offset: int) -> SourceFile:
    table = arrow.Table.from_pylist([source_row(offset + index) for index in range(100)])
    feature_metadata = {
        "info": {"features": {"labels": {"names": list(LABEL_NAMES)}}}
    }
    table = table.replace_schema_metadata(
        {b"huggingface": json.dumps(feature_metadata).encode()}
    )
    parquet.write_table(table, path)
    split = path.stem
    return SourceFile(path, split, f"data/{split}.parquet", f"https://example.test/{split}")


def candidate_row(label: int = 7) -> dict:
    source = source_row(label)
    metadata = _metadata(source)
    image_hash = hashlib.sha256(source["images"][0]["bytes"]).hexdigest()
    return _record(source, metadata, "validation", 0, "media/example.png", image_hash)


class DocLayNetConversionTests(unittest.TestCase):
    def test_record_preserves_declared_label_order_and_asserts_source_index(self):
        row = candidate_row(7)
        parsed = validate_doclaynet_record(row)
        self.assertEqual(row["state"], STATE)
        self.assertEqual(row["question"]["criteria"], LABEL_CRITERIA)
        self.assertEqual(row["target"], {"label": "option_7"})
        self.assertEqual([candidate.description for candidate in parsed.candidates], list(LABEL_NAMES))
        row["target"]["label"] = "option_6"
        with self.assertRaisesRegex(
            ValueError, "^DocLayNet target disagrees with the source label index$"
        ):
            validate_doclaynet_record(row)

    def test_canonical_validator_rejects_unregistered_external_split(self):
        row = candidate_row()
        validate_doclaynet_record(row)
        with self.assertRaisesRegex(ValueError, "^Unknown split$"):
            validate_record(row)

    def test_conversion_is_byte_deterministic_and_compiles_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "source"
            cache.mkdir()
            sources = [
                write_source(cache / "validation.parquet", 0),
                write_source(cache / "test.parquet", 100),
            ]
            support = [cache / "README.md", cache / "provenance.json"]
            support[0].write_text("dataset card\n", encoding="utf-8")
            support[1].write_text("{}\n", encoding="utf-8")
            output = root / "candidate"
            receipt_path = root / "audit" / "doclaynet-conversion.json"
            compiler = FakeCompiler()
            first = convert_files(sources, support, output, receipt_path, compiler)
            first_bytes = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            receipt_bytes = receipt_path.read_bytes()
            second = convert_files(sources, support, output, receipt_path, compiler)
            second_bytes = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(receipt_bytes, receipt_path.read_bytes())
            self.assertEqual(compiler.compiled, 400)
            self.assertEqual(first["total"]["rows"], 200)
            self.assertEqual(first["images"]["count"], 200)
            self.assertEqual(first["compiled"], {
                "records": 200,
                "min_positions": 137,
                "max_positions": 137,
                "position_limit": 2048,
                "all_within_budget": True,
            })
            self.assertFalse(first["source_scope"]["train_fetched"])

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / ".audit/doclaynet-conversion.json").is_file(),
                         "needs the local DocLayNet conversion receipt, which is not published")
    def test_checked_candidate_receipt_and_files_are_self_consistent(self):
        root = Path(__file__).resolve().parents[1]
        receipt_path = root / ".audit" / "doclaynet-conversion.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["pinned_revision"], DATASET_REVISION)
        self.assertEqual(receipt["staging"]["source"], SOURCE)
        self.assertEqual(receipt["staging"]["split"], SPLIT)
        self.assertFalse(receipt["staging"]["admitted_to_canonical_corpus"])
        self.assertFalse(receipt["source_scope"]["train_fetched"])
        self.assertEqual(receipt["source_scope"]["fetched_splits"], ["validation", "test"])
        self.assertEqual(receipt["license"]["name"], LICENSE)
        self.assertEqual(receipt["total"]["rows"], 200)
        self.assertEqual(receipt["images"]["count"], 200)
        self.assertTrue(receipt["compiled"]["all_within_budget"])
        self.assertLessEqual(receipt["compiled"]["max_positions"], 2048)
        self.assertIn("PDF renders rather than scans", receipt["coverage_statement"])
        self.assertIn("do not cover photographs", receipt["coverage_statement"])
        self.assertIn("capture degradation", receipt["coverage_statement"])

        counted = {split: dict.fromkeys(LABEL_NAMES, 0) for split in ("validation", "test")}
        candidate = root / ".audit" / "doclaynet-candidate"
        rows = []
        for split in ("validation", "test"):
            record_path = candidate / "records" / f"{split}.jsonl"
            self.assertEqual(sha256_file(record_path), receipt["per_split"][split]["output_sha256"])
            for line in record_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                validate_doclaynet_record(row)
                label = row["audit"]["source_label_index"]
                self.assertEqual(row["target"]["label"], f"option_{label}")
                counted[split][LABEL_NAMES[label]] += 1
                rows.append(row)
        self.assertEqual(len(rows), 200)
        for split in counted:
            self.assertEqual(counted[split], receipt["per_split"][split]["per_class"])
        with self.assertRaisesRegex(ValueError, "^Unknown split$"):
            validate_record(rows[0])

        total_bytes = 0
        for image in receipt["images"]["files"]:
            path = root / image["path"]
            self.assertEqual(sha256_file(path), image["sha256"])
            self.assertEqual(path.stat().st_size, image["bytes"])
            total_bytes += image["bytes"]
        self.assertEqual(total_bytes, receipt["images"]["total_bytes"])

    def test_public_entry_point_is_exposed(self):
        self.assertTrue(callable(convert_doclaynet))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
