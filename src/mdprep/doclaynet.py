"""Stage pinned DocLayNet region examples as external evaluation candidates."""

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image

from mdprep.common import GENERATOR_VERSION, fetch, sha256_file, stable_id, write_json, write_jsonl
from mdprep.native import MAX_POSITIONS, NativeCompiler
from mdprep.records import DecisionInput, parse_request

DATASET = "tasksource/doclaynet-region"
DATASET_CONFIG = "default"
DATASET_REPOSITORY = f"https://huggingface.co/datasets/{DATASET}"
DATASET_REVISION = "a023fe646d5ba2fc83fcc553f356f9bba58d06b1"
REVISION_URL = f"{DATASET_REPOSITORY}/tree/{DATASET_REVISION}"
LICENSE = "cdla-permissive-1.0"
LICENSE_URL = "https://cdla.io/permissive-1-0/"
LICENSE_USE = "commercial"
ATTRIBUTION = "DocLayNet, IBM Research; region parquet repackaging by tasksource"
SOURCE = "doclaynet-region"
SPLIT = "external_ood"
FAMILY = "document_layout_region"
STATE = "Use only the attached image."
RECORDS_PATH = Path(__file__).with_name("records.py")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
LABEL_NAMES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)
LABEL_CRITERIA = {f"option_{index}": name for index, name in enumerate(LABEL_NAMES)}
_SOURCE_PATHS = {
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
_SUPPORT_PATHS = ("README.md", "provenance.json")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    upstream_split: str
    upstream_path: str
    url: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def _declared_label_names(path: Path) -> tuple[str, ...]:
    metadata = parquet.read_schema(path).metadata or {}
    encoded = metadata.get(b"huggingface")
    if encoded is None:
        raise ValueError(f"{path} does not declare Hugging Face feature metadata")
    features = json.loads(encoded)["info"]["features"]
    names = features["labels"]["names"]
    if isinstance(names, dict):
        names = [names[str(index)] for index in range(len(names))]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{path} has invalid declared label names")
    return tuple(names)


def _read_source(source_file: SourceFile) -> list[dict]:
    declared = _declared_label_names(source_file.path)
    if declared != LABEL_NAMES:
        raise ValueError(
            f"{source_file.path} label order changed: expected {LABEL_NAMES!r}, got {declared!r}"
        )
    rows = parquet.read_table(source_file.path).to_pylist()
    if len(rows) != 100:
        raise ValueError(f"{source_file.path} requires exactly 100 rows, got {len(rows)}")
    return rows


def _image_bytes(source: dict, metadata: dict) -> bytes:
    images = source.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError("DocLayNet row must contain exactly one image")
    payload = images[0].get("bytes")
    if not isinstance(payload, bytes) or not payload.startswith(PNG_SIGNATURE):
        raise ValueError("DocLayNet row image must contain PNG bytes")
    augmentation = metadata.get("augmentation")
    if not isinstance(augmentation, dict) or augmentation.get("encoding") != "PNG":
        raise ValueError("DocLayNet augmentation must declare PNG encoding")
    with Image.open(BytesIO(payload)) as image:
        if image.format != "PNG" or list(image.size) != augmentation.get("image_size"):
            raise ValueError("DocLayNet image bytes disagree with augmentation metadata")
    return payload


def _metadata(source: dict) -> dict:
    value = source.get("metadata")
    if not isinstance(value, str):
        raise ValueError("DocLayNet metadata must be a JSON string")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("DocLayNet metadata JSON must decode to an object")
    required = (
        "augmentation",
        "bbox_xywh",
        "collection",
        "doc_category",
        "image_group_id",
        "original_filename",
        "page_hash",
        "page_no",
        "source_dataset",
        "source_revision",
        "source_row",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"DocLayNet metadata is missing {missing}")
    if metadata.get("source_license", "").lower() != LICENSE:
        raise ValueError("DocLayNet row license disagrees with the pinned dataset license")
    if metadata.get("source_license_url") != LICENSE_URL:
        raise ValueError("DocLayNet row license URL disagrees with the pinned dataset license URL")
    return metadata


def _record(
    source: dict,
    metadata: dict,
    upstream_split: str,
    row_number: int,
    media_path: str,
    image_sha256: str,
) -> dict:
    label = source.get("labels")
    if type(label) is not int or not 0 <= label < len(LABEL_NAMES):
        raise ValueError(f"DocLayNet label must be an integer in [0, {len(LABEL_NAMES)})")
    instructions = source.get("inputs")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("DocLayNet instructions must be non-empty text")
    group_id = metadata["image_group_id"]
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("DocLayNet image_group_id must be non-empty text")
    variant = [upstream_split, metadata["source_row"]]
    target = f"option_{label}"
    row = {
        "id": stable_id(SOURCE, group_id, FAMILY, variant),
        "group_id": group_id,
        "split": SPLIT,
        "modality": "image",
        "state": STATE,
        "images": [media_path],
        "videos": [],
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(LABEL_CRITERIA),
        },
        "target": {"label": target},
        "provenance": {
            "dataset": DATASET,
            "config": DATASET_CONFIG,
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "revision_url": REVISION_URL,
            "upstream_split": upstream_split,
            "upstream_row_number": row_number,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "license_use": LICENSE_USE,
            "attribution": ATTRIBUTION,
            "doc_category": metadata["doc_category"],
            "collection": metadata["collection"],
            "page_no": metadata["page_no"],
            "original_filename": metadata["original_filename"],
            "page_hash": metadata["page_hash"],
            "image_group_id": metadata["image_group_id"],
            "source_dataset": metadata["source_dataset"],
            "source_revision": metadata["source_revision"],
            "source_row": metadata["source_row"],
            "source": SOURCE,
            "source_domain": SOURCE,
            "origin": "real_human_label",
            "family": FAMILY,
            "ood_axis": "source_domain",
            "variant": variant,
            "generator_version": GENERATOR_VERSION,
        },
        "audit": {
            "augmentation": metadata["augmentation"],
            "bbox_xywh": metadata["bbox_xywh"],
            "image_sha256": image_sha256,
            "source_label_index": label,
            "original_candidate_keys": list(LABEL_NAMES),
            "original_target_key": LABEL_NAMES[label],
        },
    }
    if row["target"]["label"] != f"option_{source['labels']}":
        raise AssertionError("DocLayNet canonical target disagrees with the source label index")
    return row


def decision_input(row: dict) -> DecisionInput:
    """Parse one locally validated candidate for native compilation."""
    question = row["question"]
    request = {key: row[key] for key in ("id", "state", "images", "videos")}
    request.update(
        kind=question["type"],
        instructions=question["instructions"],
        candidates=[
            {"key": key, "description": description}
            for key, description in question["criteria"].items()
        ],
    )
    return parse_request(request)


def validate_doclaynet_record(row: dict) -> DecisionInput:
    """Validate the unregistered external candidate without admitting it canonically."""
    top_level = {
        "id",
        "group_id",
        "split",
        "modality",
        "state",
        "images",
        "videos",
        "question",
        "target",
        "provenance",
        "audit",
    }
    if not isinstance(row, dict) or set(row) != top_level:
        raise ValueError(f"DocLayNet record requires exactly {sorted(top_level)}")
    if row["split"] != SPLIT:
        raise ValueError("DocLayNet candidates must use the external_ood split")
    if row["modality"] != "image" or len(row["images"]) != 1 or row["videos"]:
        raise ValueError("DocLayNet candidates require exactly one image and no video")
    if row["state"] != STATE:
        raise ValueError("DocLayNet candidate state must use the canonical image-only wording")
    question = row["question"]
    if not isinstance(question, dict) or set(question) != {"type", "instructions", "criteria"}:
        raise ValueError("Invalid DocLayNet question")
    if question["type"] != "choice" or question["criteria"] != LABEL_CRITERIA:
        raise ValueError("DocLayNet criteria must preserve the declared label order")
    if not isinstance(question["instructions"], str) or not question["instructions"].strip():
        raise ValueError("Invalid DocLayNet instructions")
    if not isinstance(row["target"], dict) or set(row["target"]) != {"label"}:
        raise ValueError("Invalid DocLayNet target")
    provenance = row["provenance"]
    audit = row["audit"]
    if not isinstance(provenance, dict) or not isinstance(audit, dict):
        raise ValueError("DocLayNet provenance/audit must be objects")
    if (
        provenance.get("dataset") != DATASET
        or provenance.get("revision") != DATASET_REVISION
        or provenance.get("source") != SOURCE
        or provenance.get("source_domain") != SOURCE
        or provenance.get("license") != LICENSE
        or provenance.get("license_url") != LICENSE_URL
        or provenance.get("license_use") != LICENSE_USE
        or provenance.get("origin") != "real_human_label"
        or provenance.get("family") != FAMILY
        or provenance.get("ood_axis") != "source_domain"
    ):
        raise ValueError("Record does not match the DocLayNet staging contract")
    required_provenance = (
        "repository",
        "attribution",
        "doc_category",
        "collection",
        "original_filename",
        "page_hash",
        "image_group_id",
        "source_dataset",
        "source_revision",
        "source_row",
        "generator_version",
    )
    if any(not isinstance(provenance.get(key), str) or not provenance[key] for key in required_provenance):
        raise ValueError("DocLayNet provenance is incomplete")
    if type(provenance.get("page_no")) is not int:
        raise ValueError("DocLayNet page_no must be an integer")
    if row["group_id"] != provenance["image_group_id"]:
        raise ValueError("DocLayNet group identity disagrees with provenance")
    if row["id"] != stable_id(SOURCE, row["group_id"], FAMILY, provenance.get("variant")):
        raise ValueError("DocLayNet record identity does not match its variant")
    label = audit.get("source_label_index")
    if type(label) is not int or not 0 <= label < len(LABEL_NAMES):
        raise ValueError("DocLayNet source label index is invalid")
    if row["target"]["label"] != f"option_{label}":
        raise ValueError("DocLayNet target disagrees with the source label index")
    if audit.get("original_candidate_keys") != list(LABEL_NAMES):
        raise ValueError("DocLayNet original label names are missing or reordered")
    if audit.get("original_target_key") != LABEL_NAMES[label]:
        raise ValueError("DocLayNet original target disagrees with the source label index")
    if not isinstance(audit.get("augmentation"), dict) or not isinstance(audit.get("bbox_xywh"), list):
        raise ValueError("DocLayNet augmentation audit is incomplete")
    image_sha256 = audit.get("image_sha256")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError("DocLayNet image hash is invalid")
    return decision_input(row)


def acquire_doclaynet(cache_directory: Path) -> tuple[list[SourceFile], list[Path]]:
    """Acquire only the two pinned evaluation parquets and their source metadata."""
    base = f"{DATASET_REPOSITORY}/resolve/{DATASET_REVISION}"
    support = [fetch(f"{base}/{path}", cache_directory / path) for path in _SUPPORT_PATHS]
    sources = [
        SourceFile(
            fetch(f"{base}/{upstream_path}", cache_directory / f"{split}.parquet"),
            split,
            upstream_path,
            f"{base}/{upstream_path}",
        )
        for split, upstream_path in _SOURCE_PATHS.items()
    ]
    provenance = json.loads((cache_directory / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("license") != LICENSE or provenance.get("license_url") != LICENSE_URL:
        raise ValueError("Pinned dataset provenance has unexpected licensing")
    if tuple(provenance.get("label_names", ())) != LABEL_NAMES:
        raise ValueError("Pinned dataset provenance has unexpected label order")
    expected = {split: 100 for split in _SOURCE_PATHS}
    actual = {split: provenance.get("splits", {}).get(split, {}).get("rows") for split in _SOURCE_PATHS}
    if actual != expected:
        raise ValueError(f"Pinned dataset evaluation split sizes changed: {actual}")
    return sources, support


def convert_files(
    source_files: list[SourceFile],
    support_files: list[Path],
    output_directory: Path,
    receipt_path: Path,
    compiler: NativeCompiler,
) -> dict:
    """Convert, compile-gate, and serialize pinned DocLayNet evaluation files."""
    if compiler is None:
        raise ValueError("a pinned NativeCompiler is required")
    root = receipt_path.parent.parent.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records_directory = output_directory / "records"
    media_directory = output_directory / "media" / SOURCE
    records_directory.mkdir(parents=True, exist_ok=True)
    media_directory.mkdir(parents=True, exist_ok=True)
    per_split: dict[str, dict] = {}
    total_classes = collections.Counter()
    image_receipts = []
    expected_media: set[Path] = set()
    positions = []
    total_media_bytes = 0

    for source_file in source_files:
        rows = []
        class_counts = collections.Counter()
        for row_number, source in enumerate(_read_source(source_file)):
            metadata = _metadata(source)
            payload = _image_bytes(source, metadata)
            image_sha256 = hashlib.sha256(payload).hexdigest()
            filename = f"{source_file.upstream_split}-{row_number:05d}-{image_sha256[:16]}.png"
            destination = media_directory / filename
            if not destination.is_file() or destination.read_bytes() != payload:
                destination.write_bytes(payload)
            expected_media.add(destination)
            media_path = destination.relative_to(output_directory).as_posix()
            row = _record(
                source,
                metadata,
                source_file.upstream_split,
                row_number,
                media_path,
                image_sha256,
            )
            item = validate_doclaynet_record(row)
            batch = compiler.compile((item,))
            compiled_positions = int(batch.expanded_lengths[0].item())
            if compiled_positions > MAX_POSITIONS:
                raise ValueError(f"compiler returned over-budget input: {compiled_positions}")
            positions.append(compiled_positions)
            rows.append(row)
            label = source["labels"]
            class_counts[LABEL_NAMES[label]] += 1
            total_classes[LABEL_NAMES[label]] += 1
            image_receipts.append(
                {
                    "upstream_split": source_file.upstream_split,
                    "upstream_row_number": row_number,
                    "path": _relative(destination, root),
                    "sha256": image_sha256,
                    "bytes": len(payload),
                }
            )
            total_media_bytes += len(payload)
        record_path = records_directory / f"{source_file.upstream_split}.jsonl"
        write_jsonl(record_path, rows)
        ordered_counts = {name: class_counts[name] for name in LABEL_NAMES}
        values = list(ordered_counts.values())
        per_split[source_file.upstream_split] = {
            "rows": len(rows),
            "per_class": ordered_counts,
            "balanced": max(values) - min(values) <= 1,
            "balance_definition": "maximum and minimum class counts differ by at most one",
            "output_path": _relative(record_path, root),
            "output_sha256": sha256_file(record_path),
        }

    for path in media_directory.glob("*.png"):
        if path not in expected_media:
            path.unlink()
    source_receipts = {}
    for source in source_files:
        source_receipts[source.upstream_split] = {
            "path": _relative(source.path, root),
            "upstream_path": source.upstream_path,
            "url": source.url,
            "sha256": sha256_file(source.path),
            "bytes": source.path.stat().st_size,
        }
    for path in support_files:
        source_receipts[path.name] = {
            "path": _relative(path, root),
            "url": f"{DATASET_REPOSITORY}/resolve/{DATASET_REVISION}/{path.name}",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    total_ordered = {name: total_classes[name] for name in LABEL_NAMES}
    total_values = list(total_ordered.values())
    receipt = {
        "status": "VERIFIED",
        "dataset": DATASET,
        "config": DATASET_CONFIG,
        "pinned_revision": DATASET_REVISION,
        "repository": DATASET_REPOSITORY,
        "staging": {
            "dataset_status": "candidate",
            "canonical_corpus": "deliberately_not_admitted",
            "admitted_to_canonical_corpus": False,
            "source": SOURCE,
            "split": SPLIT,
            "admission_requirement": (
                "Add source 'doclaynet-region' to SOURCE_CONTRACTS and split 'external_ood' "
                "to SPLITS during a qualification cycle that deliberately freezes a new processor identity."
            ),
            "record_schema_sha256": sha256_file(RECORDS_PATH),
        },
        "source_files": source_receipts,
        "source_scope": {
            "fetched_splits": [source.upstream_split for source in source_files],
            "train_fetched": False,
            "upstream_train_rows": 100000,
        },
        "license": {
            "name": LICENSE,
            "url": LICENSE_URL,
            "use": LICENSE_USE,
            "attribution": ATTRIBUTION,
        },
        "label_names": {str(index): name for index, name in enumerate(LABEL_NAMES)},
        "per_split": per_split,
        "total": {
            "rows": sum(details["rows"] for details in per_split.values()),
            "per_class": total_ordered,
            "balanced": max(total_values) - min(total_values) <= 1,
            "balance_definition": "maximum and minimum class counts differ by at most one",
        },
        "images": {
            "count": len(image_receipts),
            "total_bytes": total_media_bytes,
            "files": image_receipts,
        },
        "compiled": {
            "records": len(positions),
            "min_positions": min(positions),
            "max_positions": max(positions),
            "position_limit": MAX_POSITIONS,
            "all_within_budget": all(position <= MAX_POSITIONS for position in positions),
        },
        "coverage_statement": (
            "These images are PDF renders rather than scans. They add real-document layout and typography, "
            "but do not cover photographs or capture degradation such as skew, blur, or photocopy noise."
        ),
    }
    write_json(receipt_path, receipt)
    return receipt


def convert_doclaynet(
    root: Path,
    *,
    output_directory: Path | None = None,
    receipt_path: Path | None = None,
) -> dict:
    """Public entry point: acquire, convert, native-compile, and receipt DocLayNet."""
    root = root.resolve()
    audit = root / ".audit"
    output = (output_directory or audit / "doclaynet-candidate").resolve()
    receipt = (receipt_path or audit / "doclaynet-conversion.json").resolve()
    sources, support = acquire_doclaynet(audit / "doclaynet-source")
    compiler = NativeCompiler.from_artifact(
        audit / "finalized" / "finalized-model", media_root=output
    )
    return convert_files(sources, support, output, receipt, compiler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = convert_doclaynet(
        args.root,
        output_directory=args.output_directory,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
