"""Prepare provenance-preserving Open Images V7 train photographs."""

import argparse
import csv
import io
import json
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .common import (
    PART_QUOTAS,
    SPLITS,
    choice,
    fetch,
    noul,
    record,
    rng_for,
    score,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)

SOURCE_NAME = "open_images_v7_train_human_verified"
EXTRACTOR_VERSION = "photos-2026-10-15-v1"
DOWNLOAD_PAGE_URL = "https://storage.googleapis.com/openimages/web/download_v7.html"
FACTS_PAGE_URL = "https://storage.googleapis.com/openimages/web/factsfigures_v7.html"
METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-with-rotation.csv"
)
LABELS_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-train-annotations-human-imagelabels.csv"
)
CLASSES_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-class-descriptions.csv"
)
CVDF_README_URL = (
    "https://raw.githubusercontent.com/cvdfoundation/"
    "open-images-dataset/master/README.md"
)
CVDF_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
ANNOTATION_LICENSE = "CC BY 4.0"
ANNOTATION_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
IMAGE_ID_PREFIX = "0"
MAX_METADATA_ROWS = 8_000
MAX_METADATA_BYTES = 192 * 1024 * 1024
MAX_IMAGE_SIDE = 768
JPEG_QUALITY = 84
NEAR_DUPLICATE_DISTANCE = 5
WORKERS_NOTE = "Downloads are deliberately sequential for deterministic fallback selection."
USER_AGENT = "multimodal-decision-data/0.1"

# A semantic family held out from every non-OOD selected image. These are exact
# Open Images class descriptions, not substring guesses.
TRANSPORT_CLASSES = frozenset(
    name.casefold()
    for name in (
        "Aircraft",
        "Airplane",
        "Ambulance",
        "Auto rickshaw",
        "Barge",
        "Bicycle",
        "Boat",
        "Bus",
        "Canoe",
        "Car",
        "Cart",
        "Catamaran",
        "Helicopter",
        "Jet ski",
        "Land vehicle",
        "Limousine",
        "Minibus",
        "Motorcycle",
        "Personal water craft",
        "Pickup truck",
        "Police car",
        "Raft",
        "Rickshaw",
        "Sailboat",
        "Segway",
        "Ship",
        "Snowmobile",
        "Submarine",
        "Taxi",
        "Train",
        "Tram",
        "Truck",
        "Unicycle",
        "Van",
        "Vehicle",
        "Watercraft",
        "Wheelchair",
    )
)

_IMAGE_COUNTS = {
    split: (quota + 5) // 6 for split, quota in PART_QUOTAS["photos"].items()
}
_ACQUISITION_CONFIG = {
    "extractor_version": EXTRACTOR_VERSION,
    "image_id_prefix": IMAGE_ID_PREFIX,
    "max_metadata_rows": MAX_METADATA_ROWS,
    "max_metadata_bytes": MAX_METADATA_BYTES,
    "metadata_url": METADATA_URL,
    "labels_url": LABELS_URL,
    "classes_url": CLASSES_URL,
}


def _request(url: str, *, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def _remote_headers(url: str) -> dict[str, str]:
    with urllib.request.urlopen(_request(url, method="HEAD"), timeout=90) as response:
        wanted = (
            "Content-Length",
            "Content-Type",
            "ETag",
            "Last-Modified",
            "x-goog-generation",
            "x-goog-hash",
        )
        return {name: response.headers[name] for name in wanted if response.headers[name]}


def _eligible_license(value: str) -> tuple[str, str] | None:
    normalized = value.strip()
    match = re.fullmatch(
        r"https?://creativecommons\.org/licenses/by/(\d\.\d)/?", normalized
    )
    if match:
        return f"CC BY {match.group(1)}", normalized
    if re.fullmatch(
        r"https?://creativecommons\.org/publicdomain/zero/1\.0/?", normalized
    ):
        return "CC0 1.0", normalized
    return None


def _extract_metadata(destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    selected = 0
    consumed = 0
    fieldnames: list[str] | None = None
    try:
        with urllib.request.urlopen(_request(METADATA_URL), timeout=180) as response:
            text = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            if reader.fieldnames is None:
                raise RuntimeError("Open Images image metadata has no CSV header")
            fieldnames = list(reader.fieldnames)
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    consumed += sum(len(value.encode("utf-8")) for value in row.values())
                    image_id = row.get("ImageID", "").lower()
                    license_info = _eligible_license(row.get("License", ""))
                    required = (
                        row.get("Subset") == "train"
                        and image_id.startswith(IMAGE_ID_PREFIX)
                        and license_info is not None
                        and row.get("OriginalURL")
                        and row.get("OriginalLandingURL")
                        and row.get("AuthorProfileURL")
                        and row.get("Author")
                        and row.get("OriginalMD5")
                    )
                    if required:
                        writer.writerow(row)
                        selected += 1
                    if selected >= MAX_METADATA_ROWS:
                        break
                    if consumed >= MAX_METADATA_BYTES:
                        raise RuntimeError(
                            "Bounded Open Images metadata scan reached "
                            f"{MAX_METADATA_BYTES} bytes with only {selected} eligible rows"
                        )
        if selected < MAX_METADATA_ROWS:
            raise RuntimeError(
                f"Open Images metadata ended after only {selected} eligible prefix rows"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "rows": selected,
        "approximate_csv_payload_bytes_scanned": consumed,
        "fieldnames": fieldnames,
    }


def _extract_labels(candidate_ids: set[str], destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.partial")
    rows = 0
    matched_images: set[str] = set()
    consumed = 0
    previous_id = ""
    ordered = True
    prefix_ceiling = IMAGE_ID_PREFIX + ("f" * 15)
    try:
        with urllib.request.urlopen(_request(LABELS_URL), timeout=180) as response:
            text = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            if reader.fieldnames is None:
                raise RuntimeError("Open Images human-label metadata has no CSV header")
            fieldnames = list(reader.fieldnames)
            with temporary.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    consumed += sum(len(value.encode("utf-8")) for value in row.values())
                    image_id = row.get("ImageID", "").lower()
                    if previous_id and image_id < previous_id:
                        ordered = False
                    previous_id = image_id
                    if ordered and image_id > prefix_ceiling:
                        break
                    if image_id in candidate_ids:
                        writer.writerow(row)
                        rows += 1
                        matched_images.add(image_id)
        if not ordered:
            raise RuntimeError(
                "V7 human-label CSV was not ordered by ImageID; refusing an unbounded "
                "2.7 GB fallback scan"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "rows": rows,
        "matched_images": len(matched_images),
        "approximate_csv_payload_bytes_scanned": consumed,
        "stopped_after_image_id_prefix": IMAGE_ID_PREFIX,
    }


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["ImageID"].lower(): row for row in csv.DictReader(stream)}


def _load_classes(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) == 2:
                classes[row[0]] = row[1]
    return classes


def _load_labels(path: Path) -> tuple[dict[str, dict[str, int]], set[str]]:
    values: dict[str, dict[str, int]] = defaultdict(dict)
    conflicts: set[str] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Source") != "verification":
                continue
            confidence_text = row.get("Confidence", "")
            if confidence_text not in {"0", "0.0", "1", "1.0"}:
                continue
            image_id = row["ImageID"].lower()
            label_id = row["LabelName"]
            confidence = int(float(confidence_text))
            previous = values[image_id].get(label_id)
            if previous is not None and previous != confidence:
                conflicts.add(image_id)
            values[image_id][label_id] = confidence
    return values, conflicts


def _acquire_metadata(raw_dir: Path) -> tuple[Path, Path, Path, dict]:
    metadata_path = raw_dir / "eligible_train_image_metadata.csv"
    labels_path = raw_dir / "eligible_train_human_labels.csv"
    classes_path = raw_dir / "oidv7-class-descriptions.csv"
    download_page_path = raw_dir / "open_images_v7_download.html"
    facts_page_path = raw_dir / "open_images_v7_facts.html"
    cvdf_readme_path = raw_dir / "cvdf_open_images_readme.txt"
    acquisition_path = raw_dir / "acquisition.json"

    cached = False
    required_cache_files = (
        metadata_path,
        labels_path,
        classes_path,
        download_page_path,
        facts_page_path,
        cvdf_readme_path,
    )
    if acquisition_path.is_file() and all(path.is_file() for path in required_cache_files):
        try:
            cached_data = json.loads(acquisition_path.read_text(encoding="utf-8"))
            cached = cached_data.get("config") == _ACQUISITION_CONFIG
            if cached:
                expected = cached_data.get("local_files", {})
                for path in required_cache_files:
                    if expected.get(path.name) != sha256_file(path):
                        raise RuntimeError(f"Cached photo source hash mismatch: {path}")
        except (json.JSONDecodeError, OSError):
            cached = False
    if cached:
        return metadata_path, labels_path, classes_path, cached_data

    metadata_remote = _remote_headers(METADATA_URL)
    labels_remote = _remote_headers(LABELS_URL)
    classes_remote = _remote_headers(CLASSES_URL)
    metadata_scan = _extract_metadata(metadata_path)
    candidate_ids = set(_load_metadata(metadata_path))
    labels_scan = _extract_labels(candidate_ids, labels_path)
    download_page_path = fetch(DOWNLOAD_PAGE_URL, download_page_path)
    facts_page_path = fetch(FACTS_PAGE_URL, facts_page_path)
    cvdf_readme_path = fetch(CVDF_README_URL, cvdf_readme_path)
    fetch(CLASSES_URL, classes_path)
    acquisition = {
        "config": _ACQUISITION_CONFIG,
        "official_revision": "Open Images V7 (released October 2022)",
        "upstream_split": "train",
        "annotation_license": {
            "name": ANNOTATION_LICENSE,
            "url": ANNOTATION_LICENSE_URL,
            "licensor": "Google LLC",
        },
        "sources": {
            "image_metadata": {"url": METADATA_URL, "headers": metadata_remote},
            "human_image_labels": {"url": LABELS_URL, "headers": labels_remote},
            "class_descriptions": {"url": CLASSES_URL, "headers": classes_remote},
            "download_page": DOWNLOAD_PAGE_URL,
            "license_facts_page": FACTS_PAGE_URL,
            "cvdf_mirror_documentation": CVDF_README_URL,
        },
        "bounded_scans": {
            "image_metadata": metadata_scan,
            "human_image_labels": labels_scan,
        },
        "local_files": {
            "eligible_train_image_metadata.csv": sha256_file(metadata_path),
            "eligible_train_human_labels.csv": sha256_file(labels_path),
            "oidv7-class-descriptions.csv": sha256_file(classes_path),
            "open_images_v7_download.html": sha256_file(download_page_path),
            "open_images_v7_facts.html": sha256_file(facts_page_path),
            "cvdf_open_images_readme.txt": sha256_file(cvdf_readme_path),
        },
    }
    write_json(acquisition_path, acquisition)
    return metadata_path, labels_path, classes_path, acquisition


def _named_labels(
    labels: dict[str, int], classes: dict[str, str], confidence: int
) -> list[tuple[str, str]]:
    by_name: dict[str, str] = {}
    for label_id, value in labels.items():
        name = classes.get(label_id)
        if value == confidence and name:
            by_name.setdefault(name, label_id)
    return sorted(((label_id, name) for name, label_id in by_name.items()), key=lambda x: x[1])


def _candidate_rows(
    metadata: dict[str, dict[str, str]],
    labels: dict[str, dict[str, int]],
    conflicts: set[str],
    classes: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    regular: list[dict] = []
    ood: list[dict] = []
    for image_id, image_metadata in metadata.items():
        if image_id in conflicts or image_id not in labels:
            continue
        positive = _named_labels(labels[image_id], classes, 1)
        negative = _named_labels(labels[image_id], classes, 0)
        if len(positive) < 4 or len(negative) < 4:
            continue
        row = {
            "image_id": image_id,
            "metadata": image_metadata,
            "positive": positive,
            "negative": negative,
        }
        has_transport = any(name.casefold() in TRANSPORT_CLASSES for _, name in positive)
        (ood if has_transport else regular).append(row)
    regular.sort(key=lambda row: stable_id("regular-candidate", row["image_id"]))
    ood.sort(key=lambda row: stable_id("transport-candidate", row["image_id"]))
    return regular, ood


def _rotation(row: dict[str, str]) -> int:
    value = row.get("Rotation", "").strip().lower()
    if not value or value == "nan":
        return 0
    try:
        rotation = int(float(value))
    except ValueError as error:
        raise ValueError(f"Invalid Open Images rotation value: {value!r}") from error
    if rotation not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported Open Images rotation value: {rotation}")
    return rotation


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            result = (result << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return result


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(_request(url), timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _materialize_candidate(
    candidate: dict,
    raw_images_dir: Path,
    media_dir: Path,
) -> tuple[dict | None, dict | None]:
    image_id = candidate["image_id"]
    source_url = CVDF_IMAGE_URL.format(image_id=image_id)
    raw_path = raw_images_dir / f"{image_id}.jpg"
    opaque_name = f"{stable_id(SOURCE_NAME, image_id, 'media')}.jpg"
    media_path = media_dir / opaque_name
    try:
        _download(source_url, raw_path)
        source_sha256 = sha256_file(raw_path)
        with Image.open(raw_path) as opened:
            opened.load()
            image = opened.convert("RGB")
        rotation = _rotation(candidate["metadata"])
        if rotation:
            image = image.rotate(rotation, expand=True)
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        difference_hash = _difference_hash(image)
        media_dir.mkdir(parents=True, exist_ok=True)
        temporary = media_path.with_suffix(".jpg.partial")
        try:
            image.save(
                temporary,
                format="JPEG",
                quality=JPEG_QUALITY,
                optimize=True,
                progressive=True,
            )
            temporary.replace(media_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        asset = {
            **candidate,
            "raw_path": raw_path,
            "media_path": media_path,
            "media_relative": f"media/photos/{opaque_name}",
            "source_url": source_url,
            "source_sha256": source_sha256,
            "media_sha256": sha256_file(media_path),
            "width": image.width,
            "height": image.height,
            "difference_hash": f"{difference_hash:016x}",
            "difference_hash_int": difference_hash,
            "applied_rotation_ccw": rotation,
        }
        return asset, None
    except (
        OSError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        UnidentifiedImageError,
    ) as error:
        if media_path.exists():
            media_path.unlink()
        return None, {
            "image_id": image_id,
            "url": source_url,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _is_near_duplicate(value: int, accepted: list[int]) -> bool:
    return any((value ^ other).bit_count() <= NEAR_DUPLICATE_DISTANCE for other in accepted)


def _select_assets(
    candidates: list[dict],
    count: int,
    raw_images_dir: Path,
    media_dir: Path,
    used_creators: set[str],
    used_source_assets: set[str],
    used_original_md5: set[str],
    accepted_hashes: list[int],
    failures: list[dict],
) -> tuple[list[dict], Counter]:
    selected: list[dict] = []
    skipped: Counter = Counter()
    for candidate in candidates:
        metadata = candidate["metadata"]
        creator = metadata["AuthorProfileURL"].strip().casefold()
        source_asset = metadata["OriginalLandingURL"].strip().casefold()
        original_md5 = metadata["OriginalMD5"].strip()
        if creator in used_creators:
            skipped["creator_already_selected"] += 1
            continue
        if source_asset in used_source_assets:
            skipped["source_asset_already_selected"] += 1
            continue
        if original_md5 in used_original_md5:
            skipped["original_md5_already_selected"] += 1
            continue
        asset, failure = _materialize_candidate(candidate, raw_images_dir, media_dir)
        if failure:
            failures.append(failure)
            skipped["download_or_decode_failure"] += 1
            continue
        if asset is None:
            raise AssertionError("materialization returned neither an asset nor a failure")
        if _is_near_duplicate(asset["difference_hash_int"], accepted_hashes):
            asset["media_path"].unlink(missing_ok=True)
            skipped["perceptual_near_duplicate"] += 1
            continue
        used_creators.add(creator)
        used_source_assets.add(source_asset)
        used_original_md5.add(original_md5)
        accepted_hashes.append(asset["difference_hash_int"])
        selected.append(asset)
        if len(selected) == count:
            break
    return selected, skipped


def _evidence(label: tuple[str, str], confidence: int) -> dict:
    return {
        "label_id": label[0],
        "class_name": label[1],
        "confidence": confidence,
        "source": "verification",
        "annotation_kind": "human_verified_image_label",
    }


def _assign_choice(
    group_id: str,
    variant: int,
    selected: list[tuple[tuple[str, str], bool]],
    target_confidence: int,
) -> tuple[dict[str, str], str, list[dict]]:
    ordered = list(selected)
    rng_for(group_id, "choice-key-assignment", variant).shuffle(ordered)
    criteria: dict[str, str] = {}
    target = ""
    audit_evidence: list[dict] = []
    for index, (label, is_target) in enumerate(ordered):
        key = f"option_{chr(ord('a') + index)}"
        criteria[key] = label[1]
        if is_target:
            target = key
        confidence = target_confidence if is_target else 1 - target_confidence
        audit_evidence.append(_evidence(label, confidence))
    if not target:
        raise AssertionError("choice has no target")
    return criteria, target, audit_evidence


def _question_for(
    asset: dict,
    split: str,
    ordinal: int,
    variant: int,
) -> tuple[str, dict, str, dict]:
    group_id = stable_id(SOURCE_NAME, asset["image_id"])
    positive = list(asset["positive"])
    negative = list(asset["negative"])
    rng_for(group_id, "positive-labels", variant).shuffle(positive)
    rng_for(group_id, "negative-labels", variant).shuffle(negative)
    if split == "test_ood":
        positive.sort(key=lambda item: item[1].casefold() not in TRANSPORT_CLASSES)
    suffix = "transport_ood" if split == "test_ood" else "in_domain"

    if variant == 0:
        label = positive[0]
        return (
            f"verified_present_boolean_{suffix}",
            noul(f"Is “{label[1]}” human-verified as present in this photograph?"),
            "true",
            {
                "facts": [_evidence(label, 1)],
                "rule": "Return Yes exactly when the cited human-verified image label has confidence 1.",
                "computed_label": "true",
            },
        )
    if variant == 1:
        label = negative[0]
        return (
            f"verified_absent_boolean_{suffix}",
            noul(f"Is “{label[1]}” human-verified as present in this photograph?"),
            "false",
            {
                "facts": [_evidence(label, 0)],
                "rule": "Return No exactly when the cited human-verified image label has confidence 0.",
                "computed_label": "false",
            },
        )
    if variant == 2:
        options = [(positive[0], True), *((item, False) for item in negative[:3])]
        criteria, target, evidence = _assign_choice(
            group_id, variant, options, target_confidence=1
        )
        return (
            f"verified_present_choice_{suffix}",
            choice(
                "Which class is human-verified as present in this photograph?",
                criteria,
            ),
            target,
            {
                "facts": evidence,
                "rule": "Choose the sole option with human-verification confidence 1.",
                "computed_label": target,
            },
        )
    if variant == 3:
        options = [(negative[0], True), *((item, False) for item in positive[:3])]
        criteria, target, evidence = _assign_choice(
            group_id, variant, options, target_confidence=0
        )
        return (
            f"verified_absent_choice_{suffix}",
            choice(
                "Which class is human-verified as absent from this photograph?",
                criteria,
            ),
            target,
            {
                "facts": evidence,
                "rule": "Choose the sole option with human-verification confidence 0.",
                "computed_label": target,
            },
        )

    desired_present = (ordinal * 2 + variant - 4) % 4
    labels = positive[:desired_present] + negative[: 3 - desired_present]
    rng_for(group_id, "score-label-order", variant).shuffle(labels)
    names = "; ".join(f"“{name}”" for _, name in labels)
    evidence = [
        _evidence(item, 1 if item in positive[:desired_present] else 0) for item in labels
    ]
    label = str(desired_present)
    return (
        f"verified_set_cardinality_{suffix}",
        score(
            f"Among these three classes — {names} — how many are human-verified as present?",
            ["None", "One", "Two", "All three"],
        ),
        label,
        {
            "facts": evidence,
            "rule": "Count confidence-1 human-verified labels in the named three-label set; this is not an object count.",
            "verified_present_count": desired_present,
            "computed_label": label,
        },
    )


def _attribution(asset: dict, split: str) -> dict:
    metadata = asset["metadata"]
    license_name, license_url = _eligible_license(metadata["License"]) or ("", "")
    return {
        "image_id": asset["image_id"],
        "upstream_split": "train",
        "dataset_split": split,
        "title": metadata.get("Title", ""),
        "author": metadata["Author"],
        "author_profile_url": metadata["AuthorProfileURL"],
        "original_url": metadata["OriginalURL"],
        "original_landing_url": metadata["OriginalLandingURL"],
        "original_license": license_name,
        "original_license_url_exact": license_url,
        "original_size_bytes": metadata.get("OriginalSize", ""),
        "original_md5_base64": metadata["OriginalMD5"],
        "thumbnail_url": metadata.get("Thumbnail300KURL", ""),
        "cvdf_mirror_url": asset["source_url"],
        "cvdf_file_sha256": asset["source_sha256"],
        "processed_path": asset["media_relative"],
        "processed_sha256": asset["media_sha256"],
        "processed_dimensions": [asset["width"], asset["height"]],
        "applied_rotation_ccw": asset["applied_rotation_ccw"],
        "creator_group": stable_id("open-images-creator", metadata["AuthorProfileURL"]),
        "original_asset_group": stable_id("open-images-original-md5", metadata["OriginalMD5"]),
        "near_duplicate_group": stable_id("open-images-dhash", asset["difference_hash"]),
        "difference_hash": asset["difference_hash"],
    }


def prepare(root: Path) -> dict:
    """Acquire eligible train photos and write exactly 3,000 decision rows."""
    root = root.resolve()
    data_dir = root / "data"
    raw_dir = data_dir / "raw" / "open_images_v7"
    raw_images_dir = raw_dir / "images"
    media_dir = data_dir / "media" / "photos"
    parts_path = data_dir / "parts" / "photos.jsonl"
    source_path = data_dir / "sources" / "open_images_v7_photos.json"
    failures_path = raw_dir / "download_failures.jsonl"
    if source_path.is_file():
        previous = json.loads(source_path.read_text())
        for asset in previous["attributions"]:
            cached = raw_images_dir / f"{asset['image_id']}.jpg"
            if cached.is_file() and sha256_file(cached) != asset["cvdf_file_sha256"]:
                raise RuntimeError(f"Cached original photo hash mismatch: {cached}")

    metadata_path, labels_path, classes_path, acquisition = _acquire_metadata(raw_dir)
    metadata = _load_metadata(metadata_path)
    labels, conflicts = _load_labels(labels_path)
    classes = _load_classes(classes_path)
    regular_candidates, ood_candidates = _candidate_rows(
        metadata, labels, conflicts, classes
    )

    failures: list[dict] = []
    used_creators: set[str] = set()
    used_source_assets: set[str] = set()
    used_original_md5: set[str] = set()
    accepted_hashes: list[int] = []
    ood_assets, ood_skipped = _select_assets(
        ood_candidates,
        _IMAGE_COUNTS["test_ood"],
        raw_images_dir,
        media_dir,
        used_creators,
        used_source_assets,
        used_original_md5,
        accepted_hashes,
        failures,
    )
    if len(ood_assets) != _IMAGE_COUNTS["test_ood"]:
        write_jsonl(failures_path, failures)
        raise RuntimeError(
            f"Only {len(ood_assets)} of {_IMAGE_COUNTS['test_ood']} required "
            f"transport-OOD assets survived; candidate_count={len(ood_candidates)}, "
            f"skipped={dict(ood_skipped)}, download_failures={len(failures)}"
        )
    regular_required = sum(
        count for split, count in _IMAGE_COUNTS.items() if split != "test_ood"
    )
    regular_assets, regular_skipped = _select_assets(
        regular_candidates,
        regular_required,
        raw_images_dir,
        media_dir,
        used_creators,
        used_source_assets,
        used_original_md5,
        accepted_hashes,
        failures,
    )
    write_jsonl(failures_path, failures)
    if len(regular_assets) != regular_required:
        raise RuntimeError(
            f"Only {len(regular_assets)} of {regular_required} required in-domain "
            f"assets survived; candidate_count={len(regular_candidates)}, "
            f"skipped={dict(regular_skipped)}, download_failures={len(failures)}"
        )

    retained_media = {
        asset["media_path"].resolve() for asset in [*ood_assets, *regular_assets]
    }
    if media_dir.is_dir():
        for path in media_dir.glob("*.jpg"):
            if path.resolve() not in retained_media:
                path.unlink()

    assets_by_split: dict[str, list[dict]] = {}
    offset = 0
    for split in SPLITS:
        if split == "test_ood":
            assets_by_split[split] = ood_assets
            continue
        count = _IMAGE_COUNTS[split]
        assets_by_split[split] = regular_assets[offset : offset + count]
        offset += count

    rows: list[dict] = []
    primitive_counts: Counter = Counter()
    split_counts: Counter = Counter()
    attributions: list[dict] = []
    for split in SPLITS:
        remaining = PART_QUOTAS["photos"][split]
        for ordinal, asset in enumerate(assets_by_split[split]):
            attributions.append(_attribution(asset, split))
            variant_count = min(6, remaining)
            group_id = stable_id(SOURCE_NAME, asset["image_id"])
            attribution = attributions[-1]
            for variant in range(variant_count):
                family, question, target, audit = _question_for(
                    asset, split, ordinal, variant
                )
                rows.append(
                    record(
                        split=split,
                        source=SOURCE_NAME,
                        group_id=group_id,
                        family=family,
                        modality="image",
                        state="Use only the visible photograph and the named human-verification decision.",
                        images=[asset["media_relative"]],
                        question=question,
                        label=target,
                        provenance={
                            "license": attribution["original_license"],
                            "license_url": attribution["original_license_url_exact"],
                            "open_images_version": "V7",
                            "image_id": asset["image_id"],
                            "upstream_split": "train",
                            "source_manifest": "sources/open_images_v7_photos.json",
                            "original_landing_url": attribution["original_landing_url"],
                            "original_url": attribution["original_url"],
                            "author": attribution["author"],
                            "author_profile_url": attribution["author_profile_url"],
                            "original_license": attribution["original_license"],
                            "original_license_url_exact": attribution[
                                "original_license_url_exact"
                            ],
                            "attribution": (
                                f"“{attribution['title']}” by {attribution['author']}; "
                                f"{attribution['original_license']} "
                                f"({attribution['original_license_url_exact']}); "
                                f"{attribution['original_landing_url']}"
                            ),
                            "annotation_license": ANNOTATION_LICENSE,
                            "annotation_license_url": ANNOTATION_LICENSE_URL,
                            "cvdf_mirror_url": attribution["cvdf_mirror_url"],
                            "cvdf_file_sha256": attribution["cvdf_file_sha256"],
                            "processed_sha256": attribution["processed_sha256"],
                            "creator_group": attribution["creator_group"],
                            "original_asset_group": attribution["original_asset_group"],
                            "near_duplicate_group": attribution["near_duplicate_group"],
                        },
                        audit={
                            **audit,
                            "image_id": asset["image_id"],
                            "annotation_source": LABELS_URL,
                            "annotation_source_sha256": acquisition["local_files"][
                                "eligible_train_human_labels.csv"
                            ],
                            "negative_semantics": "Confidence 0 is an explicit human verification; missing labels are never treated as negative.",
                            "ood_intent": (
                                "Held-out transport-positive class subset; the same question templates are used in ID"
                                if split == "test_ood"
                                else "In-domain; all explicit transport-positive images excluded"
                            ),
                        },
                        variant=variant,
                    )
                )
                primitive_counts[question["type"]] += 1
                split_counts[split] += 1
            remaining -= variant_count
        if remaining != 0:
            raise AssertionError(f"Unfilled {split} quota: {remaining}")

    source_manifest = {
        "source": SOURCE_NAME,
        "revision": "Open Images V7 (October 2022)",
        "upstream_split": "train only",
        "official_download_page": DOWNLOAD_PAGE_URL,
        "cvdf_mirror": "s3://open-images-dataset/train (accessed by anonymous HTTPS)",
        "selection": {
            "eligible_original_licenses": ["CC BY (any stated version)", "CC0 1.0"],
            "license_evidence": "Exact per-image License field in official image metadata",
            "human_label_rule": "Only Source=verification and Confidence exactly 0 or 1",
            "image_id_prefix": IMAGE_ID_PREFIX,
            "candidate_metadata_rows": len(metadata),
            "regular_question_candidates": len(regular_candidates),
            "transport_ood_candidates": len(ood_candidates),
            "unique_creator_policy": "At most one selected photograph per AuthorProfileURL",
            "source_asset_policy": "At most one selected photograph per OriginalLandingURL",
            "exact_duplicate_policy": "At most one selected photograph per OriginalMD5",
            "near_duplicate_policy": (
                f"Reject 64-bit difference hashes within Hamming distance "
                f"{NEAR_DUPLICATE_DISTANCE}"
            ),
            "skipped": {
                "regular": dict(regular_skipped),
                "test_ood": dict(ood_skipped),
            },
        },
        "annotation_license": {
            "name": ANNOTATION_LICENSE,
            "url": ANNOTATION_LICENSE_URL,
            "licensor": "Google LLC",
        },
        "image_license_caveat": (
            "Open Images asks users to verify each image license independently. "
            "This builder retains only rows whose official original-image metadata "
            "states an exact CC BY or CC0 URL and preserves that URL and landing page."
        ),
        "processing": {
            "format": "JPEG",
            "max_side_pixels": MAX_IMAGE_SIDE,
            "quality": JPEG_QUALITY,
            "metadata_removed": True,
            "filenames": "Opaque stable IDs unrelated to labels or answers",
        },
        "acquisition": acquisition,
        "download_failures_file": "raw/open_images_v7/download_failures.jsonl",
        "download_failure_count": len(failures),
        "asset_count": len(attributions),
        "question_count": len(rows),
        "attributions": attributions,
        "known_limits": [
            "Selection is a deterministic bounded sample of eligible train images whose IDs begin with 0, not a representative sample of all 9M images.",
            "Human image-level labels support only verified presence/absence decisions; they do not support spatial, OCR, or object-instance-count claims.",
            "Perceptual duplicate control uses a conservative difference-hash heuristic and is not a semantic duplicate detector.",
            WORKERS_NOTE,
        ],
    }
    write_json(source_path, source_manifest)
    row_count = write_jsonl(parts_path, rows)
    if row_count != sum(PART_QUOTAS["photos"].values()):
        raise AssertionError(f"Expected 3000 rows, wrote {row_count}")

    return {
        "part": "photos",
        "rows": row_count,
        "unique_photographs": len(attributions),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "source_revision": "Open Images V7 (October 2022)",
        "upstream_split": "train",
        "download_failures": len(failures),
        "output": str(parts_path),
        "source_manifest": str(source_path),
        "invocation": f"python -m mdprep.photos --root {root}",
        "limits": source_manifest["known_limits"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare licensed Open Images V7 train-only decision data."
    )
    parser.add_argument("--root", type=Path, required=True, help="Project root")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
