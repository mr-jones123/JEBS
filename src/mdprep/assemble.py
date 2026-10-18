"""Assemble source parts, validate evidence separation, and build review/pilot packs."""

import argparse
import collections
import json
import math
import subprocess
from pathlib import Path

from PIL import Image

from mdprep.common import PART_QUOTAS, SPLITS, sha256_file, stable_id, write_json, write_jsonl
from mdprep.records import validate_record

PARTS = ("text", "photos", "visuals", "videos", "hardv1", "rule_variants")


def load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inspect_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    return {"width": stream["width"], "height": stream["height"],
            "fps": fps, "duration_s": float(stream["duration"]),
            "frame_count": int(stream["nb_frames"])}


def media_paths(row: dict) -> list[str]:
    return row["images"] + [video["path"] for video in row["videos"]]


def summarize(rows: list[dict]) -> dict:
    return {
        "questions": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "unique_images": len({p for row in rows for p in row["images"]}),
        "unique_videos": len({v["path"] for row in rows for v in row["videos"]}),
        "modalities": dict(collections.Counter(row["modality"] for row in rows)),
        "primitives": dict(collections.Counter(row["question"]["type"] for row in rows)),
        "sources": dict(collections.Counter(row["provenance"]["source"] for row in rows)),
        "families": dict(collections.Counter(row["provenance"]["family"] for row in rows)),
        "targets": dict(collections.Counter(row["target"]["label"] for row in rows)),
        "origins": dict(collections.Counter(row["provenance"]["origin"] for row in rows)),
        "holdout_axes": dict(collections.Counter(row["provenance"]["ood_axis"] for row in rows)),
        "source_domains": dict(collections.Counter(row["provenance"]["source_domain"] for row in rows)),
    }


def balanced_sample(rows: list[dict], limit: int) -> list[dict]:
    buckets: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        key = (row["modality"], row["question"]["type"], row["provenance"]["source"],
               row["provenance"]["family"])
        buckets[key].append(row)
    axes = collections.defaultdict(collections.deque)
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: stable_id("selection", row["id"]))
        axes[key[:2]].append(collections.deque(bucket))
    selected = []
    while len(selected) < min(limit, len(rows)):
        for axis in sorted(axes):
            if not axes[axis]:
                continue
            bucket = axes[axis].popleft()
            selected.append(bucket.popleft())
            if bucket:
                axes[axis].append(bucket)
            if len(selected) == limit:
                return selected
    return selected


def assemble(root: Path) -> dict:
    data = root / "data"
    rows = []
    quotas = collections.Counter()
    for part in PARTS:
        part_rows = load_rows(data / "parts" / f"{part}.jsonl")
        actual = collections.Counter((row["split"], row["modality"]) for row in part_rows)
        expected = collections.Counter()
        mappings = [(
            part,
            {
                "text": "text",
                "photos": "image",
                "videos": "video",
                "hardv1": "text",
                "rule_variants": "text",
            }.get(part),
        )]
        if part == "visuals":
            mappings = [("visual_image", "image"), ("visual_mixed", "mixed")]
        for quota_name, modality in mappings:
            for split, count in PART_QUOTAS[quota_name].items():
                expected[(split, modality)] += count
                quotas[(split, modality)] += count
        if actual != expected:
            raise ValueError(f"{part}: quota mismatch actual={dict(actual)} expected={dict(expected)}")
        rows.extend(part_rows)

    ids = set()
    group_splits = {}
    evidence_splits = {}
    text_splits = {}
    media = {}
    asset_splits = {}
    source_groups = {}
    for row in rows:
        validate_record(row)
        if row["id"] in ids:
            raise ValueError(f"Duplicate record ID: {row['id']}")
        ids.add(row["id"])
        previous = group_splits.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Group leakage: {row['group_id']}")
        provenance = row["provenance"]
        for field in ("creator_group", "original_asset_group", "original_landing_url", "near_duplicate_group"):
            if field not in provenance:
                continue
            value = provenance[field]
            identity = (row["split"], row["group_id"])
            prior = source_groups.setdefault((provenance["source"], field, value), identity)
            if prior != identity:
                raise ValueError(f"Source asset grouping mismatch ({field}): {row['id']}")
        if provenance.get("upstream_split") in ("test", "validation", "dev") and row["split"] in (
            "train", "dev", "calibration", "policy"
        ):
            raise ValueError(f"Upstream evaluation content leaked: {row['id']}")
        signatures = []
        for relative in media_paths(row):
            path = (data / relative).resolve()
            if not path.is_relative_to(data.resolve()) or not path.is_file():
                raise ValueError(f"Invalid/missing media path: {relative}")
            if relative not in media:
                entry = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    with Image.open(path) as image:
                        image.load()
                        entry.update({"kind": "image", "width": image.width, "height": image.height})
                else:
                    entry.update({"kind": "video", **inspect_video(path)})
                media[relative] = entry
            signature = media[relative]["sha256"]
            asset_previous = asset_splits.setdefault(signature, (row["split"], row["group_id"]))
            if asset_previous != (row["split"], row["group_id"]):
                raise ValueError(f"Identical media assigned to different source groups: {relative}")
            signatures.append(signature)
        for video in row["videos"]:
            actual = media[video["path"]]
            for field in ("fps", "duration_s", "frame_count"):
                if not math.isclose(float(video[field]), float(actual[field]), rel_tol=1e-4, abs_tol=1e-4):
                    raise ValueError(f"Video metadata mismatch {field}: {row['id']}")
        normalized_text = " ".join(row["state"].casefold().split())
        evidence = stable_id(normalized_text, signatures)
        identity = (row["split"], row["group_id"])
        evidence_previous = evidence_splits.setdefault(evidence, identity)
        if evidence_previous != identity:
            raise ValueError(f"Identical evidence assigned to different groups: {row['id']}")
        if row["modality"] == "text":
            text_previous = text_splits.setdefault(normalized_text, identity)
            if text_previous != identity:
                raise ValueError(f"Normalized text assigned to different groups: {row['id']}")

    splits = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        selected.sort(key=lambda row: stable_id("row-order", row["id"]))
        destination = data / "prepared" / f"{split}.jsonl"
        write_jsonl(destination, selected)
        splits[split] = {**summarize(selected), "sha256": sha256_file(destination)}
    train = [row for row in rows if row["split"] == "train"]
    original_train = [
        row for row in train if row["provenance"]["source"] not in {"hard-v1", "rule-variants-v1"}
    ]
    write_jsonl(data / "prepared" / "pilot.jsonl", balanced_sample(original_train, 1000))
    write_jsonl(data / "prepared" / "diagnostic.jsonl", balanced_sample(original_train, 64))
    review = balanced_sample(rows, 200)
    write_jsonl(data / "review" / "review-200.jsonl", review)
    write_jsonl(data / "media-manifest.jsonl", (media[key] for key in sorted(media)))
    write_json(data / "review" / "review-status.json", {
        "status": "pack-prepared-not-human-reviewed", "questions": len(review),
        "note": "Schema, metadata and media validation are not a human semantic review or model quality evaluation.",
    })
    manifest = {
        "schema_version": "1.0", "seed_policy": "sha256-keyed deterministic builders",
        "summary": summarize(rows), "splits": splits,
        "media_bytes": sum(entry["bytes"] for entry in media.values()),
        "checks": {"unique_record_ids": True, "group_split_isolation": True,
                   "exact_media_split_isolation": True, "exact_evidence_split_isolation": True,
                   "media_decoding_and_video_metadata": True, "target_membership_and_schema": True},
        "limitations": ["No model accuracy or calibration claim.",
                        "Semantic/manual review pack remains explicitly separate from automated validation.",
                        "Perceptual near-duplicate and semantic-source leakage require separate audits."]
    }
    write_json(data / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(assemble(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
