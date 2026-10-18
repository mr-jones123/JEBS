"""Build and exercise source-isolated data and executable-code archives."""

import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from PIL import Image

from mdprep.assemble import inspect_video, load_rows, media_paths
from mdprep.common import SPLITS, sha256_file, write_json, write_jsonl
from mdprep.native import NativeCompiler, validate_contract
from mdprep.records import parse_record, validate_record
from mdprep.verify import validate_verification_contract

TRAIN_SPLITS = ("train", "dev", "calibration", "policy")
TEST_SPLITS = ("test_id", "test_ood")
SUPPORT_FILES = {
    "raw/visuals/atkinson-hyperlegible/OFL.txt": {"synthetic-visuals-v1"},
    "raw/visuals/atkinson-hyperlegible/AtkinsonHyperlegible-Regular.ttf": {
        "synthetic-visuals-v1"
    },
    "raw/banking77/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE": {
        "banking77"
    },
    "raw/hard-v1/LICENSE": {"hard-v1"},
}
SOURCE_FILES = {
    "sources/banking77.json": {"banking77", "mdprep_text_rules"},
    "sources/open_images_v7_photos.json": {"open_images_v7_train_human_verified"},
    "sources/visuals.json": None,
    "sources/videos.json": {"generated_temporal_videos"},
    "sources/hard-v1.json": {"hard-v1"},
    "sources/rule-variants-v1.json": {"rule-variants-v1"},
}
CODE_DOCUMENTS = (
    "README.md", "RUNPOD.md", "DATA_CARD.md", "VERIFICATION_PLAN.md",
    "SUBAGENT_TRAINING_PLAN.md", "AGENT_CONTRACTS.md", "decisions.tsv",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _archive(root: Path, name: str, payload: dict[str, Path]) -> dict:
    destination = root / "dist" / f"{name}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".building")
    members = {}
    try:
        with temporary.open("wb") as output, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for relative, path in sorted(payload.items()):
                    name_path = PurePosixPath(relative)
                    if name_path.is_absolute() or ".." in name_path.parts or path.is_symlink() or not path.is_file():
                        raise ValueError(f"Unsafe or missing archive member: {relative}")
                    members[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination.relative_to(root)), "files": len(members),
            "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "members": members}


def _project_contract(contract: dict, contract_path: Path, scope: tuple[str, ...], rows: list[dict], media: set[str]) -> dict:
    projected = copy.deepcopy(contract)
    ids = {row["id"] for row in rows}
    projected["qualification"] = {
        "contract_sha256": sha256_file(contract_path),
        "questions_checked": contract["questions_checked"],
        "scope": contract.get("scope", list(SPLITS)),
    }
    projected["scope"] = list(scope)
    projected["questions_checked"] = len(ids)
    projected["prepared_hashes"] = {split: contract["prepared_hashes"][split] for split in scope}
    projected["media_hashes"] = {path: contract["media_hashes"][path] for path in sorted(media)}
    if "row_receipts" in projected:
        projected["row_receipts"] = {key: value for key, value in projected["row_receipts"].items() if key in ids}
    if "derived_hashes" in projected:
        derived = {"prepared/pilot.jsonl", "prepared/diagnostic.jsonl"} if "train" in scope else set()
        projected["derived_hashes"] = {key: value for key, value in projected["derived_hashes"].items() if key in derived}
    return projected


def _source_projections(data: Path, rows: list[dict], scope: tuple[str, ...]) -> dict[str, dict]:
    projections = {}
    for relative, declared_sources in SOURCE_FILES.items():
        original_path = data / relative
        original = _json(original_path)
        sources = declared_sources or {original["source"]}
        selected = [row["provenance"] for row in rows if row["provenance"]["source"] in sources]
        if not selected:
            continue
        attributions = {}
        for provenance in selected:
            encoded = json.dumps(provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            attributions[encoded] = provenance
        safe_metadata = {key: original[key] for key in (
            "source", "revision", "repository", "upstream_split", "license", "license_url",
            "annotation_license", "official_download_page", "image_license_caveat", "processing",
            "font", "font_license", "known_limits", "known_limitations",
        ) if key in original}
        projections[relative] = {
            "scope": list(scope), "source_manifest_sha256": sha256_file(original_path),
            "source_metadata": safe_metadata,
            "attributions": [attributions[key] for key in sorted(attributions)],
        }
    return projections


def make_bundle(root: Path, name: str, splits: tuple[str, ...]) -> dict:
    data = root / "data"
    processor = validate_contract(root)
    verification = validate_verification_contract(root)
    scope = tuple(split for split in splits if split in SPLITS)
    rows = [row for split in scope for row in load_rows(data / "prepared" / f"{split}.jsonl")]
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("Primary bundle splits contain repeated IDs")
    media = {relative for row in rows for relative in media_paths(row)}
    payload = {f"data/{relative}": data / relative for relative in sorted(media)}
    for split in splits:
        path = data / "prepared" / f"{split}.jsonl"
        selected = load_rows(path)
        if not {row["id"] for row in selected} <= ids:
            raise ValueError(f"Derived split is not a subset of primary scope: {split}")
        payload[f"data/prepared/{split}.jsonl"] = path
    represented_sources = {row["provenance"]["source"] for row in rows}
    for relative, sources in SUPPORT_FILES.items():
        if represented_sources & sources:
            payload[f"data/{relative}"] = data / relative
    if "train" in scope:
        forbidden = {row["id"] for split in TEST_SPLITS
                     for row in load_rows(data / "prepared" / f"{split}.jsonl")}
        if ids & forbidden:
            raise ValueError("Training archive includes final-test IDs")
    else:
        forbidden = set()
    with tempfile.TemporaryDirectory(prefix="decision-bundle-") as temporary:
        staging = Path(temporary)
        processor_scope = _project_contract(processor, data / "processor-contract.json", scope, rows, media)
        verification_scope = _project_contract(verification, data / "verification-contract.json", scope, rows, media)
        checks = [entry for entry in load_rows(data / "processor-checks.jsonl") if entry["id"] in ids]
        if len(checks) != len(ids) or {entry["id"] for entry in checks} != ids:
            raise ValueError("Processor checks do not cover the archive scope exactly")
        write_jsonl(staging / "data/processor-checks.jsonl", checks)
        checks_hash = sha256_file(staging / "data/processor-checks.jsonl")
        processor_scope["checks_sha256"] = checks_hash
        if "slices" in processor_scope:
            slices = {}
            for field in ("modality", "origin"):
                slices[field] = {}
                for value in sorted({entry[field] for entry in checks}):
                    lengths = sorted(entry["positions"] for entry in checks if entry[field] == value)
                    slices[field][value] = {"questions": len(lengths), "max_positions": lengths[-1],
                                            "p95_positions": lengths[min(len(lengths)-1, int(.95*len(lengths)))]}
            processor_scope["slices"] = slices
        for certificate in (processor_scope, verification_scope):
            if "qualification_sha256" in certificate:
                unsigned = {key: value for key, value in certificate.items() if key != "qualification_sha256"}
                canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                certificate["qualification_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        documents = {
            "data/processor-contract.json": processor_scope,
            "data/verification-contract.json": verification_scope,
            "data/manifest.json": {
                "schema_version": 1, "scope": list(scope), "questions": len(rows),
                "corpus_manifest_sha256": sha256_file(data / "manifest.json"),
                "splits": {split: _json(data / "manifest.json")["splits"][split] for split in scope},
                "prepared_hashes": processor_scope["prepared_hashes"],
                "derived_hashes": {split: sha256_file(data / "prepared" / f"{split}.jsonl")
                                   for split in splits if split not in scope},
                "origin_counts": dict(collections.Counter(row["provenance"]["origin"] for row in rows)),
                "human_review": verification.get("human_review", {"status": "NOT_VERIFIED"}),
            },
            "data/media-manifest.json": {path: {"sha256": sha256_file(data / path),
                                                  "bytes": (data / path).stat().st_size} for path in sorted(media)},
            "data/qualification.json": {
                "processor": processor_scope["qualification"],
                "data": verification_scope["qualification"],
                "forbidden_evaluation_ids": sorted(forbidden),
            },
        }
        documents.update({f"data/{path}": value for path, value in _source_projections(data, rows, scope).items()})
        for relative, value in documents.items():
            write_json(staging / relative, value)
        for path in staging.rglob("*"):
            if path.is_file():
                payload[path.relative_to(staging).as_posix()] = path
        result = _archive(root, name, payload)
    result.update({"splits": list(splits), "primary_scope": list(scope), "questions": len(rows),
                   "unique_media": len(media)})
    return result


def make_code_bundle(root: Path) -> dict:
    payload = {name: root / name for name in ("pyproject.toml", "uv.lock", *CODE_DOCUMENTS)}
    for directory in ("src", "tests", "tools"):
        for path in (root / directory).rglob("*.py"):
            if "__pycache__" not in path.parts:
                payload[path.relative_to(root).as_posix()] = path
    if not any(name.startswith("src/mdtrain/") for name in payload):
        raise ValueError("The executable training package is missing")
    return _archive(root, "training-code", payload)


def _extract(archive_path: Path, destination: Path, expected: dict) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f"Archive inventory mismatch: {archive_path}")
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive entry: {member.name}")
        archive.extractall(destination, filter="data")
    for relative, facts in expected.items():
        path = destination / relative
        if path.stat().st_size != facts["bytes"] or sha256_file(path) != facts["sha256"]:
            raise ValueError(f"Extracted member changed: {relative}")


def verify_bundles(root: Path, index_path: Path, include_locked: bool = False, smoke: bool = False) -> dict:
    index = _json(index_path)
    required = ("code", "training", "locked_evaluation") if include_locked else ("code", "training")
    reports = {}
    with tempfile.TemporaryDirectory(prefix="decision-extracted-") as temporary:
        scratch = Path(temporary)
        for key in required:
            entry = index[key]
            archive_path = (root / entry["path"]).resolve()
            if not archive_path.is_relative_to(root.resolve()) or sha256_file(archive_path) != entry["sha256"]:
                raise ValueError(f"Archive checksum failed: {key}")
            _extract(archive_path, scratch / key, entry["members"])
        install_target = scratch / "installed"
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target",
                        str(install_target), str(scratch / "code")], check=True, capture_output=True, text=True)
        environment = {**os.environ, "PYTHONPATH": str(install_target),
                       "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", str(root / "data/raw/processor-cache"))}
        for key in required:
            if key == "code":
                continue
            bundle_root = scratch / key
            scope = index[key]["primary_scope"]
            shutil.copytree(scratch / "code", bundle_root, dirs_exist_ok=True)
            rows = [row for split in scope for row in load_rows(bundle_root / "data/prepared" / f"{split}.jsonl")]
            ids = {row["id"] for row in rows}
            if len(rows) != index[key]["questions"] or len(ids) != len(rows):
                raise ValueError(f"Extracted question identity/count failed: {key}")
            references = {path for row in rows for path in media_paths(row)}
            forbidden = set(_json(bundle_root / "data/qualification.json")["forbidden_evaluation_ids"])
            if ids & forbidden:
                raise ValueError("Final-test records leaked into the training archive")
            for row in rows:
                validate_record(row)
            for relative in references:
                path = bundle_root / "data" / relative
                if path.suffix == ".mp4":
                    inspect_video(path)
                else:
                    with Image.open(path) as image:
                        image.load()
            if key == "training":
                expected_rows = {f"data/prepared/{split}.jsonl" for split in (*TRAIN_SPLITS, "pilot", "diagnostic")}
                actual_rows = {name for name in index[key]["members"] if name.startswith("data/prepared/")}
                if actual_rows != expected_rows:
                    raise ValueError("Training archive has unexpected prepared splits")
            code = (
                "import json\nimport sys\nfrom pathlib import Path\nimport mdprep\n"
                "from mdprep.assemble import load_rows\nfrom mdprep.records import parse_record\n"
                "from mdprep.native import NativeCompiler,validate_contract\n"
                "from mdprep.verify import validate_verification_contract\n"
                "r=Path(sys.argv[1])\n"
                "assert Path(mdprep.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())\n"
                "validate_contract(r)\nvalidate_verification_contract(r)\n"
                "rows=[x for s in json.loads((r/'data/processor-contract.json').read_text())['scope'] "
                "for x in load_rows(r/'data/prepared'/f'{s}.jsonl')]\n"
                "samples={}\nfor x in rows:\n    samples.setdefault(x['modality'],x)\n"
                "c=NativeCompiler.load(r/'data/processor-contract.json',media_root=r/'data')\n"
                "c.compile(tuple(parse_record(x).input for x in samples.values()))\n"
                "print(json.dumps({'input_kinds':sorted(samples),'questions':len(rows)}))\n"
            )
            try:
                consumed = subprocess.run([sys.executable, "-c", code, str(bundle_root), str(install_target)], cwd=scratch / "code",
                                          env=environment, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Installed {key} consumer failed with exit {error.returncode}.\n"
                    f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
                ) from error
            reports[key] = {"questions": len(rows), "media_decoded": len(references),
                            "consumer_stdout": consumed.stdout.strip(), "status": "VERIFIED"}
            if key == "training" and smoke:
                smoke_path = scratch / "extracted-tiny-smoke.json"
                subprocess.run([sys.executable, "-m", "mdtrain", "smoke", "--root", str(bundle_root),
                                "--kind", "tiny", "--output", str(smoke_path)], cwd=scratch / "code",
                               env=environment, check=True)
                reports[key]["tiny_smoke"] = _json(smoke_path)
    result = {"status": "VERIFIED", "scope": list(required), "bundles": reports,
              "cuda_verified": False, "human_review_performed": False}
    write_json(root / "dist/package-verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", nargs="?", choices=("build", "verify"), default="build")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--index", type=Path)
    parser.add_argument("--include-locked", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.operation == "verify":
        result = verify_bundles(root, args.index or root / "dist/bundles.json", args.include_locked, args.smoke)
    else:
        result = {
            "schema_version": 1,
            "training": make_bundle(root, "training-data", (*TRAIN_SPLITS, "pilot", "diagnostic")),
            "locked_evaluation": make_bundle(root, "locked-evaluation-data", TEST_SPLITS),
            "code": make_code_bundle(root),
            "instructions": "Transfer code and training data first. Keep locked evaluation separate until weights and calibration are frozen.",
        }
        write_json(root / "dist/bundles.json", result)
        checksum_groups = {"training": ("code", "training"), "locked": ("locked_evaluation",)}
        for name, keys in checksum_groups.items():
            lines = "".join(f"{result[key]['sha256']}  {result[key]['path']}\n" for key in keys)
            (root / "dist" / f"SHA256SUMS.{name}").write_text(lines)
    summary = {key: ({field: value for field, value in entry.items() if field != "members"}
                     if isinstance(entry, dict) else entry) for key, entry in result.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
