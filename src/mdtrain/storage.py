"""Checksummed local publication and single-writer run ownership."""

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from mdprep.common import sha256_file


def _inventory_identity(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def json_safe(value: object) -> object:
    """Normalize third-party values into JSON-serializable structures.

    Foreign libraries hand back containers that truthiness checks accept but the
    strict publisher rejects. Transformers 5.19, for example, reports
    loading_info keys such as missing_keys as sets, and an empty set is falsy, so
    a completeness check passes and the later write is what fails. Sets become
    sorted lists, and an unsupported leaf becomes its repr so a report stays
    inspectable instead of failing the artifact write.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def report_json(value: dict) -> dict:
    """Encode reports as {original fields, nonfinite_values: {JSON pointer: status}}.

    Infinite numeric fields become null. Status is positive_infinity or
    negative_infinity at an RFC 6901 pointer into the original report.
    Finite fields remain unchanged. NaN is an error, never a missing value.
    """
    if "nonfinite_values" in value:
        raise ValueError("Report already contains reserved nonfinite_values metadata")
    nonfinite = {}

    def convert(item, pointer):
        if isinstance(item, float):
            if math.isnan(item):
                raise ValueError(f"NaN report value at {pointer}")
            if math.isinf(item):
                nonfinite[pointer] = "positive_infinity" if item > 0 else "negative_infinity"
                return None
        if isinstance(item, dict):
            return {key: convert(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))
                    for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child, pointer + "/" + str(index)) for index, child in enumerate(item)]
        return item

    return {**convert(value, ""), "nonfinite_values": nonfinite}

def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(directory: Path) -> str:
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "checksums.json":
            if path.is_symlink():
                raise ValueError("Artifact members cannot be links")
            hashes[path.relative_to(directory).as_posix()] = sha256_file(path)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    identity = _inventory_identity(hashes)
    atomic_json(directory / "checksums.json", {"identity": identity, "files": hashes})
    sync_directory(directory)
    return identity


def verify_seal(directory: Path) -> str:
    record = json.loads((directory / "checksums.json").read_text())
    hashes = record["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name != "checksums.json"}
    if actual != set(hashes) or _inventory_identity(hashes) != record["identity"]:
        raise ValueError("Artifact inventory or identity mismatch")
    for relative, expected in hashes.items():
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe artifact path")
        if sha256_file(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    return record["identity"]


@contextlib.contextmanager
def writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another process owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
