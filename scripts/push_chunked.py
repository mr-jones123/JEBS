#!/usr/bin/env python3
"""Resumable, parallel upload of a file or directory to a pod over SSH.

A slow or flaky uplink makes one long scp unreliable: a single dropped connection
restarts the whole transfer. This splits the payload into fixed-size parts, sends
each part independently with retries, skips parts the pod already holds with the
right hash, and assembles and verifies the result on the pod. Rerunning converges.

  push_chunked.py --host H --port P --key K SOURCE DESTINATION

A directory SOURCE is sent as an uncompressed tar and extracted so DESTINATION
holds its contents, matching `scp -r SOURCE DESTINATION` into a fresh path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PART_BYTES = 4 * 1024 * 1024


def ssh_base(args) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-p", str(args.port), "-i", args.key, f"root@{args.host}"]


def remote(args, command: str, data: bytes | None = None, attempts: int = 6) -> str:
    for attempt in range(1, attempts + 1):
        result = subprocess.run([*ssh_base(args), command], input=data, capture_output=True)
        if result.returncode == 0:
            return result.stdout.decode()
        time.sleep(min(30, 5 * attempt))
    raise SystemExit(f"remote command failed after {attempts} attempts: {command[:80]}\n"
                     f"{result.stderr.decode()[-400:]}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tar_directory(source: Path) -> Path:
    handle, name = tempfile.mkstemp(suffix=".tar")
    os.close(handle)
    with tarfile.open(name, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(source.rglob("*")):
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(source)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            if path.is_file():
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            else:
                archive.addfile(info)
    return Path(name)


def upload(args, payload: Path, target: str) -> str:
    size = payload.stat().st_size
    with payload.open("rb") as stream:
        whole = hashlib.sha256()
        parts = []
        while chunk := stream.read(PART_BYTES):
            whole.update(chunk)
            parts.append(sha256(chunk))
    digest = whole.hexdigest()

    existing = remote(args, f"test -f {target} && sha256sum {target} | cut -d' ' -f1 || true").strip()
    if existing == digest:
        print(f"  already present, sha256 {digest[:16]}")
        return digest

    staging = f"{target}.parts"
    remote(args, f"mkdir -p {staging} $(dirname {target})")
    held = {}
    for line in remote(args, f"cd {staging} && sha256sum part.* 2>/dev/null || true").splitlines():
        value, name = line.split()
        held[name] = value
    missing = [i for i, part in enumerate(parts) if held.get(f"part.{i:05d}") != part]
    print(f"  {size / 1e6:.1f} MB in {len(parts)} parts, {len(parts) - len(missing)} already held")

    def send(index: int) -> None:
        with payload.open("rb") as stream:
            stream.seek(index * PART_BYTES)
            chunk = stream.read(PART_BYTES)
        name = f"part.{index:05d}"
        for attempt in range(1, 9):
            result = subprocess.run(
                [*ssh_base(args), f"cat > {staging}/{name}.tmp && mv {staging}/{name}.tmp {staging}/{name}"],
                input=chunk, capture_output=True)
            if result.returncode == 0:
                return
            time.sleep(min(30, 5 * attempt))
        raise RuntimeError(f"part {index} failed after 8 attempts")

    started, done = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(send, missing):
            done += 1
            if done % 10 == 0 or done == len(missing):
                rate = done * PART_BYTES / max(1e-9, time.monotonic() - started) / 1e3
                print(f"    {done}/{len(missing)} parts, ~{rate:.0f} KB/s", flush=True)

    held = {}
    for line in remote(args, f"cd {staging} && sha256sum part.*").splitlines():
        value, name = line.split()
        held[name] = value
    bad = [i for i, part in enumerate(parts) if held.get(f"part.{i:05d}") != part]
    if bad:
        raise SystemExit(f"{len(bad)} parts failed verification; rerun to resend them")

    remote(args, f"cat {staging}/part.* > {target}.assembling && mv {target}.assembling {target}")
    assembled = remote(args, f"sha256sum {target} | cut -d' ' -f1").strip()
    if assembled != digest:
        raise SystemExit(f"assembled sha256 {assembled[:16]} does not match {digest[:16]}")
    remote(args, f"rm -rf {staging}")
    print(f"  verified sha256 {digest[:16]}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()

    if args.source.is_dir():
        archive = tar_directory(args.source)
        try:
            print(f"{args.source} -> {args.destination}/ (as tar)")
            digest = upload(args, archive, f"{args.destination}.push.tar")
            # A sibling of the destination, never inside it: sealed artifacts verify their
            # exact file inventory, so any extra file in the tree breaks loading.
            marker = f"{args.destination}.push-sha256"
            if remote(args, f"cat {marker} 2>/dev/null || true").strip() != digest:
                remote(args, f"rm -rf {args.destination} && mkdir -p {args.destination} && "
                             f"tar -xf {args.destination}.push.tar -C {args.destination} && "
                             f"echo {digest} > {marker}")
            print("  extracted")
        finally:
            archive.unlink(missing_ok=True)
    else:
        print(f"{args.source} -> {args.destination}")
        upload(args, args.source, args.destination)


if __name__ == "__main__":
    main()
