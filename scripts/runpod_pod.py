#!/usr/bin/env python3
"""Plan, create, inspect, and remove the network volume and GPU pod for a paid run.

Every command that spends money or deletes storage runs only with --yes; without it
the command prints the exact request it would send. The API key is read from
~/.runpod-key and never printed.

A network volume is used because RunPod preserves it when a pod stops, is
terminated, or is stopped for a zero balance, while container and pod volume disks
are lost in those cases. Network volumes exist only in Secure Cloud and are bound to
one datacenter, so the pod must be placed in the volume's datacenter.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
KEY_FILE = Path.home() / ".runpod-key"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_GPU = "NVIDIA GeForce RTX 3090"
NETWORK_VOLUME_USD_PER_GB_MONTH = 0.07


def _key() -> str:
    if not KEY_FILE.is_file():
        raise SystemExit(f"RunPod key not found: {KEY_FILE}")
    return KEY_FILE.read_text().strip()


def _request(method: str, url: str, body: dict | None = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    # Cloudflare in front of the API rejects Python's default urllib user agent (error 1010).
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
        "User-Agent": "jebs-runpod-pod/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {url} failed with HTTP {error.code}: {error.read().decode()[:500]}")
    return json.loads(payload) if payload else None


def _graphql(query: str) -> dict:
    result = _request("POST", GRAPHQL, {"query": query})
    if result.get("errors"):
        raise SystemExit(f"GraphQL error: {result['errors']}")
    return result["data"]


def availability(gpu: str) -> list[dict]:
    """Secure Cloud datacenters that support network volumes, with this GPU's stock."""
    centers = _graphql("query { dataCenters { id storageSupport listed "
                       "gpuAvailability { gpuTypeId stockStatus } } }")["dataCenters"]
    rows = []
    for center in centers:
        stock = {entry["gpuTypeId"]: entry.get("stockStatus") for entry in center.get("gpuAvailability") or []}
        if center.get("storageSupport") and center.get("listed") and gpu in stock:
            rows.append({"dataCenterId": center["id"], "stock": stock[gpu]})
    return sorted(rows, key=lambda row: {"High": 0, "Medium": 1, "Low": 2}.get(row["stock"] or "", 3))


def balance() -> dict:
    return _graphql("query { myself { clientBalance currentSpendPerHr } }")["myself"]


def plan(args: argparse.Namespace) -> None:
    rows = availability(args.gpu)
    money = balance()
    print(json.dumps({
        "gpu": args.gpu,
        "datacenters_with_network_volumes_and_this_gpu": rows,
        "existing_network_volumes": _request("GET", f"{REST}/networkvolumes"),
        "balance_usd": round(money["clientBalance"], 2),
        "current_spend_usd_per_hour": money["currentSpendPerHr"],
        "volume_cost_usd_per_day": round(args.size * NETWORK_VOLUME_USD_PER_GB_MONTH / 30, 3),
    }, indent=2))
    if not any(row["stock"] for row in rows):
        print(f"No Secure Cloud datacenter with network volumes reports {args.gpu} stock now.", file=sys.stderr)


def _act(args: argparse.Namespace, method: str, url: str, body: dict | None) -> object:
    if not args.yes:
        print(f"dry run (add --yes to send): {method} {url}\n{json.dumps(body, indent=2) if body else ''}")
        return None
    return _request(method, url, body)


def create_volume(args: argparse.Namespace) -> None:
    rows = {row["dataCenterId"]: row["stock"] for row in availability(args.gpu)}
    if args.datacenter not in rows:
        raise SystemExit(f"{args.datacenter} does not offer {args.gpu} with network volumes: {rows}")
    if not rows[args.datacenter] and not args.allow_no_stock:
        raise SystemExit(f"{args.datacenter} reports no {args.gpu} stock; refusing to create a volume there")
    result = _act(args, "POST", f"{REST}/networkvolumes",
                  {"name": args.name, "size": args.size, "dataCenterId": args.datacenter})
    if result is not None:
        print(json.dumps(result, indent=2))


def create_pod(args: argparse.Namespace) -> None:
    volumes = {volume["id"]: volume for volume in _request("GET", f"{REST}/networkvolumes")}
    if args.volume_id not in volumes:
        raise SystemExit(f"network volume {args.volume_id} not found: {sorted(volumes)}")
    datacenter = volumes[args.volume_id]["dataCenterId"]
    body = {
        "name": args.name,
        "imageName": IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "containerDiskInGb": args.container_disk,
        "networkVolumeId": args.volume_id,
        "volumeMountPath": "/workspace",
        "dataCenterIds": [datacenter],
        "ports": ["22/tcp"],
    }
    pod = _act(args, "POST", f"{REST}/pods", body)
    if pod is None:
        return
    print(f"created pod {pod['id']} in {datacenter}; waiting for SSH")
    for _ in range(60):
        current = _request("GET", f"{REST}/pods/{pod['id']}")
        mapping = current.get("portMappings") or {}
        if current.get("publicIp") and "22" in mapping:
            print(json.dumps({"pod_id": pod["id"], "host": current["publicIp"], "port": mapping["22"],
                              "costPerHr": current.get("costPerHr"), "lastStartedAt": current.get("lastStartedAt")},
                             indent=2))
            return
        time.sleep(10)
    raise SystemExit(f"pod {pod['id']} has no SSH mapping after 10 minutes; inspect it before retrying")


def status(args: argparse.Namespace) -> None:
    pods = [{key: pod.get(key) for key in ("id", "name", "desiredStatus", "costPerHr", "networkVolumeId",
                                             "publicIp", "portMappings")}
            for pod in _request("GET", f"{REST}/pods")]
    print(json.dumps({"pods": pods, "network_volumes": _request("GET", f"{REST}/networkvolumes"),
                      "balance": balance()}, indent=2))


def delete_volume(args: argparse.Namespace) -> None:
    attached = [pod["id"] for pod in _request("GET", f"{REST}/pods") if pod.get("networkVolumeId") == args.volume_id]
    if attached:
        raise SystemExit(f"volume {args.volume_id} is still attached to pods {attached}; terminate them first")
    _act(args, "DELETE", f"{REST}/networkvolumes/{args.volume_id}", None)
    if args.yes:
        print(f"deleted network volume {args.volume_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "create-volume", "create-pod", "status", "delete-volume"):
        command = commands.add_parser(name)
        command.add_argument("--gpu", default=DEFAULT_GPU)
        if name in ("create-volume", "create-pod", "delete-volume"):
            command.add_argument("--yes", action="store_true", help="send the request; spends money or deletes data")
        if name in ("plan", "create-volume"):
            command.add_argument("--size", type=int, default=50, help="network volume size in GB")
        if name == "create-volume":
            command.add_argument("--name", default="jebs-v3")
            command.add_argument("--datacenter", required=True)
            command.add_argument("--allow-no-stock", action="store_true")
        if name == "create-pod":
            command.add_argument("--volume-id", required=True)
            command.add_argument("--name", default="jebs-v3")
            command.add_argument("--container-disk", type=int, default=30)
        if name == "delete-volume":
            command.add_argument("--volume-id", required=True)
    args = parser.parse_args()
    {"plan": plan, "create-volume": create_volume, "create-pod": create_pod,
     "status": status, "delete-volume": delete_volume}[args.command](args)


if __name__ == "__main__":
    main()
