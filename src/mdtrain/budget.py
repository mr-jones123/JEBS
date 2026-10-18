"""Durable elapsed allowance, not a provider billing or termination guarantee."""

import datetime as dt
import json
import math
import time
import uuid
from pathlib import Path

from mdtrain.deadline import stop_requested
from mdtrain.storage import atomic_json, writer_lock

GPU_BUDGET_USD = 8
RESERVE_SECONDS = 600


class BudgetStopped(RuntimeError):
    pass


def allowance_seconds(hourly_usd: float) -> int:
    """Elapsed seconds the fixed GPU budget buys at this allocation's hourly price."""
    if isinstance(hourly_usd, bool) or not isinstance(hourly_usd, (int, float)) \
            or not math.isfinite(hourly_usd) or hourly_usd <= 0:
        raise ValueError("Hourly price must be a positive finite number")
    return math.floor(GPU_BUDGET_USD / hourly_usd * 3600)


def utc_timestamp(value: str) -> float:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("Allocation timestamp must be RFC3339 UTC")
    return parsed.timestamp()


class BudgetLedger:
    def __init__(self, path: Path, *, paid_started_at: str | None = None, hourly_usd: float | None = None):
        self.path = path
        self.paid_started_at = paid_started_at
        self.hourly_usd = hourly_usd
        self.record = None
        self.lock = None

    def __enter__(self):
        self.lock = writer_lock(self.path.with_suffix(self.path.suffix + ".lock"))
        self.lock.__enter__()
        try:
            now = time.time()
            if self.path.exists():
                self.record = json.loads(self.path.read_text())
                if (self.record["gpu_budget_usd"] != GPU_BUDGET_USD
                        or self.record["allowance_seconds"] != allowance_seconds(self.record["hourly_usd"])):
                    raise ValueError("Ledger allowance identity differs")
                if self.hourly_usd is not None and self.hourly_usd != self.record["hourly_usd"]:
                    raise ValueError("Ledger hourly price differs from this allocation")
                if now < self.record["last_observed_utc"]:
                    raise ValueError("UTC clock regressed across processes")
                if self.paid_started_at is not None:
                    start = utc_timestamp(self.paid_started_at)
                    current = self.record["allocations"][-1]
                    if start != current["started_utc"]:
                        if not current.get("closed_utc") or start < current["closed_utc"]:
                            raise ValueError("Close the previous allocation explicitly before adding another")
                        self.record["allocations"].append({"started_utc": start, "closed_utc": None})
            else:
                if self.paid_started_at is None or self.hourly_usd is None:
                    raise ValueError("A new ledger requires the actual paid allocation start in UTC and its hourly price")
                self.record = {"id": uuid.uuid4().hex, "allowance_seconds": allowance_seconds(self.hourly_usd),
                               "gpu_budget_usd": GPU_BUDGET_USD, "hourly_usd": self.hourly_usd,
                               "allocations": [{"started_utc": utc_timestamp(self.paid_started_at),
                                                "closed_utc": None}],
                               "last_observed_utc": now, "elapsed_seconds": 0,
                               "reserve_seconds": RESERVE_SECONDS, "receipts": {},
                               "billing_guarantee": False}
            if self.record["allocations"][-1]["closed_utc"] is not None:
                raise ValueError("This allocation is closed; preflight must add a new allocation")
            if self.record["allocations"][-1]["started_utc"] > now:
                raise ValueError("Allocation cannot begin in the future")
            self.anchor_utc, self.anchor_mono = now, time.monotonic()
            self.commit()
            return self
        except BaseException:
            self.lock.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        try:
            self.commit()
        finally:
            self.lock.__exit__(*args)

    def elapsed(self) -> float:
        now = time.time()
        monotonic_now = self.anchor_utc + time.monotonic() - self.anchor_mono
        if now < self.record["last_observed_utc"] - .05:
            raise ValueError("UTC clock regressed during operation")
        effective = max(now, monotonic_now)
        elapsed = sum((entry["closed_utc"] or effective) - entry["started_utc"]
                      for entry in self.record["allocations"])
        return max(elapsed, self.record["elapsed_seconds"])

    @property
    def remaining(self) -> float:
        return self.record["allowance_seconds"] - self.elapsed()

    def admit(self, seconds: float) -> None:
        if stop_requested():
            raise BudgetStopped("Guardian or operator requested a boundary stop")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid projected operation duration")
        remaining = min(self.remaining,
                        self.record.get("process_deadline_elapsed", self.record["allowance_seconds"]) - self.elapsed())
        if seconds + self.record["reserve_seconds"] >= remaining:
            raise BudgetStopped("Remaining allowance cannot admit this complete operation plus save reserve")

    def commit(self) -> None:
        self.record["elapsed_seconds"] = self.elapsed()
        self.record["last_observed_utc"] = time.time()
        atomic_json(self.path, self.record)

    def receipt(self, key: str, value: dict) -> None:
        self.record["receipts"][key] = value
        self.commit()

    def observe_save(self, seconds: float) -> None:
        self.record["reserve_seconds"] = max(self.record["reserve_seconds"], math.ceil(seconds * 2 + 60))
        self.commit()

    def set_process_deadline(self, elapsed_limit: float) -> None:
        self.record["process_deadline_elapsed"] = min(
            self.record.get("process_deadline_elapsed", self.record["allowance_seconds"]), elapsed_limit)
        self.commit()

    def close_allocation(self) -> None:
        self.commit()
        self.record["allocations"][-1]["closed_utc"] = time.time()
        self.commit()
