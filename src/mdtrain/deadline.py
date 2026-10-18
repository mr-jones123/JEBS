"""Local process deadlines, never provider billing control.

The public CLI initializes the ledger, releases its lock, then starts the paid
worker in a new session. Only that worker owns the ledger lock during paid work.
The guardian reads atomic ledger snapshots without importing the model runtime.
It sends SIGTERM at the save reserve and SIGKILL to the entire worker process
group one second before the cumulative allowance. A blocked update may lose
unsaved work, but cannot replace its last published recovery checkpoint.
"""

import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_stop_requested = False


def request_stop(signum, frame) -> None:
    global _stop_requested
    _stop_requested = True


def stop_requested() -> bool:
    return _stop_requested


@dataclass(frozen=True)
class ProcessDeadline:
    hard_monotonic: float
    reserve_seconds: float


def ledger_deadline(record: dict, *, now_utc: float, now_monotonic: float) -> ProcessDeadline:
    """Translate cumulative allocation wall time into a nonextendable deadline."""
    elapsed = max(record["elapsed_seconds"], sum(
        (entry["closed_utc"] if entry["closed_utc"] is not None else now_utc) - entry["started_utc"]
        for entry in record["allocations"]))
    limit = min(record["allowance_seconds"], record.get("process_deadline_elapsed", record["allowance_seconds"]))
    remaining = limit - elapsed - 1.0
    reserve = record["reserve_seconds"]
    if not math.isfinite(remaining) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Invalid ledger process deadline")
    return ProcessDeadline(now_monotonic + remaining, reserve)


def _signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def supervise(command: list[str], *, deadline: ProcessDeadline, ledger_path: Path | None = None,
              ledger_id: str | None = None) -> dict:
    """Run one local worker session, with a literal deadline usable by CPU tests."""
    if deadline.hard_monotonic <= time.monotonic():
        return {"returncode": None, "graceful_requested": False, "hard_stopped": True,
                "timer_is_billing_guarantee": False}
    hard, reserve = deadline.hard_monotonic, deadline.reserve_seconds
    graceful = False
    hard_stopped = False
    child = subprocess.Popen(command, start_new_session=True)
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while child.poll() is None:
            if ledger_path is not None:
                record = json.loads(ledger_path.read_text())
                if record["id"] != ledger_id:
                    raise ValueError("Guardian ledger identity changed")
                observed = ledger_deadline(record, now_utc=time.time(), now_monotonic=time.monotonic())
                hard = min(hard, observed.hard_monotonic)
                reserve = max(reserve, observed.reserve_seconds)
            now = time.monotonic()
            if now >= hard:
                _signal_group(child.pid, signal.SIGKILL)
                hard_stopped = True
                break
            if not graceful and (interrupted or now >= hard - reserve):
                try:
                    os.kill(child.pid, signal.SIGINT if interrupted else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                graceful = True
            time.sleep(min(.05, max(0., hard - now)))
        returncode = child.wait()
        # Descendants must not outlive a completed paid worker, either.
        if not hard_stopped:
            _signal_group(child.pid, signal.SIGKILL)
        return {"returncode": returncode, "graceful_requested": graceful,
                "hard_stopped": hard_stopped, "timer_is_billing_guarantee": False}
    finally:
        if child.poll() is None:
            _signal_group(child.pid, signal.SIGKILL)
            child.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)
