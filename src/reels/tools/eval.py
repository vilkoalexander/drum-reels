import logging
from dataclasses import dataclass
from pathlib import Path

from reels.service import Service, StageError, read_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Options:
    service: str
    truth: Path | None = None
    max_ms: float = 5.0
    max_ppm: float = 10.0


def compare(sync_data: dict, truth: dict) -> dict[str, dict]:
    report = {}
    for cam, want in truth.items():
        entry = sync_data.get("sources", {}).get(cam)
        if entry is None or entry.get("failed"):
            report[cam] = {"error": "not synced" if entry is None else entry.get("error", "failed")}
            continue
        report[cam] = {
            "intercept_error_ms": (entry["intercept_s"] - want["intercept_s"]) * 1000,
            "drift_error_ppm": (entry["slope"] - want["slope"]) * 1e6,
            "fit": entry.get("fit"),
        }
    return report


def run(opts: Options, cfg) -> dict:
    service = Service(opts.service)
    sync_data = read_json(service.work / "sync.json")
    if sync_data is None:
        raise StageError("run reels sync first")
    truth_path = Path(opts.truth) if opts.truth else service.root / "truth.json"
    truth = read_json(truth_path)
    if truth is None:
        raise StageError(f"missing {truth_path}")
    report = compare(sync_data, truth)
    bad = 0
    for cam, result in report.items():
        if "error" in result:
            log.error("%s: %s", cam, result["error"])
            bad += 1
            continue
        ok = abs(result["intercept_error_ms"]) <= opts.max_ms and abs(result["drift_error_ppm"]) <= opts.max_ppm
        log.log(logging.INFO if ok else logging.ERROR, "%s: intercept error %+.2f ms, drift error %+.2f ppm, fit %s%s",
                cam, result["intercept_error_ms"], result["drift_error_ppm"], result["fit"], "" if ok else " (OUT)")
        bad += 0 if ok else 1
    return {"ok": not bad, "sources": report}
