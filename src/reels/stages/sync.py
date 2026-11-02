import dataclasses
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reels import media
from reels.analysis.clock import clock_map, coarse_offset, envelope, fine_points, fit_clock
from reels.service import CAMERAS, Service, StageError, read_json, up_to_date, write_json

log = logging.getLogger(__name__)

MANUAL_KEYS = (
    "intercept_s", "slope", "drift_ppm", "offset_frames", "drift_frames_over_recording",
    "confidence", "fine_residual_ms", "windows", "fit", "manual",
)
YOUTUBE_KEYS = ("path", "fps", "width", "height", "duration_s")
SOURCE_KEYS = ("chunks", "path", "fps", "width", "height", "duration_s")
NUMERIC_KEYS = ("fps", "width", "height", "duration_s")
SUMMARY_KEYS = ("intercept_s", "drift_ppm", "confidence", "manual", "error")


@dataclass(frozen=True)
class Options:
    service: str
    force: bool = False


def summary(sources: dict) -> dict:
    return {name: {k: entry[k] for k in SUMMARY_KEYS if k in entry} for name, entry in sources.items()}


def _check_entry(name: str, entry, keys: tuple[str, ...]) -> None:
    if not isinstance(entry, dict):
        raise StageError(f"work/sync.json: {name} entry must be an object")
    missing = [k for k in keys if k not in entry]
    if missing:
        raise StageError(f"work/sync.json: {name} entry is missing {', '.join(missing)}")
    if not isinstance(entry["path"], str):
        raise StageError(f"work/sync.json: {name} path must be a string")
    for key in NUMERIC_KEYS:
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise StageError(f"work/sync.json: {name} {key} must be a positive number, got {value!r}")


def validate(data: dict) -> None:
    _check_entry("youtube", data.get("youtube"), YOUTUBE_KEYS)
    sources = data.get("sources")
    if not isinstance(sources, dict):
        raise StageError("work/sync.json: sources must be an object")
    for name, entry in sources.items():
        if name not in CAMERAS:
            raise StageError(f"work/sync.json: unknown camera {name!r}, expected one of {CAMERAS}")
        if not isinstance(entry, dict):
            raise StageError(f"work/sync.json: {name} entry must be an object")
        if entry.get("failed"):
            continue
        _check_entry(name, entry, SOURCE_KEYS)
        clock_map(entry, name)


def fresh(service: Service, cfg, data: dict) -> None:
    if "inputs" in data and data["inputs"] != fingerprint(service, cfg):
        raise StageError("sources/ or the [sync] config changed since the last sync, run reels sync first")


def joined_path(service: Service, name: str, files: list[Path]) -> Path:
    return service.work / f"{name}.joined.mp4" if len(files) > 1 else files[0]


def _rel(service: Service, path: Path) -> str:
    return path.relative_to(service.root).as_posix()


def fingerprint(service: Service, cfg, cameras: dict[str, list[Path]] | None = None) -> dict:
    if cameras is None:
        cameras = {name: files for name, files in service.camera_files().items() if files}
    return service.fingerprint(
        [service.youtube_path, *(f for files in cameras.values() for f in files)], dataclasses.asdict(cfg.sync)
    )


def _warn_audio_offset(name: str, info: media.MediaInfo) -> None:
    if abs(info.audio_offset_s) > 0.5 / info.fps:
        log.warning("%s audio starts %+.3f s after its video; the clock map follows the audio", name, info.audio_offset_s)


def _sync_camera(service: Service, cfg, name: str, joined: Path, yt: np.ndarray, yt_env: np.ndarray, yt_sr: int,
                 info: media.MediaInfo, force: bool) -> dict:
    sc = cfg.sync
    wav = media.extract_audio(joined, service.audio / f"{name}.wav", sc.sample_rate, force=force)
    cam, sr = media.load_wav(wav)
    if sr != yt_sr:
        raise StageError(f"sample rate mismatch: youtube {yt_sr} Hz, {name} {sr} Hz")
    _warn_audio_offset(name, info)
    offset, confidence = coarse_offset(envelope(cam, sc.envelope_hop), yt_env, sc.envelope_hop, sr)
    log.info("%s coarse offset %+.3f s confidence %.1f", name, offset, confidence)
    if confidence < sc.min_confidence:
        raise StageError(f"coarse confidence {confidence:.1f} below minimum {sc.min_confidence}")
    if confidence < sc.warn_confidence:
        log.warning("%s coarse confidence %.1f is low (warn below %.1f)", name, confidence, sc.warn_confidence)
    points = fine_points(cam, yt, sr, offset, sc.fine_window_s, sc.fine_every_s, sc.fine_search_s)
    fit = fit_clock(points, sc.max_window_residual_ms, sc.max_drift_ppm)
    used = sum(fit.used)
    if used >= 2:
        intercept, slope, residual_ms, kind = fit.intercept_s, fit.slope, fit.residual_ms, "windows"
        if used == 2:
            log.warning("%s only 2 fine windows agree; the drift estimate is unverified", name)
    else:
        intercept = fit.intercept_s if points else offset
        slope, residual_ms, kind = 1.0, None, "median" if points else "coarse"
        log.warning("%s no consistent fine windows (%d measured), using the %s offset with slope 1; "
                    "drift is unknown", name, len(points), "median window" if points else "coarse")
    for (yt_s, cam_s, conf), was_used in zip(points, fit.used, strict=True):
        residual = (cam_s - (intercept + slope * yt_s)) * 1000
        log.info("%s window yt %.2f s cam %.2f s conf %.1f residual %+.2f ms %s",
                 name, yt_s, cam_s, conf, residual, "used" if was_used else "dropped")
    drift_frames = abs(slope - 1) * info.duration_s * info.fps
    log.info("%s fit intercept %.4f s slope %.9f (%+.1f ppm) over %.2f frames, residual %s ms",
             name, intercept, slope, (slope - 1) * 1e6, drift_frames,
             "n/a" if residual_ms is None else f"{residual_ms:.2f}")
    if drift_frames > 1:
        log.warning("%s drifts %.2f frames over the recording", name, drift_frames)
    return {
        "intercept_s": round(intercept, 6),
        "slope": round(slope, 9),
        "drift_ppm": round((slope - 1) * 1e6, 2),
        "offset_frames": round(intercept * info.fps, 2),
        "drift_frames_over_recording": round(drift_frames, 2),
        "confidence": round(confidence, 2),
        "fine_residual_ms": None if residual_ms is None else round(residual_ms, 2),
        "windows": [
            {"yt_s": round(y, 4), "cam_s": round(c, 4), "confidence": round(k, 2), "used": u}
            for (y, c, k), u in zip(points, fit.used, strict=True)
        ],
        "fit": kind,
        "manual": False,
    }


def _derived_fresh(service: Service, cfg, cameras: dict[str, list[Path]]) -> bool:
    rate = cfg.sync.sample_rate
    if not media.audio_fresh(service.youtube_path, service.audio / "youtube.wav", rate):
        return False
    for name, files in cameras.items():
        joined = joined_path(service, name, files)
        if len(files) > 1 and not media.join_fresh(files, joined):
            return False
        if not media.audio_fresh(joined, service.audio / f"{name}.wav", rate):
            return False
    return True


def _complete(service: Service, cfg, previous: dict | None, inputs: dict, sidecar_path: Path,
              cameras: dict[str, list[Path]]) -> bool:
    if not up_to_date(previous, inputs, [sidecar_path]):
        return False
    try:
        validate(previous)
        sources = previous["sources"]
        if any(s.get("failed") for s in sources.values()):
            return False
        if not all(service.path(s["path"]).exists() for s in sources.values()):
            return False
    except StageError as error:
        log.warning("work/sync.json is not usable as is (%s), redoing sync", error)
        return False
    manual = {name for name, s in sources.items() if s.get("manual") is True}
    return _derived_fresh(service, cfg, {n: f for n, f in cameras.items() if n not in manual})


def run(opts: Options, cfg) -> dict:
    service = Service(opts.service)
    service.ensure()
    if not service.youtube_path.exists():
        raise StageError(f"missing {service.youtube_path}")
    found = service.camera_files()
    cameras = {name: files for name, files in found.items() if files}
    sidecar_path = service.work / "sync.json"
    previous = read_json(sidecar_path)
    inputs = fingerprint(service, cfg, cameras)
    if not opts.force and _complete(service, cfg, previous, inputs, sidecar_path, cameras):
        log.info("work/sync.json up to date")
        return {"ok": True, "sync": str(sidecar_path), "up_to_date": True, "sources": summary(previous["sources"])}
    if not cameras:
        log.warning("no camera sources found in %s", service.sources)

    yt_info = media.probe(service.youtube_path)
    _warn_audio_offset("youtube", yt_info)
    wav = media.extract_audio(service.youtube_path, service.audio / "youtube.wav", cfg.sync.sample_rate,
                              force=opts.force)
    yt, yt_sr = media.load_wav(wav)
    yt_env = envelope(yt, cfg.sync.envelope_hop)
    old_sources = (previous or {}).get("sources", {})
    for name, files in found.items():
        if files:
            continue
        if old_sources.get(name, {}).get("manual") is True:
            log.warning("%s: no files under sources/, its manual sync entry is dropped", name)
        else:
            log.info("%s: no files under sources/, skipped", name)
    sources: dict[str, dict] = {}
    failures = 0
    for name, files in cameras.items():
        old = old_sources.get(name, {})
        manual = old.get("manual") is True
        entry: dict = {"chunks": [_rel(service, f) for f in files]}
        try:
            joined = media.join_chunks(files, joined_path(service, name, files), force=opts.force)
            if len(files) > 1:
                log.info("%s joined %d chunks into %s", name, len(files), _rel(service, joined))
            info = media.probe(joined)
            entry.update({
                "path": _rel(service, joined),
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "duration_s": round(info.duration_s, 3),
            })
            if manual:
                clock_map(old, name)
                if old.get("chunks") != entry["chunks"]:
                    raise StageError(f"chunks changed since the manual entry was written ({old.get('chunks')} -> "
                                     f"{entry['chunks']}); re-enter the offset or remove \"manual\"")
                entry.update({k: old[k] for k in MANUAL_KEYS if k in old})
                log.info("%s manual entry kept", name)
            else:
                entry.update(_sync_camera(service, cfg, name, joined, yt, yt_env, yt_sr, info, opts.force))
        except (StageError, OSError, ValueError) as error:
            log.error("%s sync failed: %s", name, error)
            entry = {**old, **entry} if manual else entry
            entry.update({"failed": True, "error": str(error)})
            failures += 1
        sources[name] = entry

    write_json(sidecar_path, {
        "version": 1,
        "inputs": inputs,
        "youtube": {
            "path": _rel(service, service.youtube_path),
            "fps": yt_info.fps,
            "width": yt_info.width,
            "height": yt_info.height,
            "duration_s": round(yt_info.duration_s, 3),
        },
        "sources": sources,
    })
    return {"ok": not failures, "sync": str(sidecar_path), "up_to_date": False, "sources": summary(sources)}
