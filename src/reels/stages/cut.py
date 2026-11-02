import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

from reels import media
from reels.analysis import beats, notes
from reels.service import CAMERAS, Service, StageError, UsageError, read_json, up_to_date, write_json
from reels.stages import sync

log = logging.getLogger(__name__)

OWNED_FILES = frozenset(f"{cam}.mov" for cam in CAMERAS) | {"youtube.mov"}


@dataclass(frozen=True)
class Options:
    service: str
    notes: Path | None = None
    force: bool = False
    dry_run: bool = False


def plan_sources(win0: float, win1: float, sources: dict) -> tuple[dict[str, float], dict[str, str]]:
    starts: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for cam, entry in sources.items():
        if entry.get("failed"):
            skipped[cam] = f"sync failed: {entry.get('error')}"
            continue
        clock = sync.clock_map(entry, cam)
        c0, c1 = clock.cam_time(win0), clock.cam_time(win1)
        where = f"window [{win0:.1f}, {win1:.1f}] maps to cam [{c0:.1f}, {c1:.1f}]"
        if c0 < 0:
            skipped[cam] = f"{where}, recording starts at 0"
        elif max(c1, c0 + (win1 - win0)) > entry["duration_s"]:
            skipped[cam] = f"{where}, recording ends at {entry['duration_s']:.1f}"
        else:
            starts[cam] = c0
    return starts, skipped


def _window(moment, len_s: float, duration_s: float, wav: Path, cc) -> tuple[float, float, beats.Grid]:
    if moment.noted_s >= duration_s:
        raise StageError(f"note at {moment.noted_s:.0f} s is past the end of the stream ({duration_s:.0f} s)")
    r0, r1 = beats.region(moment.noted_s, len_s, duration_s)
    y, sr = media.read_wav_region(wav, r0, r1)
    grid = beats.grid(y, sr, r0, cc.beats_per_bar)
    start_s, end_s = beats.snap(grid, moment.noted_s, len_s, cc.snap_to_bars)
    if end_s <= start_s:
        start_s, end_s = moment.noted_s, moment.noted_s + len_s
    return start_s, min(end_s, duration_s), grid


def _cut_moment(service, cfg, moment, slug, sync_data, wav, previous_beats, force, dry_run=False) -> dict:
    cc = cfg.cut
    folder = service.cuts / slug
    len_s = moment.len_s or cc.default_len_s
    sources = sync_data["sources"]
    inputs_files = [service.youtube_path, service.work / "sync.json"]
    inputs_files += [service.path(s["path"]) for s in sources.values() if s.get("path")]
    params = {
        "note": {"noted_s": moment.noted_s, "label": moment.label, "len_s": moment.len_s},
        "cut": dataclasses.asdict(cc),
    }
    inputs = service.fingerprint(inputs_files, params)

    sidecar = read_json(folder / "moment.json") or {}
    previous_files = _owned(sidecar.get("files"))
    if (not force and sidecar and slug in previous_beats
            and up_to_date(sidecar, inputs, [folder / name for name in previous_files])):
        log.info("cuts/%s up to date", slug)
        return previous_beats[slug]

    duration_s = sync_data["youtube"]["duration_s"]
    start_s, end_s, grid = _window(moment, len_s, duration_s, wav, cc)
    log.info("%s: noted %.2f -> start %.2f end %.2f at %.1f bpm", slug, moment.noted_s, start_s, end_s, grid.bpm)
    win0 = max(0.0, start_s - cc.handle_s)
    win1 = min(duration_s, end_s + cc.handle_s)

    starts, skipped = plan_sources(win0, win1, sources)
    for cam, reason in skipped.items():
        log.warning("%s: skip %s: %s", slug, cam, reason)

    duration = win1 - win0
    if dry_run:
        plan = [("youtube.mov", media.cut_args(service.youtube_path, None, win0, duration, cc.codec, cc.bitrate_mbps))]
        plan += [(f"{cam}.mov", media.cut_args(service.path(sources[cam]["path"]), service.youtube_path, cam_start,
                                                duration, cc.codec, cc.bitrate_mbps, audio_start_s=win0))
                 for cam, cam_start in starts.items()]
        for name, cut_args in plan:
            log.info("dry run cuts/%s/%s: ffmpeg %s %s", slug, name, " ".join(cut_args), name)
        return {"bpm": grid.bpm, "beats_s": grid.beats_s, "downbeats_s": grid.downbeats_s,
                "noted_s": moment.noted_s, "start_s": start_s, "end_s": end_s}
    folder.mkdir(parents=True, exist_ok=True)
    if sidecar:
        write_json(folder / "moment.json", {**sidecar, "inputs": None})
    files = {"youtube": "youtube.mov"}
    media.cut(service.youtube_path, None, win0, duration, folder / "youtube.mov", cc.codec, cc.bitrate_mbps)
    log.info("cuts/%s/youtube.mov written", slug)
    for cam, cam_start in starts.items():
        media.cut(service.path(sources[cam]["path"]), service.youtube_path, cam_start, duration,
                  folder / f"{cam}.mov", cc.codec, cc.bitrate_mbps, audio_start_s=win0)
        files[cam] = f"{cam}.mov"
        log.info("cuts/%s/%s.mov written", slug, cam)
    for stale in sorted((previous_files | _owned_on_disk(folder)) - set(files.values())):
        (folder / stale).unlink(missing_ok=True)
        log.info("cuts/%s/%s removed, not part of this cut", slug, stale)

    write_json(folder / "moment.json", {
        "slug": slug, "label": moment.label, "person": moment.person,
        "noted_s": moment.noted_s, "start_s": start_s, "end_s": end_s, "handle_s": cc.handle_s,
        "bpm": grid.bpm, "files": files, "skipped": skipped, "inputs": inputs,
    })
    return {
        "bpm": grid.bpm, "beats_s": grid.beats_s, "downbeats_s": grid.downbeats_s,
        "noted_s": moment.noted_s, "start_s": start_s, "end_s": end_s,
    }


def _owned(files) -> set[str]:
    if not isinstance(files, dict):
        return set()
    return {name for name in files.values() if isinstance(name, str) and name in OWNED_FILES}


def _owned_on_disk(folder: Path) -> set[str]:
    return {name for name in OWNED_FILES if (folder / name).exists()}


def run(opts: Options, cfg) -> dict:
    service = Service(opts.service)
    service.ensure()
    sync_data = read_json(service.work / "sync.json")
    if sync_data is None:
        raise StageError("run reels sync first")
    sync.validate(sync_data)
    sync.fresh(service, cfg, sync_data)
    notes_path = Path(opts.notes) if opts.notes else service.notes / "moments.txt"
    if not notes_path.exists():
        raise UsageError(f"missing notes file {notes_path}")
    try:
        moments = notes.parse(notes_path.read_text(), cfg.cut.people, cfg.cut.min_len_s)
    except notes.NotesError as error:
        raise UsageError(str(error)) from error

    seen: dict[str, int] = {}
    for moment in moments:
        name = notes.slug(moment)
        if name in seen:
            raise StageError(f"duplicate moment {name} on note lines {seen[name]} and {moment.line}")
        seen[name] = moment.line

    previous_beats = (read_json(service.work / "beats.json") or {}).get("moments", {})
    wav = media.extract_audio(service.youtube_path, service.audio / "youtube.wav", cfg.sync.sample_rate)
    entries: dict[str, dict] = {}
    failed: dict[str, str] = {}
    for moment in moments:
        slug = notes.slug(moment)
        try:
            entries[slug] = _cut_moment(service, cfg, moment, slug, sync_data, wav, previous_beats, opts.force,
                                        opts.dry_run)
        except (StageError, OSError, ValueError) as error:
            log.error("cuts/%s failed: %s (left out of beats.json until it cuts cleanly)", slug, error)
            failed[slug] = str(error)

    for slug in sorted(set(previous_beats) - set(seen)):
        log.info("cuts/%s no longer in notes, dropped from beats.json, folder kept", slug)
    result = {"ok": not failed, "cut": sorted(entries), "failed": failed, "dry_run": opts.dry_run}
    if opts.dry_run:
        log.info("dry run: %d moment(s) planned, nothing written", len(entries))
        return result
    write_json(service.work / "beats.json", {"version": 1, "moments": entries})
    return result
