import dataclasses
import logging
import os
from dataclasses import dataclass

from reels.service import Service, StageError, read_json, up_to_date, write_json
from reels.stages import sync
from reels.timeline import fcpxml

log = logging.getLogger(__name__)

MOMENT_KEYS = ("slug", "label", "start_s", "end_s", "handle_s", "files", "skipped", "inputs")


@dataclass(frozen=True)
class Options:
    service: str
    mode: str = "fcpxml"
    force: bool = False
    dry_run: bool = False


def validate_moment(moment: dict, where: str) -> None:
    missing = [k for k in MOMENT_KEYS if k not in moment]
    if missing:
        raise StageError(f"{where} is missing {', '.join(missing)}, run reels cut first")
    if not isinstance(moment["files"], dict) or not isinstance(moment["skipped"], dict):
        raise StageError(f"{where}: files and skipped must be objects")
    for key in ("start_s", "end_s", "handle_s"):
        if isinstance(moment[key], bool) or not isinstance(moment[key], (int, float)):
            raise StageError(f"{where}: {key} must be a number")


def _source_paths(service: Service, sync_data: dict) -> list:
    paths = [service.path(sync_data["youtube"]["path"])]
    paths += [service.path(s["path"]) for s in sync_data["sources"].values() if not s.get("failed") and s.get("path")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise StageError(f"missing source media {missing[0]}, run reels sync first")
    return paths


def _check_moment_fresh(slug: str, moment: dict, current: dict) -> None:
    recorded = moment["inputs"]["files"] if isinstance(moment["inputs"], dict) else {}
    stale = [name for name, stamp in current.items() if recorded.get(name) != stamp]
    if stale:
        raise StageError(f"cuts/{slug} was cut against an older {stale[0]} or its last cut was interrupted, "
                         "run reels cut first")


def run(opts: Options, cfg) -> dict:
    if opts.mode != "fcpxml":
        raise StageError(f"mode {opts.mode} is not built yet")
    service = Service(opts.service)
    service.ensure()
    sync_path = service.work / "sync.json"
    sync_data = read_json(sync_path)
    if sync_data is None:
        raise StageError("run reels sync first")
    sync.validate(sync_data)
    sync.fresh(service, cfg, sync_data)
    beats = read_json(service.work / "beats.json")
    if beats is None:
        raise StageError("run reels cut first")
    sources = _source_paths(service, sync_data)
    current = service.fingerprint([sync_path, *sources], {})["files"]
    moment_paths = []
    moments = []
    for path in sorted(service.cuts.glob("*/moment.json")):
        moment = read_json(path)
        slug = path.parent.name
        validate_moment(moment, f"cuts/{slug}/moment.json")
        if moment["slug"] not in beats.get("moments", {}):
            log.info("cuts/%s not in work/beats.json (removed from notes or its last cut failed), skipped", slug)
            continue
        _check_moment_fresh(slug, moment, current)
        moment_paths.append(path)
        moments.append(moment)
    moments.sort(key=lambda m: m["start_s"])
    if not moments:
        log.warning("no moments in cuts/, writing the per-service project only")

    output = service.out / "timeline.fcpxml"
    sidecar_path = service.work / "render.json"
    params = {**dataclasses.asdict(cfg.fcpxml), "root": str(service.root.resolve())}
    inputs = service.fingerprint([sync_path, *sources, *moment_paths], params)
    result = {"ok": True, "output": str(output), "moments": len(moments)}
    if not opts.force and up_to_date(read_json(sidecar_path), inputs, [output]):
        log.info("out/timeline.fcpxml up to date")
        return {**result, "up_to_date": True}

    timeline = fcpxml.build(service, sync_data, moments, cfg.fcpxml)
    for note in timeline.notes:
        log.info(note)
    if opts.dry_run:
        log.info("dry run: %d project(s), %d moment(s), %d bytes of FCPXML, nothing written",
                 timeline.xml.count("<project "), len(moments), len(timeline.xml.encode()))
        return {"ok": True, "dry_run": True, "projects": timeline.xml.count("<project "), "moments": len(moments)}
    tmp = output.with_name(output.name + ".part")
    tmp.write_text(timeline.xml, encoding="utf-8")
    os.replace(tmp, output)
    write_json(sidecar_path, {"version": 1, "inputs": inputs})
    log.info("wrote %s", output)
    return {**result, "up_to_date": False}
