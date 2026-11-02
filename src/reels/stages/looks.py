import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from reels import media
from reels.analysis import people
from reels.media.detect import Detector, grab
from reels.service import Service, StageError, read_json
from reels.stages.render import validate_moment
from reels.timeline.looks import ROTATIONS, Look, to_toml

log = logging.getLogger(__name__)

EQUIRECT_WIDTH = 3840
FLAT_WIDTH = 1280
HEADER = [
    "written by reels looks from the people it found; edit freely, reels looks only rewrites it with --force",
    "rename looks, fix yaw (degrees, see looks.jpg), drop or reorder the pattern, then run reels reel",
]


@dataclass(frozen=True)
class Options:
    service: str
    # one moment; None for every folder under cuts/
    slug: str | None = None
    force: bool = False
    samples: int = 5


def is_equirect(info: media.MediaInfo) -> bool:
    return info.width == 2 * info.height


def sample_times(moment: dict, count: int) -> list[float]:
    start = min(moment["handle_s"], moment["start_s"])
    length = moment["end_s"] - moment["start_s"]
    return [start + length * (i + 0.5) / count for i in range(count)]


def _flat_look(detector: Detector, camera: str, frames: list[Path]) -> Look:
    scores = {r: sum(detector.upright_score(f, r) for f in frames) for r in ROTATIONS}
    rotate = people.best_rotation(scores)
    log.info("%s: rotate %d (person scores %s)", camera, rotate, ", ".join(f"{r}: {s:.1f}" for r, s in scores.items()))
    return Look(camera, rotate=rotate)


def _views(detector: Detector, camera: str, frames: list[Path], prefix: str, preview: Path) -> dict[str, Look]:
    found, boxes_mid = [], []
    for i, frame in enumerate(frames):
        boxes, width, height = detector.equirect(frame)
        found += people.detections(i, boxes, width, height)
        if i == len(frames) // 2:
            boxes_mid = boxes
    seen = people.people(found, len(frames))
    views = {}
    for n, person in enumerate(seen, start=1):
        views[f"{prefix}person{n}"] = Look(camera, yaw=person.yaw, pitch=person.pitch, fov=person.fov, move=True)
        log.info("%s person%d at yaw %+.0f pitch %+.0f, fov %.0f, in %.0f%% of frames",
                 camera, n, person.yaw, person.pitch, person.fov, person.share * 100)
    yaw, fov = people.wide(seen)
    views[f"{prefix}wide"] = Look(camera, yaw=yaw, pitch=0.0, fov=fov, move=True)
    if not seen:
        log.warning("%s: nobody found in %d frames, only a wide view", camera, len(frames))
    detector.preview(frames[len(frames) // 2], [(n, v.yaw) for n, v in views.items()], boxes_mid, preview)
    return views


def _moment_looks(detector: Detector, folder: Path, moment: dict, samples: int) -> tuple[dict, dict]:
    times = sample_times(moment, samples)
    sources = {cam: folder / f for cam, f in moment["files"].items() if cam != "youtube" and (folder / f).exists()}
    infos = {cam: media.probe(path) for cam, path in sources.items()}
    spherical = [cam for cam in sources if is_equirect(infos[cam])]
    flat, views = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for cam, path in sources.items():
            work = Path(tmp) / cam
            work.mkdir()
            if cam in spherical:
                prefix = f"{cam}-" if len(spherical) > 1 else ""
                preview = folder / (f"looks-{cam}.jpg" if prefix else "looks.jpg")
                views.update(_views(detector, cam, grab(path, times, EQUIRECT_WIDTH, work), prefix, preview))
            else:
                flat[cam] = _flat_look(detector, cam, grab(path, times[1:-1] or times, FLAT_WIDTH, work))
    looks = {**flat, **views}
    return looks, {"default": people.default_pattern(list(flat), list(views))}


def run(opts: Options, cfg) -> dict:
    service = Service(opts.service)
    folders = [service.cuts / opts.slug] if opts.slug else sorted(p.parent for p in service.cuts.glob("*/moment.json"))
    if not folders:
        raise StageError("nothing under cuts/, run reels cut first")
    detector = None
    written, kept = {}, []
    for folder in folders:
        slug = folder.name
        target = folder / "looks.toml"
        if target.exists() and not opts.force:
            log.info("cuts/%s/looks.toml exists, kept (--force rewrites it)", slug)
            kept.append(slug)
            continue
        moment = read_json(folder / "moment.json")
        if moment is None:
            raise StageError(f"cuts/{slug}/moment.json not found, run reels cut first")
        validate_moment(moment, f"cuts/{slug}/moment.json")
        detector = detector or Detector()
        looks, patterns = _moment_looks(detector, folder, moment, opts.samples)
        target.write_text(to_toml(looks, patterns, HEADER))
        log.info("cuts/%s/looks.toml written: %s", slug, ", ".join(looks) or "no camera looks")
        written[slug] = {"looks": sorted(looks), "pattern": patterns["default"]}
    return {"ok": True, "written": written, "kept": kept}
