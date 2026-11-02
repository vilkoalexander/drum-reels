import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reels import media
from reels.service import Service, StageError, read_json, write_json
from reels.stages.render import validate_moment
from reels.timeline import looks as looks_file
from reels.timeline import reel

log = logging.getLogger(__name__)

REMOTION_DIR = Path(__file__).resolve().parents[3] / "remotion"
LOOK_BITRATE_MBPS = 30


@dataclass(frozen=True)
class Options:
    service: str
    slug: str
    # a pattern name from cuts/<slug>/looks.toml
    pattern: str = "default"
    skip_bars: int = 0
    # 0 means the whole moment
    bars: int = 0
    grade: str = "film"
    title: str | None = None
    caption: str | None = None
    look_text: Sequence[str] = ()
    name: str | None = None


def render_look(src: Path, start_s: float, length_s: float, vf: str, dst: Path) -> None:
    log.info("writing %s", dst.name)
    media.write_atomic(
        ["-ss", f"{start_s:.3f}", "-t", f"{length_s:.3f}", "-i", str(src)],
        ["-vf", f"{vf},fps={reel.FPS},setsar=1,format=yuv420p", "-an", "-c:v", "h264_videotoolbox",
         "-b:v", f"{LOOK_BITRATE_MBPS}M", "-allow_sw", "1", "-movflags", "+faststart", "-f", "mp4"],
        dst,
    )


def check_remotion() -> str:
    npx = shutil.which("npx")
    if npx is None:
        raise StageError("npx not found, install Node to render reels")
    if not (REMOTION_DIR / "node_modules").is_dir():
        raise StageError(f"run npm install in {REMOTION_DIR} first")
    return npx


def remotion_render(props_path: Path, public_dir: Path, output: Path) -> None:
    cmd = [check_remotion(), "remotion", "render", "src/index.ts", "Reel", str(output), f"--props={props_path}",
           f"--public-dir={public_dir}", "--log=error"]
    log.debug("run %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REMOTION_DIR, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise StageError(f"remotion render failed ({result.returncode}): {result.stderr.strip()[-2000:]}")


def _look_texts(pairs: Sequence[str], names) -> dict[str, str]:
    texts = {}
    for pair in pairs:
        look, sep, text = pair.partition("=")
        if not sep or look not in names:
            raise StageError(f"--look-text {pair!r} must be LOOK=TEXT with LOOK one of {', '.join(names)}")
        texts[look] = text
    return texts


def run(opts: Options, cfg) -> dict:
    looks_file.check_grade(opts.grade)
    check_remotion()
    service = Service(opts.service)
    folder = service.cuts / opts.slug
    moment = read_json(folder / "moment.json")
    if moment is None:
        raise StageError(f"cuts/{opts.slug}/moment.json not found, run reels cut first")
    validate_moment(moment, f"cuts/{opts.slug}/moment.json")
    looks_path = folder / "looks.toml"
    if not looks_path.exists():
        raise StageError(f"cuts/{opts.slug}/looks.toml not found, run reels looks {opts.service} {opts.slug} first")
    looks, patterns = looks_file.parse(looks_path.read_text(), f"cuts/{opts.slug}/looks.toml")
    if opts.pattern not in patterns:
        raise StageError(f"no pattern {opts.pattern!r} in cuts/{opts.slug}/looks.toml, it has {', '.join(patterns)}")
    texts = _look_texts(opts.look_text, looks)
    cameras = {camera for camera, f in moment["files"].items() if (folder / f).exists()}
    reel_plan = reel.plan(moment, patterns[opts.pattern], looks, opts.skip_bars, opts.bars, cameras)
    name = opts.name or f"{opts.slug}_{opts.pattern}"
    work = service.work / "reels" / name
    work.mkdir(parents=True, exist_ok=True)

    for look in sorted({shot.look for shot in reel_plan.shots}):
        src = folder / moment["files"][looks[look].camera]
        render_look(src, reel_plan.start_s, reel_plan.length_s, looks_file.look_filter(looks[look], opts.grade),
                    work / f"{look}.mp4")
    media.write_atomic(
        ["-ss", f"{reel_plan.start_s:.3f}", "-t", f"{reel_plan.length_s:.3f}", "-i",
         str(folder / moment["files"]["youtube"])],
        ["-vn", "-c:a", "pcm_s16le", "-ar", str(media.AUDIO_RATE), "-f", "wav"],
        work / "audio.wav",
    )
    props_path = work / "props.json"
    write_json(props_path, reel.props(reel_plan, looks, opts.title, opts.caption, texts))

    output = service.out / "reels" / f"{name}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info("rendering out/reels/%s.mp4: %.1f s, %d shots", name, reel_plan.length_s, len(reel_plan.shots))
    remotion_render(props_path.resolve(), work.resolve(), output.resolve())
    log.info("out/reels/%s.mp4 written", name)
    return {"ok": True, "output": str(output), "seconds": round(reel_plan.length_s, 2), "shots": len(reel_plan.shots)}
