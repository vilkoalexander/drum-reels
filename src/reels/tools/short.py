import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from reels import media
from reels.analysis import beats
from reels.service import StageError, write_json
from reels.stages.reel import check_remotion, remotion_render, render_look
from reels.timeline import reel
from reels.timeline.looks import SRC, check_grade, look_filter

log = logging.getLogger(__name__)

SAMPLE_RATE = 22050
BEATS_PER_BAR = 4


@dataclass(frozen=True)
class Options:
    video: str
    seconds: float = 20.0
    # start near this second instead of at the busiest bars
    start: float | None = None
    out: str | None = None
    name: str | None = None
    grade: str = "none"
    title: str | None = None
    caption: str | None = None


def best_window(downbeats_s: list[float], times_s: np.ndarray, strength: np.ndarray, bars: int) -> int:
    """Index of the downbeat that starts the busiest run of `bars` bars."""
    if len(downbeats_s) <= bars:
        raise StageError(f"only {len(downbeats_s) - 1} whole bars found, need {bars}")
    best, best_score = 0, -1.0
    for i in range(len(downbeats_s) - bars):
        inside = (times_s >= downbeats_s[i]) & (times_s < downbeats_s[i + bars])
        score = float(strength[inside].mean()) if inside.any() else 0.0
        if score > best_score:
            best, best_score = i, score
    return best


def extend_back(downbeats_s: list[float], bar_s: float) -> list[float]:
    """The beat tracker often locks on a few bars in; carry its grid back to the start of the file."""
    earlier = []
    t = downbeats_s[0] - bar_s
    while t >= 0:
        earlier.append(t)
        t -= bar_s
    return earlier[::-1] + list(downbeats_s)


def nearest_downbeat(downbeats_s: list[float], start_s: float, bars: int) -> int:
    usable = len(downbeats_s) - bars
    if usable <= 0:
        raise StageError(f"only {len(downbeats_s) - 1} whole bars found, need {bars}")
    return int(np.argmin(np.abs(np.asarray(downbeats_s[:usable]) - start_s)))


def run(opts: Options, cfg) -> dict:
    src = Path(opts.video)
    if not src.is_file():
        raise StageError(f"{src} not found")
    check_grade(opts.grade)
    check_remotion()
    out_dir = Path(opts.out) if opts.out else src.parent / "shorts"
    name = opts.name or src.stem.replace(" ", "_")
    work = out_dir / ".work" / name
    work.mkdir(parents=True, exist_ok=True)

    wav = media.extract_audio(src, work / "analysis.wav", SAMPLE_RATE)
    y, sr = media.load_wav(wav)
    grid = beats.grid(y, sr, 0.0, BEATS_PER_BAR)
    if grid.bpm <= 0:
        raise StageError(f"no beat grid in {src.name}, cannot cut on bars")
    bar_s = float(np.median(np.diff(grid.downbeats_s)))
    downbeats = extend_back(grid.downbeats_s, bar_s)
    bars = max(BEATS_PER_BAR, round(opts.seconds / bar_s))
    strength = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(strength, sr=sr)
    if opts.start is None:
        first = best_window(downbeats, times, strength, bars)
    else:
        first = nearest_downbeat(downbeats, opts.start, bars)
    start_s = downbeats[first]
    reel_plan = reel.single_plan(start_s, bars, bar_s)
    log.info("%s: %.1f bpm, keeping %d bars from %.2f s (%.1f s)", src.name, grid.bpm, bars, start_s,
             reel_plan.length_s)

    render_look(src, start_s, reel_plan.length_s, look_filter(SRC, opts.grade), work / "src.mp4")
    media.write_atomic(
        ["-ss", f"{start_s:.3f}", "-t", f"{reel_plan.length_s:.3f}", "-i", str(src)],
        ["-vn", "-c:a", "pcm_s16le", "-ar", str(media.AUDIO_RATE), "-f", "wav"],
        work / "audio.wav",
    )
    props_path = work / "props.json"
    write_json(props_path, reel.props(reel_plan, {"src": SRC}, opts.title, opts.caption, {}))
    output = out_dir / f"{name}.mp4"
    remotion_render(props_path.resolve(), work.resolve(), output.resolve())
    log.info("%s written", output)
    return {"ok": True, "output": str(output), "start_s": round(start_s, 2), "seconds": round(reel_plan.length_s, 2),
            "bpm": round(grid.bpm, 1)}
