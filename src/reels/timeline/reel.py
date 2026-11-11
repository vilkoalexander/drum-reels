from dataclasses import dataclass, replace

from reels.service import StageError
from reels.subtitles import Caption
from reels.timeline.looks import HEIGHT, WIDTH, Look

FPS = 30
BARS_PER_SHOT = 2


@dataclass(frozen=True)
class Shot:
    look: str
    from_frame: int
    frames: int


@dataclass(frozen=True)
class Plan:
    start_s: float
    length_s: float
    frames: int
    shots: list[Shot]


def plan(moment: dict, names: list[str], looks: dict[str, Look], skip_bars: int, bars: int,
         cameras: set[str]) -> Plan:
    bpm = moment.get("bpm") or 0
    if bpm <= 0:
        raise StageError(f"cuts/{moment['slug']} has no beat grid, cannot cut on bars")
    bar = 4 * 60 / bpm
    # cut clamps the leading handle at 0, so a moment near the stream start has a shorter one
    start = min(moment["handle_s"], moment["start_s"]) + skip_bars * bar
    length = moment["end_s"] - moment["start_s"] - skip_bars * bar
    if bars:
        length = min(length, bars * bar)
    if length <= 0:
        raise StageError(f"--skip-bars {skip_bars} leaves nothing of cuts/{moment['slug']}")
    names = [n if looks[n].camera in cameras else "master" for n in names]
    return Plan(start, length, round(length * FPS), _shots(names, bar, round(length * FPS)))


def single_plan(start_s: float, bars: int, bar_s: float) -> Plan:
    length = bars * bar_s
    return Plan(start_s, length, round(length * FPS), _shots(["src"], bar_s, round(length * FPS)))


def _shots(names: list[str], bar_s: float, total: int) -> list[Shot]:
    shots, i, at = [], 0, 0
    while at < total:
        end = min(total, round((i + 1) * BARS_PER_SHOT * bar_s * FPS))
        shots.append(Shot(names[i % len(names)], at, end - at))
        at, i = end, i + 1
    return shots


TITLE_DELAY, TITLE_FRAMES = 4, 100
CAPTION_DELAY = 18
MIN_SUBTITLE_S = 0.25
STICKER_DELAY, STICKER_TAIL = 4, 6


def window_subtitles(captions: list[Caption], start_s: float, length_s: float) -> list[Caption]:
    """Captions overlapping the window, on the window's own clock and clipped to it, one at a time."""
    inside = []
    for c in sorted(captions, key=lambda c: c.start):
        start, end = max(c.start, start_s) - start_s, min(c.end, start_s + length_s) - start_s
        if inside and start < inside[-1].end:
            inside[-1] = replace(inside[-1], end=max(inside[-1].start, start))
        if end - start > 0:
            inside.append(replace(c, start=start, end=end))
    return [c for c in inside if c.end - c.start >= MIN_SUBTITLE_S]


def props(reel_plan: Plan, looks: dict[str, Look], title: str | None, caption: str | None,
          look_texts: dict[str, str], subtitles: list[Caption] = ()) -> dict:
    shots, texts, pushes = [], [], 0
    for shot in reel_plan.shots:
        move = None
        if looks[shot.look].move:
            move = "in" if pushes % 2 == 0 else "out"
            pushes += 1
        shots.append({"src": f"{shot.look}.mp4", "startFrom": shot.from_frame, "from": shot.from_frame,
                      "duration": shot.frames, "move": move})
        sticker = look_texts.get(shot.look)
        if sticker and shot.frames > STICKER_DELAY + STICKER_TAIL:
            texts.append({"text": sticker, "from": shot.from_frame + STICKER_DELAY,
                          "duration": shot.frames - STICKER_DELAY - STICKER_TAIL, "kind": "sticker"})
    if title and reel_plan.frames > TITLE_DELAY:
        texts.append({"text": title, "from": TITLE_DELAY, "duration": min(TITLE_FRAMES, reel_plan.frames - TITLE_DELAY),
                      "kind": "title"})
    if caption and reel_plan.frames > CAPTION_DELAY:
        texts.append({"text": caption, "from": CAPTION_DELAY, "duration": reel_plan.frames - CAPTION_DELAY,
                      "kind": "caption"})
    for c in subtitles:
        first, last = round(c.start * FPS), min(reel_plan.frames, round(c.end * FPS))
        if last > first:
            texts.append({"text": c.text, "from": first, "duration": last - first, "kind": "subtitle"})
    return {"fps": FPS, "width": WIDTH, "height": HEIGHT, "durationInFrames": reel_plan.frames,
            "audio": "audio.wav", "shots": shots, "texts": texts}
