import math
from dataclasses import dataclass

from reels.timeline.looks import h_fov

SAME_PERSON_DEG = 12
# a person must show up in this share of the sampled frames; drops people walking past
MIN_SHARE = 0.4
# vertical view = this many times the person's height, so head room and hands stay in frame
FRAME_PER_PERSON = 1.8
MIN_FOV, MAX_FOV = 20.0, 100.0
WIDE_MARGIN_DEG = 30


@dataclass(frozen=True)
class Detection:
    frame: int
    yaw: float
    pitch: float
    # angular height of the box; degrees
    height: float


@dataclass(frozen=True)
class Person:
    yaw: float
    pitch: float
    fov: float
    share: float


def detections(frame: int, boxes: list[tuple[float, float, float, float]], width: int, height: int) -> list[Detection]:
    """Boxes are x0, y0, x1, y1 in equirect pixels; x may run past width when the frame was padded at the seam."""
    out = []
    for x0, y0, x1, y1 in boxes:
        cx = ((x0 + x1) / 2) % width
        out.append(Detection(frame, cx / width * 360 - 180, 90 - (y0 + y1) / 2 / height * 180, (y1 - y0) / height * 180))
    return out


def _wrap(deg: float) -> float:
    return (deg + 180) % 360 - 180


def _circular_mean(degs: list[float]) -> float:
    return math.degrees(math.atan2(sum(math.sin(math.radians(d)) for d in degs),
                                   sum(math.cos(math.radians(d)) for d in degs)))


def _groups(found: list[Detection]) -> list[list[Detection]]:
    ordered = sorted(found, key=lambda d: d.yaw)
    groups: list[list[Detection]] = []
    for d in ordered:
        if groups and d.yaw - groups[-1][-1].yaw <= SAME_PERSON_DEG:
            groups[-1].append(d)
        else:
            groups.append([d])
    if len(groups) > 1 and groups[0][0].yaw + 360 - groups[-1][-1].yaw <= SAME_PERSON_DEG:
        groups[0] = groups.pop() + groups[0]
    return groups


def people(found: list[Detection], frames: int) -> list[Person]:
    out = []
    for group in _groups(found):
        share = len({d.frame for d in group}) / frames
        if share < MIN_SHARE:
            continue
        heights = sorted(d.height for d in group)
        pitches = sorted(d.pitch for d in group)
        fov = h_fov(min(179.0, heights[len(heights) // 2] * FRAME_PER_PERSON))
        out.append(Person(yaw=_circular_mean([d.yaw for d in group]), pitch=pitches[len(pitches) // 2],
                          fov=min(MAX_FOV, max(MIN_FOV, fov)), share=share))
    return sorted(out, key=lambda p: p.yaw)


def wide(found: list[Person]) -> tuple[float, float]:
    """Yaw and horizontal fov of the narrowest view that holds everyone."""
    if not found:
        return 0.0, 90.0
    yaws = sorted(p.yaw for p in found)
    gaps = [(yaws[(i + 1) % len(yaws)] - yaws[i]) % 360 or 360 for i in range(len(yaws))]
    widest = max(range(len(gaps)), key=gaps.__getitem__)
    first = yaws[(widest + 1) % len(yaws)]
    span = 360 - gaps[widest] if len(yaws) > 1 else 0
    return _wrap(first + span / 2), min(MAX_FOV, span + 2 * WIDE_MARGIN_DEG)


def best_rotation(scores: dict[int, float]) -> int:
    """The rotation under which the detector is most sure it sees upright people; 0 on a tie."""
    return max(sorted(scores), key=lambda r: (scores[r], r == 0))


def default_pattern(flat: list[str], views: list[str]) -> list[str]:
    if not flat:
        return [*views, "master"]
    out = []
    for i, view in enumerate(views):
        out += [flat[i % len(flat)], view]
    return [*(out or flat), "master"]
