import json
import math
import re
import tomllib
from dataclasses import dataclass

from reels.service import StageError

WIDTH, HEIGHT = 1080, 1920
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ROTATIONS = {0: "", 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}
CROP_9_16 = f"crop='min(iw,ih*{WIDTH}/{HEIGHT})':'min(ih,iw*{HEIGHT}/{WIDTH})',scale={WIDTH}:{HEIGHT}"


@dataclass(frozen=True)
class Look:
    camera: str
    # degrees clockwise, applied after the file's own rotation tag
    rotate: int = 0
    # set only for a view into a 360 source; degrees
    yaw: float | None = None
    pitch: float = 0.0
    # horizontal, degrees
    fov: float = 0.0
    move: bool = False


MASTER = Look("youtube")
SRC = Look("src", move=True)

GRADES = {
    "none": "",
    "film": ("eq=contrast=1.2:saturation=0.75:brightness=-0.02,"
             "colorbalance=rs=-0.03:bs=0.06:rh=0.07:gh=0.03:bh=-0.06,"
             "curves=all='0/0.05 0.5/0.48 1/0.95',vignette=PI/5"),
    "punchy": ("eq=contrast=1.12:saturation=1.15:gamma=0.97,"
               "colorbalance=rs=0.06:bs=-0.08:rm=0.04:bm=-0.05:rh=0.03:bh=-0.03,"
               "curves=all='0/0.03 0.25/0.2 0.75/0.82 1/1'"),
    "mono": "hue=s=0,eq=contrast=1.3:brightness=-0.03,curves=all='0/0.02 0.3/0.22 0.7/0.8 1/1',vignette=PI/5",
}


def check_grade(grade: str) -> None:
    if grade not in GRADES:
        raise StageError(f"unknown grade {grade!r}, pick one of {', '.join(GRADES)}")


def look_filter(look: Look, grade: str = "none") -> str:
    check_grade(grade)
    return ",".join(f for f in (_framing(look), GRADES[grade]) if f)


def v_fov(h_fov: float) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(h_fov) / 2) * HEIGHT / WIDTH))


def h_fov(v_fov_deg: float) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(v_fov_deg) / 2) * WIDTH / HEIGHT))


def _framing(look: Look) -> str:
    if look.yaw is None:
        return ",".join(f for f in (ROTATIONS[look.rotate], CROP_9_16) if f)
    yaw = (look.yaw + 180) % 360 - 180
    return (f"v360=input=e:output=flat:h_fov={look.fov:g}:v_fov={v_fov(look.fov):.2f}:yaw={yaw:g}"
            f":pitch={look.pitch:g}:w={WIDTH}:h={HEIGHT}:interp=lanczos")


def parse(text: str, where: str) -> tuple[dict[str, Look], dict[str, list[str]]]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise StageError(f"{where}: {error}") from error
    looks = {"master": MASTER}
    for name, table in (data.get("look") or {}).items():
        looks[name] = _look(name, table, f"{where} [look.{name}]")
    patterns = data.get("pattern") or {}
    if not patterns:
        raise StageError(f"{where} has no [pattern] table")
    for name, names in patterns.items():
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise StageError(f"{where}: pattern {name} must be a non-empty list of look names")
        unknown = [n for n in names if n not in looks]
        if unknown:
            raise StageError(f"{where}: pattern {name} uses unknown look {unknown[0]!r}")
    return looks, patterns


def _look(name: str, table, where: str) -> Look:
    if not NAME.match(name):
        raise StageError(f"{where}: look names are lower case letters, digits, - and _")
    if not isinstance(table, dict) or not isinstance(table.get("camera"), str):
        raise StageError(f"{where} needs camera = \"<source name>\"")
    unknown = set(table) - {"camera", "rotate", "yaw", "pitch", "fov", "move"}
    if unknown:
        raise StageError(f"{where}: unknown key {min(unknown)}")
    for key in ("yaw", "pitch", "fov"):
        if key in table and (isinstance(table[key], bool) or not isinstance(table[key], (int, float))):
            raise StageError(f"{where}: {key} must be a number")
    if table.get("rotate", 0) not in ROTATIONS:
        raise StageError(f"{where}: rotate must be one of {', '.join(map(str, ROTATIONS))}")
    is_view = "yaw" in table
    if is_view and not 0 < table.get("fov", 0) < 180:
        raise StageError(f"{where}: a 360 view needs fov between 0 and 180")
    return Look(camera=table["camera"], rotate=table.get("rotate", 0),
                yaw=float(table["yaw"]) if is_view else None, pitch=float(table.get("pitch", 0)),
                fov=float(table.get("fov", 0)), move=bool(table.get("move", is_view)))


def to_toml(looks: dict[str, Look], patterns: dict[str, list[str]], header: list[str]) -> str:
    lines = [f"# {line}" for line in header]
    for name, look in looks.items():
        if look == MASTER:
            continue
        lines += ["", f"[look.{name}]", f'camera = "{look.camera}"']
        if look.yaw is None:
            lines.append(f"rotate = {look.rotate}")
        else:
            lines += [f"yaw = {look.yaw:.1f}", f"pitch = {look.pitch:.1f}", f"fov = {look.fov:.1f}"]
    lines += ["", "[pattern]"]
    lines += [f"{name} = {json.dumps(names)}" for name, names in patterns.items()]
    return "\n".join(lines) + "\n"
