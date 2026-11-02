import re
from collections.abc import Sequence
from dataclasses import dataclass, field

LINE = re.compile(r"(?:(\d{1,2}):)?([0-5]?\d):([0-5]\d)(?:\s+(.*))?")
TAIL = re.compile(r"(.*?)\s+(\d+)")
ALIASES = {"vocal": "vocals", "vocals": "vocals", "singer": "vocals", "piano": "keys"}


class NotesError(ValueError):
    pass


@dataclass(frozen=True)
class Moment:
    noted_s: float
    label: str
    len_s: float | None
    person: str | None
    line: int = field(default=0, compare=False)


def parse(text: str, people: Sequence[str], min_len_s: float = 0.0) -> list[Moment]:
    known = {name.lower() for name in people}
    moments = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        moment = _parse_line(line, number, known)
        if moment.len_s is not None and moment.len_s <= 0:
            raise NotesError(f"line {number}: length must be positive in {line!r}")
        if moment.len_s is not None and moment.len_s < min_len_s:
            raise NotesError(f"line {number}: length {moment.len_s:g} s is below the minimum {min_len_s:g} s in "
                             f"{line!r} (a label ending in a number is read as a length)")
        moments.append(moment)
    return moments


def slug(moment: Moment) -> str:
    hours, rest = divmod(int(moment.noted_s), 3600)
    minutes, seconds = divmod(rest, 60)
    time = f"{hours}-{minutes:02d}-{seconds:02d}"
    label = _label_slug(moment.label)
    return f"{time}_{label}" if label else time


def _parse_line(line: str, number: int, known: set[str]) -> Moment:
    match = LINE.fullmatch(line)
    if match is None:
        raise NotesError(f"line {number}: cannot parse {line!r}")
    hours, minutes, seconds, rest = match.groups()
    noted_s = float(int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds))
    label, len_s = _label_and_length(rest or "")
    if not label.strip():
        raise NotesError(f"line {number}: no label in {line!r}")
    return Moment(noted_s, label, len_s, _person(label, known), number)


def _label_and_length(rest: str) -> tuple[str, float | None]:
    tail = TAIL.fullmatch(rest)
    if tail:
        return tail.group(1).strip(), float(tail.group(2))
    return rest.strip(), None


def _person(label: str, known: set[str]) -> str | None:
    for word in re.findall(r"\w+", label):
        if word.lower() in known:
            return ALIASES.get(word.lower(), word.lower())
    return None


def _label_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
