import os
import re
from dataclasses import dataclass
from pathlib import Path

from reels.media import cache
from reels.service import StageError

VERSE_GAP_S = 4.0
TIMING = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


@dataclass(frozen=True)
class Caption:
    start: float
    end: float
    text: str
    probability: float = 1.0


def clock(seconds: float) -> str:
    return f"{int(seconds) // 60:02}:{int(seconds) % 60:02}"


def srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours:02}:{minutes:02}:{rest // 1000:02},{rest % 1000:03}"


def format_srt(captions: list[Caption]) -> str:
    return "\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text}\n" for i, c in enumerate(captions, 1))


def format_txt(captions: list[Caption]) -> str:
    lines = []
    for previous, caption in zip([None, *captions], captions):
        if previous is not None and caption.start - previous.end > VERSE_GAP_S:
            lines.append("")
        lines.append(caption.text)
    return "\n".join(lines) + "\n"


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Caption]:
    normal = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    captions = []
    for number, block in enumerate(re.split(r"\n\s*\n", normal.strip()), 1):
        lines = [line.strip() for line in block.split("\n")]
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        match = TIMING.search(lines[timing]) if timing is not None else None
        if match is None:
            raise StageError(f"cannot read caption block {number} of the .srt: {block[:40]!r}")
        if match is not None and any("-->" in line for line in lines[timing + 1:]):
            raise StageError(f"cannot read caption block {number} of the .srt: two captions without a blank line")
        caption_text = " ".join(line for line in lines[timing + 1:] if line)
        if caption_text:
            captions.append(Caption(_seconds(*match.groups()[:4]), _seconds(*match.groups()[4:]), caption_text))
    if not captions:
        raise StageError("no captions found in the .srt")
    return captions


def check_free(paths: list[Path], force: bool) -> None:
    existing = [p.name for p in paths if p.exists()]
    if existing and not force:
        raise StageError(f"{', '.join(existing)} already exists, use --force to overwrite")


def write_all(files: dict[Path, str]) -> None:
    parts = {path: cache.temp_name(path) for path in files}
    replaced = []
    try:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            parts[path].write_text(text, encoding="utf-8")
        for path, part in parts.items():
            os.replace(part, path)
            replaced.append(path)
    except BaseException:
        for part in parts.values():
            part.unlink(missing_ok=True)
        for path in replaced:
            path.unlink(missing_ok=True)
        raise
