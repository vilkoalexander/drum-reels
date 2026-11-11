import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ENCODERS = ("h264_videotoolbox", "prores_ks", "libx264", "aac")
FILTERS = ("v360", "sendcmd", "loudnorm", "drawtext", "subtitles")
RESOLVE = Path("/Applications/DaVinci Resolve/DaVinci Resolve.app")
REQUIRED = ("ffmpeg", "ffprobe", "python")


@dataclass(frozen=True)
class Options:
    folder: str | None = None


def _ffmpeg_list(kind: str) -> str:
    if shutil.which("ffmpeg") is None:
        return ""
    return subprocess.run(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True, check=False).stdout


def has_videotoolbox() -> bool:
    return " h264_videotoolbox " in _ffmpeg_list("encoders")


def _version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    return out.splitlines()[0] if out else "present"


def checks(folder: Path) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    rows.append(("python", sys.version_info >= (3, 11), sys.version.split()[0]))
    ffmpeg = _version(["ffmpeg", "-version"])
    rows.append(("ffmpeg", ffmpeg is not None, ffmpeg or "missing: brew install ffmpeg"))
    rows.append(("ffprobe", _version(["ffprobe", "-version"]) is not None, ""))
    encoders, filters = _ffmpeg_list("encoders"), _ffmpeg_list("filters")
    for name in ENCODERS:
        rows.append((f"encoder {name}", f" {name} " in encoders,
                     "" if f" {name} " in encoders else "cut.codec = \"prores\" works without it" if name == "h264_videotoolbox" else "missing"))
    for name in FILTERS:
        present = f" {name} " in filters
        hint = "" if present else "needed by stage 5 text layers (homebrew-ffmpeg tap or Pillow)" if name in ("drawtext", "subtitles") else "missing"
        rows.append((f"filter {name}", present, hint))
    try:
        from yt_dlp.version import __version__ as ytdlp
    except ImportError:
        ytdlp = None
    rows.append(("yt-dlp", ytdlp is not None, ytdlp or "pip install yt-dlp"))
    for name, module in (("mlx-whisper", "mlx_whisper"), ("demucs", "demucs")):
        found = importlib.util.find_spec(module) is not None
        rows.append((name, found, "" if found else "needed by reels subs: pip install -e '.[subs]'"))
    found = importlib.util.find_spec("anthropic") is not None
    rows.append(("anthropic", found, "" if found else "needed by reels translate: pip install -e '.[translate]'"))
    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    rows.append(("ANTHROPIC_API_KEY", key, "" if key else "needed by reels translate: export your Claude API key"))
    rows.append(("DaVinci Resolve", RESOLVE.exists(), "free edition imports out/timeline.fcpxml" if RESOLVE.exists() else "not found"))
    usage = shutil.disk_usage(folder)
    rows.append(("disk free", usage.free > 50 * 2**30, f"{usage.free / 2**30:.0f} GB at {folder}"))
    return rows


def run(opts: Options, cfg) -> dict:
    missing = {}
    for name, ok, note in checks(Path(opts.folder or ".")):
        log.info("%s %s%s", "ok     " if ok else "missing", name, f"  {note}" if note else "")
        if not ok:
            missing[name] = note
    return {"ok": not any(name in missing for name in REQUIRED), "missing": missing}
