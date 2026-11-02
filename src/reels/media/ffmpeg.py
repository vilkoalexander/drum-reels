import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reels.media import cache
from reels.service import CODECS, StageError

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
PRORES_PROFILE = "2"
VFR_TOLERANCE = 0.01


@dataclass
class MediaInfo:
    path: Path
    fps: float
    width: int
    height: int
    duration_s: float
    has_audio: bool
    audio_offset_s: float = 0.0


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("run %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise StageError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")


def _fraction(text: str) -> float:
    num, _, den = text.partition("/")
    return float(num) / float(den) if den and float(den) else float(num)


def probe(path: Path) -> MediaInfo:
    cmd = ["ffprobe", "-hide_banner", "-loglevel", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    log.debug("run %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise StageError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise StageError(f"no video stream in {path}")
    duration = video.get("duration") or data.get("format", {}).get("duration")
    if duration is None:
        raise StageError(f"ffprobe reports no duration for {path} (still recording or broken file?)")
    fps = _fraction(video.get("r_frame_rate", "0"))
    if fps <= 0:
        raise StageError(f"ffprobe reports frame rate {video.get('r_frame_rate')} for {path}")
    average = _fraction(video.get("avg_frame_rate", "0"))
    if average > 0 and abs(average - fps) / fps > VFR_TOLERANCE:
        log.warning("%s: variable frame rate? r_frame_rate %.3f but avg_frame_rate %.3f", path.name, fps, average)
    offset = 0.0
    if audio is not None:
        offset = float(audio.get("start_time", 0) or 0) - float(video.get("start_time", 0) or 0)
    return MediaInfo(
        path=path,
        fps=fps,
        width=int(video["width"]),
        height=int(video["height"]),
        duration_s=float(duration),
        has_audio=audio is not None,
        audio_offset_s=offset,
    )


def write_atomic(args_before: list[str], args_after: list[str], dst: Path) -> None:
    tmp = cache.temp_name(dst)
    try:
        run_ffmpeg([*args_before, *args_after, str(tmp)])
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def join_chunks(chunks: list[Path], dst: Path, force: bool = False) -> Path:
    if len(chunks) == 1:
        return chunks[0]
    expected = cache.join_stamp(chunks)
    if not force and cache.fresh(dst, expected):
        log.info("reusing %s", dst.name)
        return dst
    listing = dst.with_suffix(".txt")
    lines = []
    for chunk in chunks:
        resolved = str(chunk.resolve())
        if "\n" in resolved or "\r" in resolved:
            raise StageError(f"cannot join a file whose name holds a line break: {chunk.name!r}")
        escaped = resolved.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s", dst.name)
    listing.write_text("\n".join(lines) + "\n")
    try:
        write_atomic(["-f", "concat", "-safe", "0", "-i", str(listing)], ["-c", "copy"], dst)
    except BaseException:
        listing.unlink(missing_ok=True)
        raise
    cache.publish(dst, expected)
    return dst


def cut_args(
    video_src: Path,
    audio_src: Path | None,
    start_s: float,
    duration_s: float,
    codec: str,
    bitrate_mbps: float,
    audio_start_s: float | None = None,
) -> list[str]:
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec}")
    if duration_s <= 0:
        raise ValueError(f"cut duration must be positive, got {duration_s:.3f}")
    args = ["-ss", f"{start_s:.3f}", "-i", str(video_src)]
    if audio_src is None:
        args += ["-map", "0:v:0", "-map", "0:a:0?"]
    else:
        audio_start = start_s if audio_start_s is None else audio_start_s
        args += ["-ss", f"{audio_start:.3f}", "-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0"]
    args += ["-t", f"{duration_s:.3f}"]
    if codec == "h264":
        args += ["-c:v", "h264_videotoolbox", "-b:v", f"{bitrate_mbps:g}M", "-allow_sw", "1", "-profile:v", "high"]
    else:
        args += ["-c:v", "prores_ks", "-profile:v", PRORES_PROFILE]
    args += ["-c:a", "pcm_s16le", "-ar", str(AUDIO_RATE), "-movflags", "+faststart"]
    return args


def cut(
    video_src: Path,
    audio_src: Path | None,
    start_s: float,
    duration_s: float,
    dst: Path,
    codec: str,
    bitrate_mbps: float,
    audio_start_s: float | None = None,
) -> Path:
    if dst.suffix != ".mov":
        raise ValueError(f"output must be .mov, got {dst}")
    args = cut_args(video_src, audio_src, start_s, duration_s, codec, bitrate_mbps, audio_start_s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args, [], dst)
    return dst
