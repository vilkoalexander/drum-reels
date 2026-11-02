import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from reels.media import cache
from reels.media.ffmpeg import write_atomic
from reels.service import StageError

log = logging.getLogger(__name__)


def extract_audio(src: Path, dst: Path, sample_rate: int, force: bool = False) -> Path:
    expected = cache.audio_stamp(src, sample_rate)
    if not force and cache.fresh(dst, expected):
        log.info("reusing %s", dst.name)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("writing %s", dst.name)
    write_atomic(["-i", str(src)], ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le"], dst)
    cache.publish(dst, expected)
    return dst


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except (RuntimeError, sf.LibsndfileError) as error:
        raise StageError(f"cannot read {path}: {error}") from error
    return data, sr


def read_wav_region(path: Path, start_s: float, stop_s: float) -> tuple[np.ndarray, int]:
    try:
        sr = sf.info(str(path)).samplerate
        data, _ = sf.read(str(path), start=int(start_s * sr), stop=int(stop_s * sr), dtype="float32")
    except (RuntimeError, sf.LibsndfileError) as error:
        raise StageError(f"cannot read {path}: {error}") from error
    return data, sr
