import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

log = logging.getLogger(__name__)

TRUTH = {
    "dji": {"intercept_s": 37.3, "slope": 1 + 80e-6, "duration_s": 230.0, "fps": "60000/1001", "size": "1920x1080",
            "snr_db": 8, "source": "testsrc"},
    "x5": {"intercept_s": -12.1, "slope": 1 - 50e-6, "duration_s": 200.0, "fps": "30", "size": "1920x960",
           "snr_db": 5, "source": "testsrc2"},
}
NOTES = "0:00:40 fill\n0:02:30 groove 20\n0:03:05 guitar solo 15\n"
SHORT_TRUTH = {
    "dji": {"intercept_s": 8.0, "slope": 1 + 100e-6, "duration_s": 60.0, "fps": "25", "size": "320x180",
            "snr_db": 8, "source": "testsrc"},
    "x5": {"intercept_s": -5.0, "slope": 1 - 60e-6, "duration_s": 50.0, "fps": "30", "size": "320x160",
           "snr_db": 6, "source": "testsrc2"},
}
SHORT_NOTES = "0:00:20 fill 10\n0:00:45 guitar solo 8\n"


def bursts(duration_s: float, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    out = rng.normal(0, 0.01, n).astype(np.float32)
    burst = int(0.05 * sr)
    decay = np.exp(-np.arange(burst) / (0.01 * sr))
    for k in range(int(duration_s / 0.5) - 1):
        if rng.random() < 0.6:
            start = int(k * 0.5 * sr)
            out[start : start + burst] += (rng.normal(0, 1, burst) * decay * rng.uniform(0.3, 1.0)).astype(np.float32)
    mid = slice(n // 3, 2 * n // 3)
    out[mid] *= 0.8
    return out


def drums(duration_s: float, sr: int, seed: int, bpm: float = 118.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    beat = 60 / bpm

    def hit(at, length, f=None, gain=1.0):
        i0 = int(at * sr)
        if i0 >= n:
            return
        size = min(int(length * sr), n - i0)
        tt = np.arange(size) / sr
        env = np.exp(-tt / (length / 4))
        s = rng.standard_normal(size) if f is None else np.sin(2 * np.pi * (f * tt - 30 * tt * tt))
        y[i0:i0 + size] += gain * env * s

    b = 0
    while b * beat < duration_s:
        at = b * beat
        bar, pos = divmod(b, 4)
        if pos in (0, 2):
            hit(at, 0.25, f=70, gain=1.2)
        if pos in (1, 3):
            hit(at, 0.18, gain=0.9)
        for k in range(2):
            hit(at + k * beat / 2, 0.05, gain=0.25)
        if bar % 8 == 7 and pos == 3:
            for k in range(8):
                hit(at + k * beat / 8, 0.08, gain=0.8 + 0.1 * k)
        b += 1
    for f in (220, 277, 330):
        y += 0.05 * np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 23))
    quiet = slice(int(duration_s * 5 / 12 * sr), int(duration_s * 13 / 24 * sr))
    y[quiet] *= 0.05
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def camera(master: np.ndarray, sr: int, intercept_s: float, slope: float, start_s: float, duration_s: float,
           snr_db: float, seed: int, lowpass: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cam_t = start_s + np.arange(int(duration_s * sr)) / sr
    yt_t = (cam_t - intercept_s) / slope
    out = np.interp(yt_t * sr, np.arange(len(master)), master, left=0.0, right=0.0)
    if lowpass:
        bb, aa = butter(2, 4000 / (sr / 2))
        out = lfilter(bb, aa, out)
    power = float(np.mean(out**2)) or 1e-12
    noise = rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(out))
    return np.clip(out + noise, -1, 1).astype(np.float32)


def mux(wav: Path, out: Path, source: str, size: str, fps: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"{source}=size={size}:rate={fps}",
         "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
         "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )


def write_service(root: Path, duration_s: float = 240.0, sr: int = 48000, style: str = "drums",
                  truth: dict | None = None, notes: str | None = None) -> dict:
    root = Path(root)
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(exist_ok=True)
    truth = truth or (TRUTH if style == "drums" else SHORT_TRUTH)
    notes = notes or (NOTES if style == "drums" else SHORT_NOTES)
    master = drums(duration_s, sr, seed=1) if style == "drums" else bursts(duration_s, sr, seed=1)
    jobs = [("youtube", master, "testsrc", "1280x720" if style == "drums" else "320x180", "30")]
    for seed, (name, spec) in enumerate(truth.items(), 2):
        cam = camera(master, sr, spec["intercept_s"], spec["slope"], 0.0, spec["duration_s"], spec["snr_db"], seed,
                     lowpass=style == "drums")
        jobs.append((name, cam, spec["source"], spec["size"], spec["fps"]))
    for name, samples, source, size, fps in jobs:
        wav = sources / f"{name}.wav"
        sf.write(wav, samples, sr)
        mux(wav, sources / f"{name}.mp4", source, size, fps)
        wav.unlink()
    (root / "notes" / "moments.txt").write_text(notes)
    compact = {name: {"intercept_s": spec["intercept_s"], "slope": spec["slope"]} for name, spec in truth.items()}
    (root / "truth.json").write_text(json.dumps(compact, indent=2))
    return compact


@dataclass(frozen=True)
class Options:
    folder: str
    duration: float = 240.0
    style: str = "drums"


def run(opts: Options, cfg) -> dict:
    truth = write_service(Path(opts.folder), duration_s=opts.duration, style=opts.style)
    log.info("wrote %s with truth %s", opts.folder, truth)
    return {"ok": True, "folder": opts.folder, "truth": truth}
