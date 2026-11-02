import logging
from dataclasses import dataclass

import librosa
import numpy as np

log = logging.getLogger(__name__)

REGION_PAD_S = 60.0
HOP = 128
PEAK_WINDOW = 3
MIN_BARS = 2


@dataclass(frozen=True)
class Grid:
    bpm: float
    beats_s: list[float]
    downbeats_s: list[float]


def region(noted_s: float, len_s: float, duration_s: float) -> tuple[float, float]:
    return max(0.0, noted_s - REGION_PAD_S), min(duration_s, noted_s + len_s + REGION_PAD_S)


def grid(y: np.ndarray, sr: int, region_start_s: float, beats_per_bar: int) -> Grid:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", hop_length=HOP)
    if len(beats) < MIN_BARS * beats_per_bar:
        log.warning("beats: only %d beats found, fewer than %d bars, no grid", len(beats), MIN_BARS)
        return Grid(bpm=0.0, beats_s=[], downbeats_s=[])
    bpm = float(np.atleast_1d(tempo)[0])
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)
    beat_strength = np.array([
        strength[max(0, f - PEAK_WINDOW):f + PEAK_WINDOW + 1].sum() for f in frames
    ])
    scores = [float(beat_strength[p::beats_per_bar].mean()) for p in range(beats_per_bar)]
    phase = int(np.argmax(scores))
    log.info("beats: bpm %.1f, %d beats, phase %d score %.2f (all %s)",
             bpm, len(beats), phase, scores[phase], [round(s, 2) for s in scores])
    beats_abs = [float(b) + region_start_s for b in beats]
    return Grid(bpm=bpm, beats_s=beats_abs, downbeats_s=beats_abs[phase::beats_per_bar])


def snap(grid: Grid, noted_s: float, len_s: float, snap_to_bars: bool) -> tuple[float, float]:
    raw = (noted_s, noted_s + len_s)
    if not snap_to_bars or len(grid.downbeats_s) < 2:
        return raw
    downbeats = np.asarray(grid.downbeats_s)
    bar_s = float(np.median(np.diff(downbeats)))
    before = downbeats[downbeats <= noted_s]
    start = noted_s
    if len(before) and noted_s - before[-1] < bar_s:
        start = float(before[-1])
    target = start + len_s
    end = target
    after = downbeats[downbeats > start]
    if len(after):
        nearest = float(after[np.argmin(np.abs(after - target))])
        if abs(nearest - target) < bar_s:
            end = nearest
    log.info("beats: snap %.2f-%.2f -> %.2f-%.2f (bar %.2f s)", *raw, start, end, bar_s)
    return start, end
