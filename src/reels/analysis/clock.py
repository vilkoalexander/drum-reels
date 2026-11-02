import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.signal import correlate

from reels.service import StageError

SILENCE_STD = 1e-4
EXCLUDE_S = 0.5
MIN_SUPPORT = 3
SLOPE_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class ClockMap:
    intercept_s: float
    slope: float

    def cam_time(self, yt_s: float) -> float:
        return self.intercept_s + self.slope * yt_s

    def yt_time(self, cam_s: float) -> float:
        return (cam_s - self.intercept_s) / self.slope


@dataclass(frozen=True)
class ClockFit:
    intercept_s: float
    slope: float
    residual_ms: float
    used: list[bool]


def clock_map(entry: dict, name: str) -> ClockMap:
    try:
        intercept, slope = float(entry["intercept_s"]), float(entry["slope"])
    except (KeyError, TypeError, ValueError) as error:
        raise StageError(f"{name}: sync entry needs numeric intercept_s and slope ({error!r})") from error
    if not math.isfinite(intercept) or not SLOPE_RANGE[0] < slope < SLOPE_RANGE[1]:
        raise StageError(f"{name}: sync entry intercept {intercept} / slope {slope} is not a clock map near rate 1")
    return ClockMap(intercept, slope)


def envelope(y: np.ndarray, hop: int) -> np.ndarray:
    n = len(y) // hop
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    env = np.log1p(50 * rms)
    env = env - env.mean()
    return env / (env.std() + 1e-9)


def _peak(corr: np.ndarray, lags: np.ndarray, exclude: int) -> tuple[float, float]:
    i = int(np.argmax(corr))
    rest = corr[np.abs(lags - lags[i]) >= exclude]
    confidence = float((corr[i] - rest.mean()) / (rest.std() + 1e-9)) if len(rest) > 1 else 0.0
    frac = 0.0
    if 0 < i < len(corr) - 1:
        denom = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if denom != 0:
            frac = 0.5 * (corr[i - 1] - corr[i + 1]) / denom
    return float(lags[i] + frac), confidence


def coarse_offset(cam_env: np.ndarray, yt_env: np.ndarray, hop: int, sr: int) -> tuple[float, float]:
    corr = correlate(cam_env, yt_env, mode="full", method="fft")
    lags = np.arange(-len(yt_env) + 1, len(cam_env))
    lag, confidence = _peak(corr, lags, max(1, round(EXCLUDE_S * sr / hop)))
    return lag * hop / sr, confidence


def fine_points(
    cam: np.ndarray, yt: np.ndarray, sr: int, coarse_s: float, window_s: float, every_s: float, search_s: float
) -> list[tuple[float, float, float]]:
    window = int(window_s * sr)
    search = int(search_s * sr)
    coarse = round(coarse_s * sr)
    first = max(0, coarse + search)
    last = min(len(cam) - window, len(yt) + coarse - window - search)
    if last < first:
        return []
    span = last - first
    every = max(1, int(every_s * sr))
    if span // every + 1 < MIN_SUPPORT:
        every = max(1, span // (MIN_SUPPORT - 1))
    points = []
    for start in range(first, last + 1, every):
        seg = cam[start : start + window]
        if seg.std() < SILENCE_STD:
            continue
        lo = start - coarse - search
        hi = start - coarse + window + search
        corr = correlate(yt[lo:hi], seg, mode="valid", method="fft")
        lags = np.arange(len(corr))
        pos, confidence = _peak(corr, lags, max(1, round(EXCLUDE_S * sr)))
        points.append(((lo + pos) / sr, start / sr, confidence))
    return points


def fit_clock(points: list[tuple[float, float, float]], max_residual_ms: float, max_drift_ppm: float) -> ClockFit:
    n = len(points)
    if n < 2:
        shift = float(np.median([c - y for y, c, _ in points])) if points else 0.0
        return ClockFit(shift, 1.0, 0.0, [False] * n)
    yt, cam, conf = (np.array(col, dtype=float) for col in zip(*points, strict=True))
    weights = np.clip(conf, 1, None)
    limit = max_residual_ms / 1000
    bound = max_drift_ppm * 1e-6
    fallback = ClockFit(float(np.median(cam - yt)), 1.0, 0.0, [False] * n)
    support = min(n, MIN_SUPPORT)

    best_key, keep = None, None
    for i, j in combinations(range(n), 2):
        if yt[j] == yt[i]:
            continue
        slope = (cam[j] - cam[i]) / (yt[j] - yt[i])
        if abs(slope - 1) > bound:
            continue
        intercept = cam[i] - slope * yt[i]
        inliers = np.abs(cam - (intercept + slope * yt)) <= limit
        key = (int(inliers.sum()), -abs(slope - 1))
        if best_key is None or key > best_key:
            best_key, keep = key, inliers
    if keep is None:
        return fallback
    while keep.sum() >= support:
        slope, intercept = np.polyfit(yt[keep], cam[keep], 1, w=weights[keep])
        residual = cam - (intercept + slope * yt)
        refit = np.abs(residual) <= limit
        if abs(slope - 1) > bound:
            return fallback
        if np.array_equal(refit, keep):
            rms = float(np.sqrt(np.mean(residual[keep] ** 2)) * 1000)
            return ClockFit(float(intercept), float(slope), rms, [bool(k) for k in keep])
        keep = refit & keep
    return fallback
