import numpy as np
import pytest

from reels.analysis.clock import ClockMap, coarse_offset, envelope, fine_points, fit_clock
from reels.tools import synth

SR = 16000
HOP = 160
INTERCEPT = -20.0
SLOPE = 1 + 100e-6


@pytest.fixture(scope="module")
def audio():
    yt = synth.bursts(90, SR, seed=1)
    cam = synth.camera(yt, SR, INTERCEPT, SLOPE, start_s=0.0, duration_s=60, snr_db=6, seed=2)
    return yt, cam


def test_coarse_offset_finds_truth_with_confidence(audio):
    yt, cam = audio
    offset, confidence = coarse_offset(envelope(cam, HOP), envelope(yt, HOP), HOP, SR)
    assert offset == pytest.approx(INTERCEPT, abs=0.02)
    assert confidence > 4


def test_fine_points_and_fit_recover_clock(audio):
    yt, cam = audio
    points = fine_points(cam, yt, SR, INTERCEPT, 10.0, 15.0, 1.0)
    assert len(points) >= 3
    fit = fit_clock(points, 50, 300)
    assert fit.intercept_s == pytest.approx(INTERCEPT, abs=0.002)
    assert fit.slope == pytest.approx(SLOPE, abs=20e-6)
    assert all(fit.used)


def test_short_recording_still_gets_three_windows(audio):
    yt, _ = audio
    cam = synth.camera(yt, SR, INTERCEPT, SLOPE, start_s=0.0, duration_s=40, snr_db=6, seed=3)
    assert len(fine_points(cam, yt, SR, INTERCEPT, 10.0, 600.0, 1.0)) == 3


def test_short_overlap_has_lower_confidence(audio):
    yt, cam = audio
    _, full = coarse_offset(envelope(cam, HOP), envelope(yt, HOP), HOP, SR)
    brief = synth.camera(yt, SR, -87.0, 1.0, start_s=0.0, duration_s=60, snr_db=6, seed=4)
    _, low = coarse_offset(envelope(brief, HOP), envelope(yt, HOP), HOP, SR)
    assert low < full


def test_fit_clock_drops_planted_outlier():
    yt = np.arange(0, 600, 60.0)
    points = [(float(t), 5.0 + 1.00005 * float(t), 10.0) for t in yt]
    points[4] = (points[4][0], points[4][1] + 0.4, 10.0)
    fit = fit_clock(points, 50, 300)
    assert fit.used == [i != 4 for i in range(len(points))]
    assert fit.intercept_s == pytest.approx(5.0, abs=1e-6)
    assert fit.slope == pytest.approx(1.00005, abs=1e-9)


def test_fit_clock_three_windows_one_bad_falls_back_instead_of_trusting_a_pair():
    points = [(0.0, 37.3, 10.0), (600.0, 637.3, 10.0), (1200.0, 1236.3, 10.0)]
    fit = fit_clock(points, 50, 300)
    assert fit.slope == 1.0
    assert fit.intercept_s == pytest.approx(37.3)
    assert fit.used == [False] * 3


def test_fit_clock_four_windows_one_false_peak_keeps_the_three_that_agree():
    points = [(595.0, 600.0, 10.0), (1195.0, 1200.0, 10.0), (1795.0, 1800.0, 10.0), (2393.5, 2400.0, 10.0)]
    fit = fit_clock(points, 50, 300)
    assert fit.used == [True, True, True, False]
    assert fit.intercept_s == pytest.approx(5.0, abs=1e-6)
    assert fit.slope == pytest.approx(1.0, abs=1e-9)


def test_fit_clock_rejects_implausible_drift():
    points = [(float(t), 1.001 * float(t), 10.0) for t in range(0, 3000, 600)]
    fit = fit_clock(points, 50, max_drift_ppm=300)
    assert fit.slope == 1.0 and fit.used == [False] * 5
    assert fit_clock(points, 50, max_drift_ppm=2000).slope == pytest.approx(1.001, abs=1e-9)


def test_fine_points_schedules_three_windows_inside_the_valid_overlap():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(120 * SR).astype(np.float32)
    points = fine_points(y, y, SR, 0.0, 20.0, 600.0, 2.0)
    assert [round(p[1]) for p in points] == [2, 50, 98]
    assert all(abs(yt - cam) < 0.001 for yt, cam, _ in points)


def test_fit_clock_rechecks_inliers_after_the_weighted_refit():
    points = [(0.0, 0.0, 10.0), (600.0, 600.049, 10.0), (1200.0, 1199.951, 10.0), (1800.0, 1800.049, 10.0),
              (2400.0, 2400.0, 10.0)]
    fit = fit_clock(points, 50, 300)
    for (yt, cam, _), used in zip(points, fit.used, strict=True):
        residual = abs(cam - (fit.intercept_s + fit.slope * yt))
        assert (residual <= 0.05) == used


@pytest.mark.parametrize("coarse", [30.0, -30.0])
def test_fine_points_respects_both_recordings_for_any_coarse_offset(coarse):
    rng = np.random.default_rng(1)
    master = rng.standard_normal(200 * SR).astype(np.float32)
    shift = int(coarse * SR)
    cam = np.roll(master, shift)[: 150 * SR]
    points = fine_points(cam, master, SR, coarse, 10.0, 600.0, 2.0)
    assert len(points) == 3
    for yt_s, cam_s, _ in points:
        assert 0 <= yt_s <= 200 and 0 <= cam_s <= 150
        assert cam_s - yt_s == pytest.approx(coarse, abs=0.001)
    assert fine_points(cam[: 8 * SR], master, SR, coarse, 10.0, 600.0, 2.0) == []


def test_fit_clock_with_one_point_assumes_unit_slope():
    fit = fit_clock([(10.0, 47.3, 9.0)], 50, 300)
    assert fit.slope == 1.0
    assert fit.intercept_s == pytest.approx(37.3)


def test_clock_map_round_trips():
    clock = ClockMap(37.3, 1.0000761)
    assert clock.yt_time(clock.cam_time(1234.5)) == pytest.approx(1234.5, abs=1e-9)
    assert clock.cam_time(0) == 37.3

def test_fit_clock_with_zero_tolerance_falls_back_to_median_shift():
    points = [(0.0, 5.0, 10.0), (10.0, 15.1, 10.0), (20.0, 25.0, 10.0)]
    fit = fit_clock(points, 0, 300)
    assert fit.slope == 1.0
    assert fit.used == [False] * 3
