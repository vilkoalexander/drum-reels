import numpy as np
import pytest

from reels.analysis import beats
from reels.analysis.beats import Grid

SR = 16000


def click_track(seconds=40, bpm=120, per_bar=4):
    rng = np.random.default_rng(0)
    y = np.zeros(seconds * SR, dtype=np.float32)
    burst = int(0.005 * SR)
    for i in range(int(seconds * bpm / 60)):
        amp = 0.9 if i % per_bar == 0 else 0.3
        s = int(i * 60 / bpm * SR)
        y[s:s + burst] = amp * rng.uniform(-1, 1, burst)
    return y


def bar_grid():
    return Grid(bpm=120.0, beats_s=[], downbeats_s=[float(t) for t in range(0, 41, 2)])


def test_grid_finds_tempo_and_accented_downbeats():
    g = beats.grid(click_track(), SR, 0.0, 4)
    assert g.bpm == pytest.approx(120, abs=2)
    assert len(g.downbeats_s) >= 15
    assert all(abs(d - 2 * round(d / 2)) < 0.03 for d in g.downbeats_s)


def test_grid_offsets_times_by_region_start():
    g = beats.grid(click_track(), SR, 100.0, 4)
    assert min(g.downbeats_s) >= 100.0
    assert all(abs((d - 100.0) - 2 * round((d - 100.0) / 2)) < 0.03 for d in g.downbeats_s)


def test_grid_on_silence_is_empty():
    g = beats.grid(np.zeros(5 * SR, dtype=np.float32), SR, 0.0, 4)
    assert g == Grid(bpm=0.0, beats_s=[], downbeats_s=[])


def test_snap_to_bar_start_and_end():
    assert beats.snap(bar_grid(), 10.3, 8, True) == (10.0, 18.0)


def test_snap_keeps_noted_time_when_over_a_bar_from_last_downbeat():
    g = Grid(bpm=120.0, beats_s=[], downbeats_s=[0.0, 2.0, 4.0, 6.0])
    start, end = beats.snap(g, 20.0, 8, True)
    assert start == 20.0
    assert end == 28.0


def test_snap_without_bars_returns_raw_window():
    assert beats.snap(bar_grid(), 10.3, 8, False) == (10.3, 18.3)


def test_snap_with_empty_grid_returns_raw_window():
    assert beats.snap(Grid(0.0, [], []), 10.3, 8, True) == (10.3, 18.3)


def test_region_clamps_to_recording():
    assert beats.region(30, 20, 500) == (0.0, 110.0)
    assert beats.region(300, 20, 350) == (240.0, 350.0)


def test_grid_picks_phase_when_first_beat_is_not_accented(monkeypatch):
    clicks = np.arange(0.5, 38.0, 0.5)
    monkeypatch.setattr(beats.librosa.beat, "beat_track", lambda **kw: (120.0, clicks))
    g = beats.grid(click_track(), SR, 0.0, 4)
    assert g.downbeats_s[0] == 2.0
    assert all(d % 2 == 0 for d in g.downbeats_s)


def test_snap_never_collapses_a_short_moment_to_zero_length():
    g = Grid(bpm=120.0, beats_s=[], downbeats_s=[0.0, 2.0, 4.0, 6.0, 8.0])
    assert beats.snap(g, 4.0, 1, True) == (4.0, 6.0)
    assert beats.snap(g, 0.5, 2, True) == (0.0, 2.0)


def test_grid_with_too_few_beats_is_empty(monkeypatch):
    monkeypatch.setattr(beats.librosa.beat, "beat_track", lambda **kw: (120.0, np.array([0.5, 1.0, 1.5])))
    assert beats.grid(click_track(), SR, 0.0, 4) == Grid(bpm=0.0, beats_s=[], downbeats_s=[])


def test_grid_needs_min_bars_of_beats(monkeypatch, caplog):
    caplog.set_level("WARNING")
    too_few = np.arange(0.5, 0.5 * (beats.MIN_BARS * 4), 0.5)
    monkeypatch.setattr(beats.librosa.beat, "beat_track", lambda **kw: (120.0, too_few))
    assert beats.grid(click_track(), SR, 0.0, 4) == Grid(bpm=0.0, beats_s=[], downbeats_s=[])
    assert "fewer than 2 bars" in caplog.text
    enough = np.arange(0.5, 0.5 * (beats.MIN_BARS * 4 + 1), 0.5)
    monkeypatch.setattr(beats.librosa.beat, "beat_track", lambda **kw: (120.0, enough))
    assert beats.grid(click_track(), SR, 0.0, 4).bpm == 120.0
