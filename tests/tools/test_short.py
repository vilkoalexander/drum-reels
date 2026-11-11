import numpy as np
import pytest

from reels.service import StageError
from reels.tools.short import Options, best_window, extend_back, nearest_downbeat, run


def test_best_window_picks_the_busiest_bars():
    downbeats = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    times = np.arange(0, 10, 0.1)
    strength = np.where((times >= 4) & (times < 8), 5.0, 1.0)
    assert best_window(downbeats, times, strength, 2) == 2


def test_best_window_needs_enough_bars():
    with pytest.raises(StageError, match="need 4"):
        best_window([0.0, 2.0, 4.0], np.arange(0, 4, 0.1), np.ones(40), 4)



def test_nearest_downbeat_snaps_and_leaves_room_for_the_bars():
    downbeats = [0.0, 2.0, 4.0, 6.0, 8.0]
    assert nearest_downbeat(downbeats, 4.3, 2) == 2
    assert nearest_downbeat(downbeats, 7.9, 2) == 2


def test_extend_back_carries_the_grid_to_zero():
    assert extend_back([7.0, 9.0, 11.0], 2.0) == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]


def test_run_rejects_an_unreadable_srt_before_any_work(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    with pytest.raises(StageError, match="missing.srt"):
        run(Options(str(video), subs=str(tmp_path / "missing.srt")), None)
    assert not (tmp_path / "shorts").exists()
