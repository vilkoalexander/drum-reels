import json
import shutil

import numpy as np
import pytest

from reels import cli
from reels.tools import synth

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def test_bursts_and_drums_are_deterministic_and_bounded():
    a, b = synth.bursts(5, 16000, seed=3), synth.bursts(5, 16000, seed=3)
    assert np.array_equal(a, b) and len(a) == 5 * 16000
    d = synth.drums(6, 16000, seed=1)
    assert len(d) == 6 * 16000 and abs(d).max() <= 0.8 + 1e-6


def test_camera_applies_offset_and_drift():
    master = synth.bursts(20, 16000, seed=1)
    cam = synth.camera(master, 16000, intercept_s=2.0, slope=1.0, start_s=0.0, duration_s=10, snr_db=60, seed=2)
    assert len(cam) == 10 * 16000
    assert abs(cam[: 2 * 16000]).max() < 0.1
    assert np.corrcoef(cam[2 * 16000 + 100 : 4 * 16000], master[100 : 2 * 16000])[0, 1] > 0.9


def test_synth_command_writes_a_service_with_truth(tmp_path):
    folder = tmp_path / "svc"
    assert cli.main(["synth", str(folder), "--duration", "8", "--style", "bursts"]) == 0
    assert {p.name for p in (folder / "sources").iterdir()} == {"youtube.mp4", "dji.mp4", "x5.mp4"}
    truth = json.loads((folder / "truth.json").read_text())
    assert truth["dji"]["intercept_s"] == 8.0 and (folder / "notes" / "moments.txt").exists()
