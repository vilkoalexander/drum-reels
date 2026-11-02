import json
import logging
import shutil
import xml.etree.ElementTree as ET

import pytest

from reels import cli
from reels.tools.doctor import has_videotoolbox
from reels.tools.synth import write_service

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


CODECS = ["prores", pytest.param("h264", marks=pytest.mark.skipif(not has_videotoolbox(), reason="no videotoolbox"))]


@pytest.fixture(scope="module", params=CODECS)
def service(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"service-{request.param}")
    truth = write_service(root, duration_s=60.0, style="bursts")
    svc = str(root)
    (root / "reels.toml").write_text(f'[cut]\ncodec = "{request.param}"\n')
    flags = ["--config", str(root / "reels.toml")]
    assert cli.main([*flags, "sync", svc]) == 0, "sync"
    assert cli.main([*flags, "cut", svc]) == 0, "cut"
    assert cli.main([*flags, "render", svc, "--mode", "fcpxml"]) == 0, "render"
    return root, truth


def test_sync_matches_truth(service):
    root, truth = service
    sources = json.loads((root / "work" / "sync.json").read_text())["sources"]
    for cam, want in truth.items():
        assert sources[cam]["intercept_s"] == pytest.approx(want["intercept_s"], abs=0.005)
        assert sources[cam]["drift_ppm"] == pytest.approx((want["slope"] - 1) * 1e6, abs=50)


def test_fill_cut_has_every_camera(service):
    root, _ = service
    folder = root / "cuts" / "0-00-20_fill"
    for name in ("youtube.mov", "dji.mov", "x5.mov"):
        assert (folder / name).exists()
    moment = json.loads((folder / "moment.json").read_text())
    assert moment["person"] is None
    assert moment["skipped"] == {}


def test_guitar_solo_skips_dji_whose_recording_ended(service):
    root, _ = service
    moment = json.loads((root / "cuts" / "0-00-45_guitar-solo" / "moment.json").read_text())
    assert moment["person"] == "guitar"
    assert "recording ends" in moment["skipped"]["dji"]


def test_timeline_has_three_projects(service):
    root, _ = service
    tree = ET.parse(root / "out" / "timeline.fcpxml")
    assert len(list(tree.iter("project"))) == 3


def test_second_sync_is_a_no_op(service, caplog):
    root, _ = service
    sidecar = root / "work" / "sync.json"
    before = sidecar.stat().st_mtime_ns
    with caplog.at_level(logging.INFO, logger="reels.stages.sync"):
        assert cli.main(["--config", str(root / "reels.toml"), "sync", str(root)]) == 0
    assert "up to date" in caplog.text
    assert sidecar.stat().st_mtime_ns == before
