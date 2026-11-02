import pytest

from reels import config
from reels.service import Service, StageError, write_json
from reels.stages import reel
from reels.tools import short

LOOKS = """
[look.top]
camera = "dji"
rotate = 90

[pattern]
default = ["top", "master"]
"""


@pytest.fixture
def remotion_ready(monkeypatch, tmp_path):
    (tmp_path / "remotion" / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(reel, "REMOTION_DIR", tmp_path / "remotion")
    monkeypatch.setattr(reel.shutil, "which", lambda name: "/usr/bin/npx")


@pytest.fixture
def service(tmp_path):
    service = Service(tmp_path / "svc")
    folder = service.cuts / "fill"
    folder.mkdir(parents=True)
    write_json(folder / "moment.json", {
        "slug": "fill", "label": "fill", "start_s": 10.0, "end_s": 20.0, "handle_s": 2.0, "bpm": 120.0,
        "files": {"youtube": "youtube.mov", "dji": "dji.mov"}, "skipped": {}, "inputs": {},
    })
    (folder / "looks.toml").write_text(LOOKS)
    return service


@pytest.mark.parametrize("overrides, message", [
    ({"grade": "sepia"}, "unknown grade"),
    ({"pattern": "polka"}, "no pattern 'polka'.*it has default"),
    ({"look_text": ["nope=hi"]}, "--look-text 'nope=hi'.*master, top"),
])
def test_reel_rejects_bad_options_before_writing_anything(service, remotion_ready, overrides, message):
    with pytest.raises(StageError, match=message):
        reel.run(reel.Options(service=str(service.root), slug="fill", **overrides), config.Config())
    assert not service.work.exists()


def test_reel_without_looks_file_says_how_to_make_one(service, remotion_ready):
    (service.cuts / "fill" / "looks.toml").unlink()
    with pytest.raises(StageError, match="run reels looks .* fill first"):
        reel.run(reel.Options(service=str(service.root), slug="fill"), config.Config())


def test_reel_and_short_need_remotion_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(reel.shutil, "which", lambda name: None)
    with pytest.raises(StageError, match="npx"):
        reel.run(reel.Options(service=str(tmp_path / "svc"), slug="fill"), config.Config())
    assert not (tmp_path / "svc").exists()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(StageError, match="npx"):
        short.run(short.Options(video=str(video)), config.Config())
    assert not (tmp_path / "shorts").exists()


def test_short_rejects_bad_grade_before_analysis(tmp_path, remotion_ready):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(StageError, match="unknown grade"):
        short.run(short.Options(video=str(video), grade="sepia"), config.Config())
    assert not (tmp_path / "shorts").exists()
