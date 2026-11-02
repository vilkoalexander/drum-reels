from pathlib import Path

import pytest

from reels import config
from reels.media import MediaInfo
from reels.service import Service, StageError, write_json
from reels.stages import looks
from reels.timeline import looks as looks_file

W, H = 3840, 1920


def px(yaw):
    return (yaw + 180) / 360 * W


class FakeDetector:
    def __init__(self):
        self.previews = []

    def equirect(self, frame: Path):
        boxes = [(px(-83) - 30, 800, px(-83) + 30, 1000), (px(92) - 20, 850, px(92) + 20, 980)]
        if frame.name == "frame_0.jpg":
            boxes.append((px(10) - 20, 850, px(10) + 20, 980))
        return boxes, W, H

    def upright_score(self, frame: Path, rotate: int) -> float:
        return 0.9 if rotate == 90 else 0.1

    def preview(self, frame, labels, boxes, dst):
        self.previews.append((dst, labels))
        dst.write_bytes(b"jpg")


@pytest.fixture
def service(tmp_path, monkeypatch):
    service = Service(tmp_path)
    folder = service.cuts / "fill"
    folder.mkdir(parents=True)
    for name in ("youtube.mov", "dji.mov", "x5.mov"):
        (folder / name).write_bytes(b"x")
    write_json(folder / "moment.json", {
        "slug": "fill", "label": "fill", "start_s": 40.0, "end_s": 70.0, "handle_s": 2.0, "bpm": 120.0,
        "files": {"youtube": "youtube.mov", "dji": "dji.mov", "x5": "x5.mov"}, "skipped": {}, "inputs": {},
    })
    sizes = {"dji.mov": (3840, 2160), "x5.mov": (5760, 2880)}
    monkeypatch.setattr(looks.media, "probe",
                        lambda p: MediaInfo(path=p, fps=30, width=sizes[p.name][0], height=sizes[p.name][1],
                                            duration_s=34, has_audio=True))
    grabbed = []

    def grab(path, times, width, folder):
        grabbed.append((path.name, times, width))
        return [folder / f"frame_{i}.jpg" for i in range(len(times))]

    monkeypatch.setattr(looks, "grab", grab)
    detector = FakeDetector()
    monkeypatch.setattr(looks, "Detector", lambda: detector)
    service.grabbed, service.detector = grabbed, detector
    return service


def test_writes_a_view_per_person_seen_in_most_frames_and_turns_the_flat_camera(service):
    result = looks.run(looks.Options(service=str(service.root)), config.Config())
    found, patterns = looks_file.parse((service.cuts / "fill" / "looks.toml").read_text(), "looks.toml")
    assert found["dji"].rotate == 90
    assert [round(found[n].yaw) for n in ("person1", "person2")] == [-83, 92]
    assert "person3" not in found
    assert found["wide"].fov > 0
    assert patterns["default"] == ["dji", "person1", "dji", "person2", "dji", "wide", "master"]
    assert result["written"]["fill"]["pattern"] == patterns["default"]
    assert service.detector.previews[0][0] == service.cuts / "fill" / "looks.jpg"


def test_samples_frames_inside_the_moment_not_the_handles(service):
    looks.run(looks.Options(service=str(service.root)), config.Config())
    x5 = next(times for name, times, _ in service.grabbed if name == "x5.mov")
    assert x5 == pytest.approx([5.0, 11.0, 17.0, 23.0, 29.0])


def test_keeps_an_edited_file_unless_forced(service):
    target = service.cuts / "fill" / "looks.toml"
    target.write_text("# mine\n")
    assert looks.run(looks.Options(service=str(service.root)), config.Config())["kept"] == ["fill"]
    assert target.read_text() == "# mine\n"
    looks.run(looks.Options(service=str(service.root), force=True), config.Config())
    assert "person1" in target.read_text()


def test_needs_cuts(tmp_path):
    with pytest.raises(StageError, match="run reels cut first"):
        looks.run(looks.Options(service=str(tmp_path)), config.Config())
