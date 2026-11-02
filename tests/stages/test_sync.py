import dataclasses
import json
import logging
from pathlib import Path

import pytest

from reels import config
from reels.media import MediaInfo
from reels.service import Service, StageError
from reels.stages import sync
from reels.tools import synth

SR = 16000


class FakeMedia:
    def __init__(self, monkeypatch):
        yt = synth.bursts(90, SR, seed=1)
        self.audio = {
            "youtube": yt,
            "dji": synth.camera(yt, SR, 5.0, 1.0, 0.0, 60, 12, seed=2),
            "x5": synth.camera(yt, SR, -7.0, 1.0, 0.0, 60, 12, seed=3),
        }
        self.fail_probe: set[str] = set()
        self.rates: dict[str, int] = {}
        self.offsets: dict[str, float] = {}
        for name in ("probe", "extract_audio", "load_wav", "join_chunks", "audio_fresh", "join_fresh"):
            monkeypatch.setattr(sync.media, name, getattr(self, name))

    @staticmethod
    def _key(path):
        return Path(path).name.split(".")[0].split("_")[0]

    def probe(self, path):
        key = self._key(path)
        if key in self.fail_probe:
            raise StageError("probe boom")
        fps = {"youtube": 30.0, "dji": 60.0, "x5": 30.0}[key]
        return MediaInfo(Path(path), fps, 1920, 1080, 60.0, True, self.offsets.get(key, 0.0))

    @staticmethod
    def audio_fresh(src, dst, sample_rate):
        return True

    @staticmethod
    def join_fresh(chunks, dst):
        return True

    def extract_audio(self, src, dst, sample_rate, force=False):
        return Path(self._key(src))

    def load_wav(self, path):
        return self.audio[str(path)], self.rates.get(str(path), SR)

    def join_chunks(self, chunks, dst, force=False):
        if len(chunks) == 1:
            return chunks[0]
        dst.write_bytes(b"joined")
        return dst


def make_service(tmp_path, files):
    for name in files:
        path = tmp_path / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return sync.Options(service=str(tmp_path))


def sidecar(tmp_path):
    return json.loads((tmp_path / "work" / "sync.json").read_text())


def test_stage_writes_sources_and_joins_chunks(tmp_path, monkeypatch):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4", "dji_002.mp4", "x5.mp4"])
    assert sync.run(args, config.Config())["ok"]
    data = sidecar(tmp_path)
    dji, x5 = data["sources"]["dji"], data["sources"]["x5"]
    assert dji["path"] == "work/dji.joined.mp4"
    assert dji["chunks"] == ["sources/dji_001.mp4", "sources/dji_002.mp4"]
    assert x5["path"] == "sources/x5.mp4"
    assert dji["intercept_s"] == pytest.approx(5.0, abs=0.01)
    assert x5["intercept_s"] == pytest.approx(-7.0, abs=0.01)
    assert dji["manual"] is False


def test_no_cameras_writes_empty_sources(tmp_path, monkeypatch):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4"])
    assert sync.run(args, config.Config())["ok"]
    assert sidecar(tmp_path)["sources"] == {}


def test_manual_entry_survives_rerun_and_probe_failure(tmp_path, monkeypatch):
    fake = FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    sync.run(args, config.Config())
    data = sidecar(tmp_path)
    data["sources"]["dji"].update(manual=True, intercept_s=99.0, slope=1.0001)
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    args = dataclasses.replace(args, force=True)
    assert sync.run(args, config.Config())["ok"]
    kept = sidecar(tmp_path)["sources"]["dji"]
    assert (kept["intercept_s"], kept["slope"], kept["manual"]) == (99.0, 1.0001, True)

    fake.fail_probe.add("dji")
    assert not sync.run(args, config.Config())["ok"]
    failed = sidecar(tmp_path)["sources"]["dji"]
    assert failed["failed"] is True
    assert (failed["intercept_s"], failed["slope"], failed["manual"]) == (99.0, 1.0001, True)

    fake.fail_probe.clear()
    assert sync.run(args, config.Config())["ok"]
    again = sidecar(tmp_path)["sources"]["dji"]
    assert "failed" not in again
    assert (again["intercept_s"], again["slope"]) == (99.0, 1.0001)


def test_failed_source_is_isolated_and_rerun_retries(tmp_path, monkeypatch):
    fake = FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4", "x5.mp4"])
    fake.fail_probe.add("x5")
    assert not sync.run(args, config.Config())["ok"]
    sources = sidecar(tmp_path)["sources"]
    assert "intercept_s" in sources["dji"]
    assert sources["x5"] == {"chunks": ["sources/x5.mp4"], "failed": True, "error": "probe boom"}

    assert not sync.run(args, config.Config())["ok"]
    fake.fail_probe.clear()
    assert sync.run(args, config.Config())["ok"]
    assert "failed" not in sidecar(tmp_path)["sources"]["x5"]
    assert sync.run(args, config.Config())["ok"]


def test_sample_rate_mismatch_fails_that_camera_naming_both_rates(tmp_path, monkeypatch):
    fake = FakeMedia(monkeypatch)
    fake.rates["x5"] = 8000
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4", "x5.mp4"])
    assert not sync.run(args, config.Config())["ok"]
    failed = sidecar(tmp_path)["sources"]["x5"]
    assert "16000" in failed["error"] and "8000" in failed["error"]
    assert "intercept_s" in sidecar(tmp_path)["sources"]["dji"]


def test_force_is_passed_to_media(tmp_path, monkeypatch):
    fake = FakeMedia(monkeypatch)
    forces = []
    monkeypatch.setattr(sync.media, "extract_audio",
                        lambda src, dst, rate, force=False: forces.append(("audio", force)) or fake.extract_audio(src, dst, rate))
    monkeypatch.setattr(sync.media, "join_chunks",
                        lambda chunks, dst, force=False: forces.append(("join", force)) or fake.join_chunks(chunks, dst))
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4", "dji_002.mp4"])
    args = dataclasses.replace(args, force=True)
    sync.run(args, config.Config())
    assert forces and all(force for _, force in forces)
    assert {kind for kind, _ in forces} == {"audio", "join"}


def test_cameras_without_files_are_logged(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    caplog.set_level(logging.INFO)
    sync.run(make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"]), config.Config())
    assert "x5: no files under sources/, skipped" in caplog.text
    assert "dji: no files" not in caplog.text


def test_low_confidence_marks_source_failed_and_warn_band_only_logs(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    assert not sync.run(args, config.Config(sync=config.SyncCfg(min_confidence=1e6)))["ok"]
    entry = sidecar(tmp_path)["sources"]["dji"]
    assert entry["failed"] is True and "below minimum" in entry["error"]
    caplog.set_level(logging.WARNING)
    assert sync.run(args, config.Config(sync=config.SyncCfg(min_confidence=0, warn_confidence=1e6)))["ok"]
    assert "confidence" in caplog.text and "is low" in caplog.text
    assert "failed" not in sidecar(tmp_path)["sources"]["dji"]


def test_manual_entry_without_slope_fails_that_camera(tmp_path, monkeypatch):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    sync.run(args, config.Config())
    data = sidecar(tmp_path)
    data["sources"]["dji"] = {"manual": True, "intercept_s": 99.0}
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    args = dataclasses.replace(args, force=True)
    assert not sync.run(args, config.Config())["ok"]
    failed = sidecar(tmp_path)["sources"]["dji"]
    assert failed["failed"] is True and "slope" in failed["error"] and failed["manual"] is True


def test_fingerprint_matches_what_run_stores(tmp_path, monkeypatch):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    sync.run(args, config.Config())
    assert sidecar(tmp_path)["inputs"] == sync.fingerprint(Service(tmp_path), config.Config())


def test_fallback_fit_is_recorded_honestly(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    monkeypatch.setattr(sync, "fine_points", lambda *a, **k: [(10.0, 15.02, 9.0)])
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    caplog.set_level(logging.WARNING)
    assert sync.run(args, config.Config())["ok"]
    entry = sidecar(tmp_path)["sources"]["dji"]
    assert entry["fit"] == "median" and entry["slope"] == 1.0 and entry["fine_residual_ms"] is None
    assert entry["windows"][0]["used"] is False
    assert entry["intercept_s"] == pytest.approx(5.02)
    assert "drift is unknown" in caplog.text


def test_audio_offset_and_dropped_manual_entry_are_warned(tmp_path, monkeypatch, caplog):
    fake = FakeMedia(monkeypatch)
    fake.offsets["dji"] = 0.2
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4", "x5.mp4"])
    caplog.set_level(logging.WARNING)
    sync.run(args, config.Config())
    assert "dji audio starts +0.200 s after its video" in caplog.text
    data = sidecar(tmp_path)
    data["sources"]["x5"].update(manual=True)
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    (tmp_path / "sources" / "x5.mp4").unlink()
    caplog.clear()
    args = dataclasses.replace(args, force=True)
    sync.run(args, config.Config())
    assert "x5: no files under sources/, its manual sync entry is dropped" in caplog.text


def test_manual_entry_fails_when_its_chunks_change(tmp_path, monkeypatch):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_002.mp4"])
    sync.run(args, config.Config())
    data = sidecar(tmp_path)
    data["sources"]["dji"].update(manual=True, intercept_s=99.0, slope=1.0)
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    (tmp_path / "sources" / "dji_001.mp4").write_bytes(b"x")
    assert not sync.run(args, config.Config())["ok"]
    failed = sidecar(tmp_path)["sources"]["dji"]
    assert failed["failed"] is True and "chunks changed" in failed["error"] and failed["intercept_s"] == 99.0


def test_validate_names_the_missing_field():
    with pytest.raises(StageError, match="youtube entry is missing"):
        sync.validate({"youtube": {}, "sources": {}})
    good = {"path": "x", "fps": 30.0, "width": 1, "height": 1, "duration_s": 1.0}
    with pytest.raises(StageError, match="dji entry is missing chunks"):
        sync.validate({"youtube": good, "sources": {"dji": {**good, "intercept_s": 0.0, "slope": 1.0}}})
    sync.validate({"youtube": good, "sources": {"dji": {"failed": True}}})


def test_stale_derived_audio_forces_a_rerun_except_for_manual_cameras(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    sync.run(args, config.Config())
    monkeypatch.setattr(sync.media, "audio_fresh", lambda src, dst, rate: dst.name != "dji.wav")
    caplog.set_level(logging.INFO)
    sync.run(args, config.Config())
    assert "up to date" not in caplog.text
    data = sidecar(tmp_path)
    data["sources"]["dji"].update(manual=True)
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    caplog.clear()
    sync.run(args, config.Config())
    assert "up to date" in caplog.text


def test_several_disagreeing_windows_fall_back_to_their_median_and_two_windows_warn(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    monkeypatch.setattr(sync, "fine_points", lambda *a, **k: [(10.0, 15.0, 9.0), (20.0, 25.5, 9.0), (30.0, 34.0, 9.0)])
    caplog.set_level(logging.WARNING)
    sync.run(args, config.Config())
    entry = sidecar(tmp_path)["sources"]["dji"]
    assert entry["fit"] == "median" and entry["intercept_s"] == pytest.approx(5.0) and entry["slope"] == 1.0
    monkeypatch.setattr(sync, "fine_points", lambda *a, **k: [(10.0, 15.0, 9.0), (20.0, 25.0, 9.0)])
    caplog.clear()
    args = dataclasses.replace(args, force=True)
    sync.run(args, config.Config())
    assert sidecar(tmp_path)["sources"]["dji"]["fit"] == "windows"
    assert "only 2 fine windows agree" in caplog.text


def test_hand_edited_sync_json_that_is_unusable_is_redone_not_trusted(tmp_path, monkeypatch, caplog):
    FakeMedia(monkeypatch)
    args = make_service(tmp_path, ["youtube.mp4", "dji_001.mp4"])
    sync.run(args, config.Config())
    data = sidecar(tmp_path)
    data["sources"]["dji"].pop("path")
    (tmp_path / "work" / "sync.json").write_text(json.dumps(data))
    caplog.set_level(logging.WARNING)
    assert sync.run(args, config.Config())["ok"]
    assert "not usable as is" in caplog.text and "path" in sidecar(tmp_path)["sources"]["dji"]
    with pytest.raises(StageError, match="unknown camera 'gopro'"):
        sync.validate({**sidecar(tmp_path), "sources": {"gopro": {"failed": True}}})
