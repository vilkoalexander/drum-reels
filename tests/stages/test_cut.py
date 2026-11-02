import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from reels import config
from reels.analysis.beats import Grid
from reels.service import StageError, UsageError
from reels.stages import cut

CAMS = {
    "dji": {"path": "work/dji.joined.mp4", "intercept_s": -10.0, "slope": 1.0, "duration_s": 100.0},
    "x5": {"path": "sources/x5.mp4", "intercept_s": 5.0, "slope": 1.0001, "duration_s": 300.0},
}


def test_plan_sources_camera_covering_window_starts_at_mapped_time():
    starts, skipped = cut.plan_sources(40.0, 60.0, CAMS)
    assert starts["dji"] == pytest.approx(30.0)
    assert starts["x5"] == pytest.approx(5.0 + 1.0001 * 40.0)
    assert skipped == {}


def test_plan_sources_recording_starts_after_window():
    starts, skipped = cut.plan_sources(4.0, 20.0, CAMS)
    assert "dji" not in starts
    assert skipped["dji"] == "window [4.0, 20.0] maps to cam [-6.0, 10.0], recording starts at 0"
    assert "x5" in starts


def test_plan_sources_recording_ends_before_window():
    starts, skipped = cut.plan_sources(100.0, 130.0, CAMS)
    assert "dji" not in starts
    assert skipped["dji"] == "window [100.0, 130.0] maps to cam [90.0, 120.0], recording ends at 100.0"


def test_plan_sources_rejects_entry_without_slope():
    with pytest.raises(StageError, match="dji.*slope"):
        cut.plan_sources(10.0, 20.0, {"dji": {"intercept_s": 0.0, "duration_s": 100.0}})


def test_plan_sources_checks_the_length_ffmpeg_will_read_not_only_the_mapped_end():
    slow = {"dji": {"path": "x", "intercept_s": 0.0, "slope": 0.999, "duration_s": 999.5}}
    starts, skipped = cut.plan_sources(0.0, 1000.0, slow)
    assert starts == {} and "recording ends" in skipped["dji"]


def test_plan_sources_failed_source_is_skipped_with_error():
    starts, skipped = cut.plan_sources(10.0, 20.0, {"dji": {"failed": True, "error": "low confidence"}})
    assert starts == {}
    assert skipped == {"dji": "sync failed: low confidence"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    root = tmp_path / "svc"
    for folder in ("sources", "work/audio", "notes"):
        (root / folder).mkdir(parents=True)
    (root / "sources/youtube.mp4").write_bytes(b"yt")
    (root / "work/dji.joined.mp4").write_bytes(b"dji")
    (root / "notes/moments.txt").write_text("0:00:40 fill\n0:02:30 groove 20\n")
    sf.write(root / "work/audio/youtube.wav", np.zeros(240 * 100, dtype=np.float32), 100)
    sync = {
        "version": 1,
        "youtube": {"path": "sources/youtube.mp4", "fps": 30.0, "width": 1280, "height": 720, "duration_s": 240.0},
        "sources": {"dji": {"chunks": ["sources/dji.mp4"], "path": "work/dji.joined.mp4", "fps": 60.0, "width": 1920,
                            "height": 1080, "intercept_s": -10.0, "slope": 1.0, "duration_s": 200.0}},
    }
    (root / "work/sync.json").write_text(json.dumps(sync))
    monkeypatch.setattr(cut.media, "extract_audio", lambda src, dst, rate, force=False: dst)

    calls = []
    grids = []

    def fake_cut(video, audio, start, duration, dst, codec, bitrate, audio_start_s=None):
        calls.append((video.name, audio and audio.name, start, duration, dst.name, audio_start_s))
        dst.write_bytes(b"mov")
        return dst

    monkeypatch.setattr(cut.media, "cut", fake_cut)
    def fake_grid(y, sr, start, bpb):
        grids.append((len(y), sr, start))
        return Grid(120.0, [1.0, 1.5], [1.0])

    monkeypatch.setattr(cut.beats, "grid", fake_grid)
    monkeypatch.setattr(cut.beats, "snap", lambda grid, noted, length, snap: (noted - 1.0, noted - 1.0 + length))
    return SimpleNamespace(root=root, calls=calls, grids=grids)


def args(service, force=False):
    return cut.Options(service=str(service.root), force=force)


def test_run_writes_moments_and_beats(service):
    assert cut.run(args(service), config.Config())["ok"]
    moment = json.loads((service.root / "cuts/0-00-40_fill/moment.json").read_text())
    assert moment["start_s"] == 39.0 and moment["end_s"] == 69.0 and moment["bpm"] == 120.0
    assert moment["files"] == {"youtube": "youtube.mov", "dji": "dji.mov"}
    assert moment["skipped"] == {}
    assert (service.root / "cuts/0-00-40_fill/dji.mov").exists()
    assert ("youtube.mp4", None, 37.0, 34.0, "youtube.mov", None) in service.calls[:2]
    assert ("dji.joined.mp4", "youtube.mp4", 27.0, 34.0, "dji.mov", 37.0) in service.calls[:2]

    groove = json.loads((service.root / "cuts/0-02-30_groove/moment.json").read_text())
    assert groove["end_s"] == 169.0
    assert set(groove["files"]) == {"youtube", "dji"}

    beats = json.loads((service.root / "work/beats.json").read_text())
    assert beats["version"] == 1
    assert beats["moments"]["0-00-40_fill"] == {
        "bpm": 120.0, "beats_s": [1.0, 1.5], "downbeats_s": [1.0],
        "noted_s": 40.0, "start_s": 39.0, "end_s": 69.0,
    }


def test_run_skips_camera_outside_recording(service):
    (service.root / "notes/moments.txt").write_text("0:03:35 solo 15\n")
    assert cut.run(args(service), config.Config())["ok"]
    moment = json.loads((service.root / "cuts/0-03-35_solo/moment.json").read_text())
    assert moment["files"] == {"youtube": "youtube.mov"}
    assert moment["skipped"] == {"dji": "window [212.0, 231.0] maps to cam [202.0, 221.0], recording ends at 200.0"}


def test_run_rerun_is_up_to_date_and_keeps_beats(service, caplog):
    cut.run(args(service), config.Config())
    first = len(service.calls)
    caplog.set_level(logging.INFO)
    assert cut.run(args(service), config.Config())["ok"]
    assert len(service.calls) == first
    assert "cuts/0-00-40_fill up to date" in caplog.text
    beats = json.loads((service.root / "work/beats.json").read_text())
    assert set(beats["moments"]) == {"0-00-40_fill", "0-02-30_groove"}


def test_run_force_recuts(service):
    cut.run(args(service), config.Config())
    first = len(service.calls)
    cut.run(args(service, force=True), config.Config())
    assert len(service.calls) == 2 * first


def test_run_logs_and_drops_removed_moment(service, caplog):
    cut.run(args(service), config.Config())
    (service.root / "notes/moments.txt").write_text("0:00:40 fill\n")
    caplog.set_level(logging.INFO)
    cut.run(args(service), config.Config())
    assert "cuts/0-02-30_groove no longer in notes" in caplog.text
    beats = json.loads((service.root / "work/beats.json").read_text())
    assert list(beats["moments"]) == ["0-00-40_fill"]
    assert (service.root / "cuts/0-02-30_groove").is_dir()


def test_run_failed_cut_returns_1_and_continues(service, monkeypatch):
    real = cut.media.cut

    def flaky(video, audio, start, duration, dst, *rest, **kw):
        if "groove" in str(dst):
            raise StageError("ffmpeg failed")
        return real(video, audio, start, duration, dst, *rest, **kw)

    monkeypatch.setattr(cut.media, "cut", flaky)
    assert not cut.run(args(service), config.Config())["ok"]
    beats = json.loads((service.root / "work/beats.json").read_text())
    assert list(beats["moments"]) == ["0-00-40_fill"]


def test_run_requires_sync_and_notes(service):
    (service.root / "notes/moments.txt").unlink()
    with pytest.raises(UsageError, match="moments.txt"):
        cut.run(args(service), config.Config())
    (service.root / "work/sync.json").unlink()
    with pytest.raises(StageError, match="run reels sync first"):
        cut.run(args(service), config.Config())


def test_run_bad_notes_line_is_usage_error(service):
    (service.root / "notes/moments.txt").write_text("nonsense\n")
    with pytest.raises(UsageError, match="line 1"):
        cut.run(args(service), config.Config())


def test_run_duplicate_slug_is_stage_error_naming_lines(service):
    (service.root / "notes/moments.txt").write_text("# x\n0:00:40 fill\n0:00:40 fill 20\n")
    with pytest.raises(StageError, match=r"0-00-40_fill.*lines 2 and 3"):
        cut.run(args(service), config.Config())
    assert service.calls == []


def test_run_reads_only_the_beat_region(service):
    cut.run(args(service), config.Config())
    assert service.grids == [(130 * 100, 100, 0.0), (140 * 100, 100, 90.0)]


def test_run_note_past_the_end_fails_that_moment_cleanly(service, caplog):
    (service.root / "notes/moments.txt").write_text("0:00:40 fill\n0:10:00 beyond 10\n")
    assert not cut.run(args(service), config.Config())["ok"]
    assert "past the end of the stream" in caplog.text
    assert not (service.root / "cuts/0-10-00_beyond").exists()
    beats = json.loads((service.root / "work/beats.json").read_text())
    assert list(beats["moments"]) == ["0-00-40_fill"]


def test_run_recuts_when_the_note_changes(service):
    cut.run(args(service), config.Config())
    first = len(service.calls)
    (service.root / "notes/moments.txt").write_text("0:00:40 fill 12\n0:02:30 groove 20\n")
    cut.run(args(service), config.Config())
    assert [c[4] for c in service.calls[first:]] == ["youtube.mov", "dji.mov"]
    assert {c[2] for c in service.calls[first:]} == {37.0, 27.0}
    moment = json.loads((service.root / "cuts/0-00-40_fill/moment.json").read_text())
    assert moment["end_s"] == 51.0


def test_run_recut_removes_stale_camera_files(service):
    cut.run(args(service), config.Config())
    folder = service.root / "cuts/0-00-40_fill"
    assert (folder / "dji.mov").exists()
    sync = json.loads((service.root / "work/sync.json").read_text())
    sync["sources"]["dji"]["duration_s"] = 10.0
    (service.root / "work/sync.json").write_text(json.dumps(sync))
    cut.run(args(service), config.Config())
    assert not (folder / "dji.mov").exists()
    moment = json.loads((folder / "moment.json").read_text())
    assert moment["files"] == {"youtube": "youtube.mov"} and "dji" in moment["skipped"]


def test_run_refuses_when_sources_changed_since_sync(service):
    sync = json.loads((service.root / "work/sync.json").read_text())
    sync["inputs"] = {"files": {"sources/youtube.mp4": {"size": 1, "mtime_ns": 1}}, "params": {}}
    (service.root / "work/sync.json").write_text(json.dumps(sync))
    with pytest.raises(StageError, match="changed since the last sync"):
        cut.run(args(service), config.Config())
    assert service.calls == []


def test_run_falls_back_to_raw_window_when_snap_collapses(service, monkeypatch):
    monkeypatch.setattr(cut.beats, "snap", lambda grid, noted, length, snap: (noted, noted))
    (service.root / "notes/moments.txt").write_text("0:00:40 fill 10\n")
    assert cut.run(args(service), config.Config())["ok"]
    moment = json.loads((service.root / "cuts/0-00-40_fill/moment.json").read_text())
    assert (moment["start_s"], moment["end_s"]) == (40.0, 50.0)


def test_run_passes_the_configured_minimum_length_to_the_notes_parser(service):
    (service.root / "notes/moments.txt").write_text("0:00:40 verse 7\n")
    with pytest.raises(UsageError, match="below the minimum 10 s"):
        cut.run(args(service), config.Config(cut=config.CutCfg(min_len_s=10)))
    assert cut.run(args(service), config.Config(cut=config.CutCfg(min_len_s=6)))["ok"]


def test_run_cleans_up_after_an_interrupted_recut(service, monkeypatch):
    cut.run(args(service), config.Config())
    folder = service.root / "cuts/0-00-40_fill"
    sync = json.loads((service.root / "work/sync.json").read_text())
    sync["sources"]["dji"]["duration_s"] = 10.0
    (service.root / "work/sync.json").write_text(json.dumps(sync))
    real = cut.media.cut
    calls = {"n": 0}

    def once(video, audio, start, duration, dst, *rest, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise StageError("interrupted")
        return real(video, audio, start, duration, dst, *rest, **kw)

    monkeypatch.setattr(cut.media, "cut", once)
    assert not cut.run(args(service), config.Config())["ok"]
    assert (folder / "dji.mov").exists()
    assert json.loads((folder / "moment.json").read_text())["inputs"] is None
    assert cut.run(args(service), config.Config())["ok"]
    assert not (folder / "dji.mov").exists()
    assert json.loads((folder / "moment.json").read_text())["files"] == {"youtube": "youtube.mov"}


def test_run_accepts_symlinked_sources(service, tmp_path):
    outside = tmp_path / "ssd" / "DJI_0001.MP4"
    outside.parent.mkdir()
    outside.write_bytes(b"dji")
    (service.root / "work/dji.joined.mp4").unlink()
    (service.root / "work/dji.joined.mp4").symlink_to(outside)
    assert cut.run(args(service), config.Config())["ok"]
    assert (service.root / "cuts/0-00-40_fill/dji.mov").exists()


def test_run_keeps_user_files_and_old_cuts_until_the_recut_succeeds(service, monkeypatch):
    cut.run(args(service), config.Config())
    folder = service.root / "cuts/0-00-40_fill"
    (folder / "graded.mov").write_bytes(b"mine")
    real = cut.media.cut

    def failing(video, audio, start, duration, dst, *rest, **kw):
        if dst.name == "dji.mov":
            raise StageError("disk full")
        return real(video, audio, start, duration, dst, *rest, **kw)

    monkeypatch.setattr(cut.media, "cut", failing)
    assert not cut.run(args(service, force=True), config.Config())["ok"]
    assert (folder / "graded.mov").read_bytes() == b"mine"
    assert (folder / "dji.mov").exists()
    assert json.loads((folder / "moment.json").read_text())["inputs"] is None


def test_run_rejects_sidecar_paths_outside_the_service(service):
    sync = json.loads((service.root / "work/sync.json").read_text())
    sync["sources"]["dji"]["path"] = "../../etc/passwd"
    (service.root / "work/sync.json").write_text(json.dumps(sync))
    assert not cut.run(args(service), config.Config())["ok"]
    assert service.calls == []


def test_run_clamps_the_window_end_to_the_stream(service, monkeypatch):
    (service.root / "notes/moments.txt").write_text("0:03:55 tail 30\n")
    assert cut.run(args(service), config.Config())["ok"]
    moment = json.loads((service.root / "cuts/0-03-55_tail/moment.json").read_text())
    assert moment["end_s"] == 240.0
    assert service.calls[0][3] == pytest.approx(240.0 - 232.0)
