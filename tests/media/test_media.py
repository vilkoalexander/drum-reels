import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from reels import media
from reels.media import ffmpeg as ffmpeg_module
from reels.service import StageError
from reels.tools.doctor import has_videotoolbox

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


needs_videotoolbox = pytest.mark.skipif(not has_videotoolbox(), reason="h264_videotoolbox missing")


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def test_probe(clip):
    info = media.probe(clip)
    assert info.fps == 25
    assert (info.width, info.height) == (160, 90)
    assert info.has_audio
    assert info.duration_s == pytest.approx(2, abs=0.2)
    assert abs(info.audio_offset_s) < 0.05


def test_probe_without_audio(clip, tmp_path):
    silent = tmp_path / "silent.mp4"
    media.run_ffmpeg(["-i", str(clip), "-an", "-c:v", "copy", str(silent)])
    assert not media.probe(silent).has_audio


def test_probe_errors(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(StageError):
        media.probe(missing)
    wav = tmp_path / "a.wav"
    media.run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440", "-t", "1", str(wav)])
    with pytest.raises(StageError, match="no video"):
        media.probe(wav)


def test_run_ffmpeg_raises_with_stderr(tmp_path):
    with pytest.raises(StageError, match="nope"):
        media.run_ffmpeg(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4")])


def test_extract_audio(clip, tmp_path):
    dst = tmp_path / "a.wav"
    assert media.extract_audio(clip, dst, 16000) == dst
    y, sr = media.load_wav(dst)
    assert sr == 16000
    assert y.dtype == "float32"
    assert y.ndim == 1
    assert len(y) == pytest.approx(2 * 16000, rel=0.01)
    assert abs(y).max() <= 1


def test_extract_audio_skips_when_up_to_date(clip, tmp_path):
    dst = tmp_path / "a.wav"
    media.extract_audio(clip, dst, 16000)
    before = dst.stat().st_mtime_ns
    media.extract_audio(clip, dst, 16000)
    assert dst.stat().st_mtime_ns == before


def test_extract_audio_reextracts_when_source_replaced_with_older_file(clip, tmp_path):
    src = tmp_path / "src.mp4"
    shutil.copy(clip, src)
    dst = tmp_path / "a.wav"
    media.extract_audio(src, dst, 16000)
    media.run_ffmpeg(["-i", str(clip), "-t", "1", "-c", "copy", str(src)])
    os.utime(src, ns=(1, 1))
    media.extract_audio(src, dst, 16000)
    assert len(media.load_wav(dst)[0]) == pytest.approx(16000, rel=0.05)


def test_join_chunks(clip, tmp_path):
    dst = tmp_path / "joined.mp4"
    assert media.join_chunks([clip, clip], dst) == dst
    assert media.probe(dst).duration_s == pytest.approx(4, abs=0.1)
    assert dst.with_suffix(".txt").read_text().splitlines() == [f"file '{clip.resolve()}'"] * 2


def test_join_chunks_escapes_quotes(clip, tmp_path):
    quoted = tmp_path / "it's.mp4"
    shutil.copy(clip, quoted)
    dst = tmp_path / "joined.mp4"
    media.join_chunks([quoted, quoted], dst)
    assert "'\\''" in dst.with_suffix(".txt").read_text()
    assert media.probe(dst).duration_s == pytest.approx(4, abs=0.1)


def test_extract_audio_reextracts_at_different_rate(clip, tmp_path):
    dst = tmp_path / "a.wav"
    media.extract_audio(clip, dst, 16000)
    media.extract_audio(clip, dst, 8000)
    assert media.load_wav(dst)[1] == 8000


def test_extract_audio_force_rewrites(clip, tmp_path):
    dst = tmp_path / "a.wav"
    media.extract_audio(clip, dst, 16000)
    before = dst.stat().st_mtime_ns
    media.extract_audio(clip, dst, 16000)
    assert dst.stat().st_mtime_ns == before
    media.extract_audio(clip, dst, 16000, force=True)
    assert dst.stat().st_mtime_ns > before


def test_extract_audio_failure_leaves_no_partial_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not media")
    dst = tmp_path / "a.wav"
    with pytest.raises(StageError):
        media.extract_audio(bad, dst, 16000)
    assert list(tmp_path.glob("a*")) == []


def test_extract_audio_failure_keeps_previous_output(clip, tmp_path):
    dst = tmp_path / "a.wav"
    media.extract_audio(clip, dst, 16000)
    bad = tmp_path / "bad.mp4"
    bad.write_text("not media")
    with pytest.raises(StageError):
        media.extract_audio(bad, dst, 16000)
    assert media.load_wav(dst)[1] == 16000
    assert not dst.with_name("a.part.wav").exists()


def test_load_wav_of_garbage_is_stage_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    with pytest.raises(StageError, match="bad.wav"):
        media.load_wav(bad)
    with pytest.raises(StageError, match="bad.wav"):
        media.read_wav_region(bad, 0, 1)


def test_join_chunks_rebuilds_when_chunk_list_differs(clip, tmp_path):
    dst = tmp_path / "joined.mp4"
    media.join_chunks([clip, clip], dst)
    assert media.probe(dst).duration_s == pytest.approx(4, abs=0.1)
    media.join_chunks([clip, clip, clip], dst)
    assert media.probe(dst).duration_s == pytest.approx(6, abs=0.1)
    assert dst.with_suffix(".txt").read_text().count("file '") == 3


def test_join_chunks_reuses_matching_output(clip, tmp_path):
    dst = tmp_path / "joined.mp4"
    media.join_chunks([clip, clip], dst)
    before = dst.stat().st_mtime_ns
    media.join_chunks([clip, clip], dst)
    assert dst.stat().st_mtime_ns == before
    media.join_chunks([clip, clip], dst, force=True)
    assert dst.stat().st_mtime_ns > before


def test_join_chunks_rebuilds_when_a_chunk_changes(clip, tmp_path):
    chunk = tmp_path / "chunk.mp4"
    shutil.copy(clip, chunk)
    dst = tmp_path / "joined.mp4"
    media.join_chunks([chunk, chunk], dst)
    media.run_ffmpeg(["-i", str(clip), "-t", "1", "-c", "copy", str(chunk)])
    os.utime(chunk, ns=(1, 1))
    media.join_chunks([chunk, chunk], dst)
    assert media.probe(dst).duration_s == pytest.approx(2, abs=0.2)


def test_join_chunks_failure_leaves_no_partial_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not media")
    dst = tmp_path / "joined.mp4"
    with pytest.raises(StageError):
        media.join_chunks([bad, bad], dst)
    assert not dst.exists()
    assert not dst.with_name("joined.part.mp4").exists()
    assert not dst.with_name("joined.mp4.src.json").exists()


def test_join_single_chunk_writes_nothing(clip, tmp_path):
    dst = tmp_path / "joined.mp4"
    assert media.join_chunks([clip], dst) == clip
    assert list(tmp_path.iterdir()) == []


@needs_videotoolbox
def test_cut(clip, tmp_path):
    dst = tmp_path / "cuts" / "out.mov"
    assert media.cut(clip, None, 0.5, 0.5, dst, "h264", 10) == dst
    info = media.probe(dst)
    assert 0.4 <= info.duration_s <= 0.6
    assert info.has_audio
    assert not dst.with_name("out.part.mov").exists()


def test_cut_prores(clip, tmp_path):
    dst = tmp_path / "out.mov"
    media.cut(clip, None, 0.5, 0.5, dst, "prores", 10)
    info = media.probe(dst)
    assert 0.4 <= info.duration_s <= 0.6
    assert info.has_audio


def test_cut_rejects_unknown_codec_and_empty_window(clip, tmp_path):
    with pytest.raises(ValueError, match="unknown codec"):
        media.cut(clip, None, 0, 1, tmp_path / "out.mov", "vp9", 10)
    with pytest.raises(ValueError, match="positive"):
        media.cut(clip, None, 1, 0, tmp_path / "out.mov", "prores", 10)


def test_cut_failure_keeps_previous_output(clip, tmp_path):
    dst = tmp_path / "out.mov"
    media.cut(clip, None, 0.5, 0.5, dst, "prores", 10)
    before = dst.stat().st_size
    bad = tmp_path / "bad.mp4"
    bad.write_text("not media")
    with pytest.raises(StageError):
        media.cut(bad, None, 0.5, 0.5, dst, "prores", 10)
    assert dst.stat().st_size == before
    assert not dst.with_name("out.part.mov").exists()


@pytest.fixture(scope="module")
def stepped_audio(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("stepped") / "stepped.wav"
    media.run_ffmpeg(
        ["-f", "lavfi", "-i", "aevalsrc=sin(2*PI*t*if(lt(t\\,1)\\,440\\,1500)):s=48000:d=3", str(path)]
    )
    return path


def dominant_hz(path: Path, tmp_path: Path) -> float:
    wav = media.extract_audio(path, tmp_path / "check.wav", 16000)
    y, sr = media.load_wav(wav)
    spectrum = np.abs(np.fft.rfft(y))
    return float(np.fft.rfftfreq(len(y), 1 / sr)[spectrum.argmax()])


def silent_video(clip: Path, tmp_path: Path) -> Path:
    silent = tmp_path / "silent.mp4"
    media.run_ffmpeg(["-i", str(clip), "-an", "-c:v", "copy", str(silent)])
    return silent


def test_cut_uses_audio_start(clip, stepped_audio, tmp_path):
    dst = tmp_path / "out.mov"
    media.cut(silent_video(clip, tmp_path), stepped_audio, 0.5, 0.4, dst, "prores", 10, audio_start_s=1.2)
    assert media.probe(dst).has_audio
    assert dominant_hz(dst, tmp_path) == pytest.approx(1500, abs=30)


def test_cut_audio_start_defaults_to_start(clip, stepped_audio, tmp_path):
    dst = tmp_path / "out.mov"
    media.cut(silent_video(clip, tmp_path), stepped_audio, 0.5, 0.4, dst, "prores", 10)
    assert dominant_hz(dst, tmp_path) == pytest.approx(440, abs=30)


def test_cut_rejects_non_mov(clip, tmp_path):
    with pytest.raises(ValueError):
        media.cut(clip, None, 0, 1, tmp_path / "out.mp4", "h264", 10)


def fake_ffprobe(monkeypatch, streams, fmt):
    import json
    import subprocess as sp

    class Result:
        returncode = 0
        stdout = json.dumps({"streams": streams, "format": fmt})
        stderr = ""

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", lambda *a, **k: Result())
    return sp


VIDEO = {"codec_type": "video", "r_frame_rate": "30/1", "avg_frame_rate": "30/1", "width": 16, "height": 9,
         "start_time": "0.000"}


def test_probe_rejects_missing_duration_and_zero_frame_rate(monkeypatch, tmp_path):
    fake_ffprobe(monkeypatch, [VIDEO], {})
    with pytest.raises(StageError, match="no duration"):
        media.probe(tmp_path / "a.mp4")
    fake_ffprobe(monkeypatch, [{**VIDEO, "r_frame_rate": "0/0"}], {"duration": "1.0"})
    with pytest.raises(StageError, match="frame rate"):
        media.probe(tmp_path / "a.mp4")


def test_probe_warns_on_variable_frame_rate_and_reports_audio_offset(monkeypatch, tmp_path, caplog):
    caplog.set_level("WARNING")
    audio = {"codec_type": "audio", "start_time": "0.477"}
    fake_ffprobe(monkeypatch, [{**VIDEO, "avg_frame_rate": "25/1"}, audio], {"duration": "2.0"})
    info = media.probe(tmp_path / "a.mp4")
    assert "variable frame rate" in caplog.text
    assert info.audio_offset_s == pytest.approx(0.477)
    assert info.has_audio and info.fps == 30


def test_probe_prefers_the_video_stream_duration(monkeypatch, tmp_path):
    fake_ffprobe(monkeypatch, [{**VIDEO, "duration": "1.90"}], {"duration": "2.0"})
    assert media.probe(tmp_path / "a.mp4").duration_s == pytest.approx(1.90)


def test_join_rejects_a_file_name_with_a_line_break(clip, tmp_path):
    odd = tmp_path / "a\nb.mp4"
    shutil.copy(clip, odd)
    with pytest.raises(StageError, match="line break"):
        media.join_chunks([odd, odd], tmp_path / "joined.mp4")
