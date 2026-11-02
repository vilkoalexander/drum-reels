import json

from reels import cli
from reels.tools import doctor


def test_checks_name_every_encoder_and_filter(tmp_path):
    rows = {name: (ok, note) for name, ok, note in doctor.checks(tmp_path)}
    assert "ffmpeg" in rows and "ffprobe" in rows and "yt-dlp" in rows and "disk free" in rows
    for name in doctor.ENCODERS:
        assert f"encoder {name}" in rows
    for name in doctor.FILTERS:
        assert f"filter {name}" in rows
    assert rows["python"][0] is True


def test_doctor_reports_only_missing_checks_and_fails_without_ffmpeg(tmp_path, capsys, monkeypatch):
    cli.main(["doctor", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert "python" not in out["missing"]
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    assert cli.main(["doctor", str(tmp_path)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "ffmpeg" in out["missing"]
