import logging
from pathlib import Path
from typing import ClassVar

import pytest
from yt_dlp.utils import DownloadError

from reels import config
from reels.service import StageError
from reels.stages import fetch

URL = "https://www.youtube.com/watch?v=abc"


@pytest.fixture
def fake_ytdl(monkeypatch):
    class FakeYoutubeDL:
        instances: ClassVar[list] = []
        error = None
        info_name = "youtube.info.json"
        payload = b"video"

        def __init__(self, options):
            self.options = options
            FakeYoutubeDL.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            if FakeYoutubeDL.error:
                raise FakeYoutubeDL.error
            template = Path(self.options["outtmpl"])
            sources = template.parent
            for done in (5, 10, 25, 50, 55, 100):
                for hook in self.options["progress_hooks"]:
                    hook({"status": "downloading", "downloaded_bytes": done, "total_bytes": 100})
            for hook in self.options["progress_hooks"]:
                hook({"status": "finished"})
            Path(str(template).replace("%(ext)s", "mp4")).write_bytes(FakeYoutubeDL.payload)
            stem = template.name.split(".%(ext)s")[0]
            (sources / FakeYoutubeDL.info_name.replace("youtube", stem, 1)).write_text("{}")
            return 0

    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return FakeYoutubeDL


def fetch_args(folder: Path, **overrides) -> fetch.Options:
    return fetch.Options(**{"url": URL, "service": str(folder), **overrides})


def test_downloads_with_configured_format_and_output_template(tmp_path, fake_ytdl):
    cfg = config.Config()
    assert fetch.run(fetch_args(tmp_path), cfg)["ok"]
    options = fake_ytdl.instances[0].options
    assert options["format"] == cfg.fetch.format
    assert options["outtmpl"].endswith("sources/youtube.%(ext)s")
    assert options["merge_output_format"] == "mp4"
    assert options["writeinfojson"] is True
    assert options["noplaylist"] is True
    assert (tmp_path / "sources" / "youtube.mp4").exists()


def test_second_run_without_force_skips_download(tmp_path, fake_ytdl):
    fetch.run(fetch_args(tmp_path), config.Config())
    fetch.run(fetch_args(tmp_path), config.Config())
    assert len(fake_ytdl.instances) == 1


def test_force_downloads_to_a_new_name_then_replaces_the_old_file(tmp_path, fake_ytdl):
    fetch.run(fetch_args(tmp_path), config.Config())
    fake_ytdl.payload = b"video v2"
    fetch.run(fetch_args(tmp_path, force=True), config.Config())
    assert len(fake_ytdl.instances) == 2
    assert fake_ytdl.instances[1].options["outtmpl"].endswith("sources/youtube.refetch.%(ext)s")
    assert (tmp_path / "sources" / "youtube.mp4").read_bytes() == b"video v2"
    assert sorted(p.name for p in (tmp_path / "sources").iterdir()) == ["youtube.info.json", "youtube.mp4"]


def test_failed_forced_download_keeps_the_old_file(tmp_path, fake_ytdl):
    fetch.run(fetch_args(tmp_path), config.Config())
    fake_ytdl.error = DownloadError("network")
    with pytest.raises(StageError):
        fetch.run(fetch_args(tmp_path, force=True), config.Config())
    assert (tmp_path / "sources" / "youtube.mp4").read_bytes() == b"video"


def test_fetch_subcommand_accepts_force(tmp_path, fake_ytdl):
    from reels.cli import main

    assert main(["fetch", URL, "--out", str(tmp_path), "--force"]) == 0


def test_download_error_becomes_stage_error(tmp_path, fake_ytdl):
    fake_ytdl.error = DownloadError("video unavailable")
    with pytest.raises(StageError, match="video unavailable"):
        fetch.run(fetch_args(tmp_path), config.Config())


def test_progress_logs_each_decile_once_and_finish(tmp_path, fake_ytdl, caplog):
    caplog.set_level(logging.INFO, logger="reels.stages.fetch")
    fetch.run(fetch_args(tmp_path), config.Config())
    messages = [r.getMessage() for r in caplog.records if r.name == "reels.stages.fetch"]
    assert [m for m in messages if m.startswith("download")] == [
        "download 10%", "download 20%", "download 50%", "download 100%", "download finished",
    ]


def test_info_json_named_differently_is_renamed(tmp_path, fake_ytdl):
    fake_ytdl.info_name = "youtube.f137.info.json"
    fetch.run(fetch_args(tmp_path), config.Config())
    assert (tmp_path / "sources" / "youtube.info.json").exists()
    assert not (tmp_path / "sources" / "youtube.f137.info.json").exists()


def test_run_command_downloads_only_with_url(tmp_path, fake_ytdl, monkeypatch):
    from reels.cli import main

    monkeypatch.setattr("reels.stages.sync.run", lambda opts, cfg: {"ok": True})
    monkeypatch.setattr("reels.stages.cut.run", lambda opts, cfg: {"ok": True})
    monkeypatch.setattr("reels.stages.render.run", lambda opts, cfg: {"ok": True})
    assert main(["run", str(tmp_path)]) == 0
    assert fake_ytdl.instances == []
    assert main(["run", str(tmp_path), "--url", URL]) == 0
    assert len(fake_ytdl.instances) == 1


def test_forced_refetch_discards_a_leftover_partial_refetch(tmp_path, fake_ytdl):
    fetch.run(fetch_args(tmp_path), config.Config())
    (tmp_path / "sources" / "youtube.refetch.mp4").write_bytes(b"stale partial")
    fake_ytdl.payload = b"video v3"
    fetch.run(fetch_args(tmp_path, force=True), config.Config())
    assert (tmp_path / "sources" / "youtube.mp4").read_bytes() == b"video v3"
