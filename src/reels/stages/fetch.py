import logging
import os
from dataclasses import dataclass

import yt_dlp

from reels.service import Service, StageError

log = logging.getLogger(__name__)

REFETCH_STEM = "youtube.refetch"


@dataclass(frozen=True)
class Options:
    service: str
    url: str
    force: bool = False


def run(opts: Options, cfg) -> dict:
    service = Service(opts.service)
    service.ensure()
    if service.youtube_path.exists() and not opts.force:
        log.info("sources/youtube.mp4 exists, skipping fetch")
        return {"ok": True, "video": str(service.youtube_path), "up_to_date": True}
    refetch = opts.force and service.youtube_path.exists()
    stem = REFETCH_STEM if refetch else "youtube"
    if refetch:
        for leftover in service.sources.glob(f"{REFETCH_STEM}.*"):
            leftover.unlink()
    options = {
        "format": cfg.fetch.format,
        "merge_output_format": "mp4",
        "outtmpl": str(service.sources / f"{stem}.%(ext)s"),
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook()],
        "logger": log,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([opts.url])
    except yt_dlp.utils.DownloadError as error:
        raise StageError(str(error)) from error
    if refetch:
        publish_refetch(service)
    rename_info_json(service)
    return {"ok": True, "video": str(service.youtube_path), "up_to_date": False}


def publish_refetch(service: Service) -> None:
    video = service.sources / f"{REFETCH_STEM}.mp4"
    if not video.exists():
        raise StageError(f"yt-dlp finished but {video.name} is missing; the previous download was kept")
    os.replace(video, service.youtube_path)
    info = service.sources / f"{REFETCH_STEM}.info.json"
    if info.exists():
        os.replace(info, service.sources / "youtube.info.json")
    log.info("replaced sources/youtube.mp4 with the new download")


def progress_hook():
    state = {"decile": 0}

    def hook(status):
        if status["status"] == "finished":
            log.info("download finished")
            state["decile"] = 0
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if status["status"] != "downloading" or not total:
            return
        decile = int((status.get("downloaded_bytes") or 0) * 10 // total) * 10
        if decile > state["decile"]:
            state["decile"] = decile
            log.info("download %d%%", decile)

    return hook


def rename_info_json(service: Service) -> None:
    target = service.sources / "youtube.info.json"
    candidates = sorted(service.sources.glob("youtube*.info.json"))
    if candidates and not target.exists():
        candidates[0].rename(target)
        log.info("renamed %s to youtube.info.json", candidates[0].name)
