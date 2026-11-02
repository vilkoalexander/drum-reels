import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from reels.analysis.clock import ClockMap, clock_map
from reels.config import FcpxmlCfg
from reels.service import CAMERAS, Service, StageError

LANES = {cam: lane for lane, cam in enumerate(CAMERAS, 1)}
NTSC_TOLERANCE = 0.001
FRAME_EPSILON_S = 1e-6


@dataclass(frozen=True)
class Timeline:
    xml: str
    notes: list[str]


def _timebase(fps: float) -> tuple[int, bool]:
    if not fps > 0:
        raise ValueError(f"frame rate must be positive, got {fps}")
    scaled = fps * 1.001
    if abs(scaled - round(scaled)) <= NTSC_TOLERANCE:
        return round(scaled) * 1000, True
    return round(fps), False


def _unit(fps: float) -> float:
    base, fractional = _timebase(fps)
    return 1001 / base if fractional else 1 / base


def frames(seconds: float, fps: float) -> int:
    return round(seconds / _unit(fps))


def frames_floor(seconds: float, fps: float) -> int:
    return math.floor((seconds + FRAME_EPSILON_S) / _unit(fps))


def snap_to_frame(seconds: float, fps: float) -> float:
    return frames(seconds, fps) * _unit(fps)


def rational_frames(n: int, fps: float) -> str:
    base, fractional = _timebase(fps)
    if n == 0:
        return "0s"
    return f"{n * 1001}/{base}s" if fractional else f"{n}/{base}s"


def rational(seconds: float, fps: float) -> str:
    return rational_frames(frames(seconds, fps), fps)


def frame_duration(fps: float) -> str:
    return rational_frames(1, fps)


def _fps_name(fps: float) -> str:
    _, fractional = _timebase(fps)
    return f"{round(fps * 100)}" if fractional else f"{fps:g}"


class _Resources:
    def __init__(self, service: Service, parent: ET.Element):
        self.service = service
        self.parent = parent
        self.formats: dict[tuple, str] = {}
        self.assets: dict[str, str] = {}
        self.count = 0

    def _next(self) -> str:
        self.count += 1
        return f"r{self.count}"

    def format(self, fps: float, width: int, height: int) -> str:
        key = (fps, width, height)
        if key not in self.formats:
            ident = self._next()
            self.formats[key] = ident
            ET.SubElement(self.parent, "format", id=ident, name=f"FFVideoFormat{height}p{_fps_name(fps)}",
                          frameDuration=frame_duration(fps), width=str(width), height=str(height))
        return self.formats[key]

    def asset(self, name: str, info: dict, audio: bool) -> str:
        if name not in self.assets:
            fmt = self.format(info["fps"], info["width"], info["height"])
            ident = self._next()
            self.assets[name] = ident
            asset = ET.SubElement(self.parent, "asset", id=ident, name=name, start="0s",
                                  duration=rational_frames(frames_floor(info["duration_s"], info["fps"]), info["fps"]),
                                  hasVideo="1", hasAudio="1" if audio else "0", format=fmt)
            src = self.service.path(info["path"]).resolve().as_uri()
            ET.SubElement(asset, "media-rep", kind="original-media", src=src)
        return self.assets[name]


def _camera_clip(res: _Resources, cam: str, info: dict, offset_s: float, start_s: float, duration_s: float,
                 yt_fps: float) -> ET.Element | None:
    fps = info["fps"]
    if start_s < 0:
        offset_s, duration_s, start_s = offset_s - start_s, duration_s + start_s, 0.0
    start = frames(start_s, fps)
    room_s = frames_floor(info["duration_s"], fps) * _unit(fps) - start * _unit(fps)
    length = min(frames(duration_s, yt_fps), frames_floor(room_s, yt_fps))
    if length <= 0:
        return None
    return ET.Element("asset-clip", ref=res.asset(cam, info, False), lane=str(LANES[cam]),
                      offset=rational(offset_s, yt_fps), name=cam, start=rational_frames(start, fps),
                      duration=rational_frames(length, yt_fps),
                      format=res.format(fps, info["width"], info["height"]))


def _project(event: ET.Element, name: str, res: _Resources, yt_info: dict, start_s: float,
             duration_s: float) -> ET.Element:
    fps = yt_info["fps"]
    start = frames(start_s, fps)
    length = min(frames(duration_s, fps), frames_floor(yt_info["duration_s"], fps) - start)
    if length <= 0:
        raise StageError(f"project {name} has an empty window ({duration_s:.3f} s)")
    fmt = res.format(fps, yt_info["width"], yt_info["height"])
    project = ET.SubElement(event, "project", name=name)
    sequence = ET.SubElement(project, "sequence", format=fmt, duration=rational_frames(length, fps),
                             tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")
    return ET.SubElement(spine, "asset-clip", ref=res.asset("youtube", yt_info, True), offset="0s",
                         name="youtube", start=rational_frames(start, fps), duration=rational_frames(length, fps),
                         format=fmt)


def _add_markers(clip: ET.Element, markers: list[tuple[float, str]], fps: float) -> None:
    for at, label in markers:
        ET.SubElement(clip, "marker", start=rational(at, fps), duration=frame_duration(fps), value=label)


def _overlap(clock: ClockMap, cam_duration_s: float, yt_duration_s: float) -> tuple[float, float]:
    t0 = max(0.0, clock.yt_time(0.0))
    t1 = min(yt_duration_s, clock.yt_time(cam_duration_s))
    return t0, t1


def _service_project(event, name, res, sync, moments, clocks) -> list[str]:
    yt = sync["youtube"]
    fps = yt["fps"]
    clip = _project(event, name, res, yt, 0.0, yt["duration_s"])
    notes = []
    for cam, info in sync["sources"].items():
        if cam not in clocks:
            continue
        clock = clocks[cam]
        t0, t1 = _overlap(clock, info["duration_s"], yt["duration_s"])
        if t1 <= t0:
            notes.append(f"{cam}: no usable overlap with the stream, left out of the service project")
            continue
        mid = (t0 + t1) / 2
        shift = clock.cam_time(mid) - mid
        if shift <= 0:
            offset, start = -shift, 0.0
        else:
            offset, start = 0.0, shift
        duration = min(info["duration_s"] - start, yt["duration_s"] - offset)
        camera = _camera_clip(res, cam, info, offset, start, duration, fps)
        if camera is None:
            notes.append(f"{cam}: no usable overlap with the stream, left out of the service project")
            continue
        clip.append(camera)
        frames_off = abs(clock.slope - 1) * max(mid - t0, t1 - mid) * info["fps"]
        notes.append(f"{cam}: placed with offset at midpoint; drift error ±{frames_off:.1f} frames at the ends")
    _add_markers(clip, [(m["start_s"], m["label"]) for m in moments], fps)
    return notes


def _moment_project(event, res, sync, moment, clocks) -> None:
    yt = sync["youtube"]
    fps = yt["fps"]
    w0 = snap_to_frame(max(0.0, moment["start_s"] - moment["handle_s"]), fps)
    w1 = min(yt["duration_s"], moment["end_s"] + moment["handle_s"])
    clip = _project(event, moment["slug"], res, yt, w0, w1 - w0)
    for cam, info in sync["sources"].items():
        if cam not in clocks or cam not in moment["files"] or cam in moment.get("skipped", {}):
            continue
        camera = _camera_clip(res, cam, info, w0, clocks[cam].cam_time(w0), w1 - w0, fps)
        if camera is not None:
            clip.append(camera)
    _add_markers(clip, [(moment["start_s"], moment["label"])], fps)


def build(service: Service, sync: dict, moments: list[dict], cfg: FcpxmlCfg) -> Timeline:
    if not (cfg.per_service or cfg.per_moment):
        raise ValueError("fcpxml.per_service and fcpxml.per_moment are both off")
    root = ET.Element("fcpxml", version="1.10")
    resources = ET.SubElement(root, "resources")
    res = _Resources(service, resources)
    event = ET.SubElement(ET.SubElement(root, "library"), "event", name=service.root.resolve().name)
    clocks = {cam: clock_map(info, cam) for cam, info in sync["sources"].items() if not info.get("failed")}
    notes: list[str] = []
    if cfg.per_service:
        notes = _service_project(event, service.root.resolve().name, res, sync, moments, clocks)
    if cfg.per_moment:
        for moment in moments:
            _moment_project(event, res, sync, moment, clocks)
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    return Timeline(f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n{xml}\n', notes)
