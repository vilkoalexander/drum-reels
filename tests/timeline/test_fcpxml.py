import xml.etree.ElementTree as ET

import pytest

from reels import config
from reels.service import Service, StageError
from reels.timeline import fcpxml

SYNC = {
    "youtube": {"path": "sources/youtube.mp4", "fps": 30.0, "width": 1920, "height": 1080, "duration_s": 7200.0},
    "sources": {
        "dji": {"chunks": ["sources/dji.mp4"], "path": "work/dji.joined.mp4", "fps": 59.94, "width": 3840,
                "height": 2160, "duration_s": 6900.0, "intercept_s": 37.3, "slope": 1.00008},
        "x5": {"chunks": ["sources/x5.mp4"], "path": "sources/x5.mp4", "fps": 30.0, "width": 1920, "height": 1080,
               "duration_s": 7000.0, "intercept_s": -12.1, "slope": 1.0},
    },
}


def moment(slug, label, start_s, end_s, skipped=None):
    files = {"youtube": "youtube.mov", "dji": "dji.mov", "x5": "x5.mov"}
    for cam in skipped or {}:
        del files[cam]
    return {"slug": slug, "label": label, "start_s": start_s, "end_s": end_s, "handle_s": 2.0,
            "files": files, "skipped": skipped or {}}


MOMENTS = [moment("1-00-10_fill", "fill", 610.0, 640.0), moment("1-30-00_solo", "solo", 5400.0, 5430.0, {"x5": "late"})]


def build(tmp_path, cfg=None, sync=SYNC):
    return fcpxml.build(Service(tmp_path / "2026-01-04"), sync, MOMENTS, cfg or config.FcpxmlCfg())


def parse(tmp_path, **kw):
    return ET.fromstring(build(tmp_path, **kw).xml.split("<!DOCTYPE fcpxml>\n")[1])


def project(root, name):
    return next(p for p in root.iter("project") if p.get("name") == name)


def test_rational_cases():
    assert fcpxml.rational(1.0, 29.97) == "30030/30000s"
    assert fcpxml.rational(0.5, 30) == "15/30s"
    assert fcpxml.rational(1 / 59.94, 59.94) == "1001/60000s"
    assert fcpxml.rational(0, 30) == "0s"
    assert fcpxml.rational(1.0, 119.88) == "120120/120000s"
    for fps in (23.976, 29.97, 59.94, 119.88):
        assert int(fcpxml.rational(37.588, fps).split("/")[0]) % 1001 == 0


def test_timebase_only_treats_true_ntsc_rates_as_fractional():
    assert fcpxml.frame_duration(10) == "1/10s"
    assert fcpxml.frame_duration(5) == "1/5s"
    assert fcpxml.frame_duration(24000 / 1001) == "1001/24000s"
    with pytest.raises(ValueError):
        fcpxml.frame_duration(0)


def test_frame_duration_cases():
    assert fcpxml.frame_duration(30) == "1/30s"
    assert fcpxml.frame_duration(29.97) == "1001/30000s"
    assert fcpxml.frame_duration(59.94) == "1001/60000s"
    assert fcpxml.frame_duration(119.88) == "1001/120000s"
    assert fcpxml.frame_duration(23.976) == "1001/24000s"
    assert fcpxml.frame_duration(25) == "1/25s"
    assert fcpxml.frame_duration(24) == "1/24s"


def test_document_header(tmp_path):
    xml = build(tmp_path).xml
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.10">')


def test_resources_one_format_per_distinct_shape(tmp_path):
    root = parse(tmp_path)
    shapes = sorted((f.get("width"), f.get("frameDuration")) for f in root.iter("format"))
    assert shapes == [("1920", "1/30s"), ("3840", "1001/60000s")]
    audio = {a.get("name"): a.get("hasAudio") for a in root.iter("asset")}
    assert audio == {"youtube": "1", "dji": "0", "x5": "0"}
    assert next(root.iter("media-rep")).get("src").startswith("file:///")


def test_service_project_layout(tmp_path):
    root = parse(tmp_path)
    clip = project(root, "2026-01-04").find("sequence/spine/asset-clip")
    assert clip.get("name") == "youtube" and clip.get("offset") == "0s"
    assert {c.get("lane") for c in clip.findall("asset-clip")} == {"1", "2"}
    assert [m.get("value") for m in clip.findall("marker")] == ["fill", "solo"]
    dji = next(c for c in clip.findall("asset-clip") if c.get("lane") == "1")
    assert dji.get("offset") == "0s"
    assert dji.get("start") == "2254252/60000s"
    x5 = next(c for c in clip.findall("asset-clip") if c.get("lane") == "2")
    assert x5.get("offset") == fcpxml.rational(12.1, 30)
    assert x5.get("start") == "0s"


def test_service_camera_clamped_to_youtube_end(tmp_path):
    long = {**SYNC, "sources": {"x5": {**SYNC["sources"]["x5"], "duration_s": 9000.0}}}
    clip = parse(tmp_path, sync=long).find(".//project/sequence/spine/asset-clip/asset-clip")
    assert clip.get("duration") == fcpxml.rational(7200 - 12.1, 30)


def test_moment_projects(tmp_path):
    root = parse(tmp_path)
    fill = project(root, "1-00-10_fill").find("sequence/spine/asset-clip")
    assert fill.get("start") == fcpxml.rational(608.0, 30)
    assert fill.get("duration") == fcpxml.rational(34.0, 30)
    dji = next(c for c in fill.findall("asset-clip") if c.get("lane") == "1")
    assert dji.get("start") == fcpxml.rational(37.3 + 1.00008 * (610.0 - 2.0), 59.94)
    assert dji.get("duration") == fcpxml.rational(34.0, 30)
    assert dji.get("offset") == fcpxml.rational(608.0, 30)
    x5 = next(c for c in fill.findall("asset-clip") if c.get("lane") == "2")
    assert x5.get("offset") == fcpxml.rational(608.0, 30)
    assert [(m.get("value"), m.get("start")) for m in fill.findall("marker")] == [("fill", fcpxml.rational(610.0, 30))]
    solo = project(root, "1-30-00_solo")
    assert [c.get("lane") for c in solo.iter("asset-clip") if c.get("lane")] == ["1"]


def test_moment_window_clamped_to_stream_start(tmp_path):
    early = [moment("0-00-01_hit", "hit", 1.0, 20.0)]
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), SYNC, early, config.FcpxmlCfg(per_service=False))
                         .xml.split("<!DOCTYPE fcpxml>\n")[1])
    clip = root.find(".//project/sequence/spine/asset-clip")
    assert clip.get("start") == "0s"
    assert all(not c.get("offset").startswith("-") for c in clip.findall("asset-clip"))


def test_no_negative_offsets(tmp_path):
    for clip in parse(tmp_path).iter("asset-clip"):
        assert not clip.get("offset").startswith("-")


def test_notes_one_per_camera(tmp_path):
    notes = build(tmp_path).notes
    assert len(notes) == 2
    assert notes[0] == "dji: placed with offset at midpoint; drift error ±16.5 frames at the ends"
    assert notes[1].startswith("x5: ") and "±0.0 frames" in notes[1]


def test_camera_without_overlap_is_left_out_with_a_note(tmp_path):
    sync = {**SYNC, "youtube": {**SYNC["youtube"], "duration_s": 60.0},
            "sources": {"x5": {**SYNC["sources"]["x5"], "duration_s": 30.0, "intercept_s": 70.0}}}
    timeline = fcpxml.build(Service(tmp_path / "s"), sync, [], config.FcpxmlCfg())
    root = ET.fromstring(timeline.xml.split("<!DOCTYPE fcpxml>\n")[1])
    assert [c.get("name") for c in root.iter("asset-clip")] == ["youtube"]
    assert timeline.notes == ["x5: no usable overlap with the stream, left out of the service project"]
    for clip in root.iter("asset-clip"):
        assert not clip.get("duration").startswith("-")


def test_camera_clip_never_runs_past_its_asset(tmp_path):
    sync = {**SYNC, "youtube": {**SYNC["youtube"], "duration_s": 100.0},
            "sources": {"x5": {**SYNC["sources"]["x5"], "duration_s": 1.01, "intercept_s": -0.017}}}
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, [], config.FcpxmlCfg()).xml
                         .split("<!DOCTYPE fcpxml>\n")[1])
    asset = next(a for a in root.iter("asset") if a.get("name") == "x5")
    clip = next(c for c in root.iter("asset-clip") if c.get("name") == "x5")
    def numerator(text):
        return 0 if text == "0s" else int(text.split("/")[0])

    assert numerator(clip.get("start")) + numerator(clip.get("duration")) <= numerator(asset.get("duration"))


def test_manual_entry_without_slope_is_a_stage_error(tmp_path):
    sync = {**SYNC, "sources": {"dji": {k: v for k, v in SYNC["sources"]["dji"].items() if k != "slope"}}}
    with pytest.raises(StageError, match="dji.*slope"):
        build(tmp_path, sync=sync)


def test_failed_camera_left_out(tmp_path):
    sync = {**SYNC, "sources": {**SYNC["sources"], "x5": {**SYNC["sources"]["x5"], "failed": "weak"}}}
    root = parse(tmp_path, sync=sync)
    assert {c.get("name") for c in root.iter("asset-clip")} == {"youtube", "dji"}


def test_toggles(tmp_path):
    root = parse(tmp_path, cfg=config.FcpxmlCfg(per_service=False))
    assert [p.get("name") for p in root.iter("project")] == ["1-00-10_fill", "1-30-00_solo"]
    root = parse(tmp_path, cfg=config.FcpxmlCfg(per_moment=False))
    assert [p.get("name") for p in root.iter("project")] == ["2026-01-04"]
    with pytest.raises(ValueError):
        build(tmp_path, cfg=config.FcpxmlCfg(False, False))


def test_moment_camera_start_uses_frame_rounded_window(tmp_path):
    sync = {**SYNC, "sources": {"dji": {**SYNC["sources"]["dji"], "intercept_s": 0.0, "slope": 1.0}}}
    off_grid = [moment("0-10-00_hit", "hit", 600.0 + 0.5 / 30 + 2.0, 640.0)]
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, off_grid, config.FcpxmlCfg(per_service=False))
                         .xml.split("<!DOCTYPE fcpxml>\n")[1])
    clip = root.find(".//project/sequence/spine/asset-clip")
    snapped = fcpxml.snap_to_frame(600.0 + 0.5 / 30 + 0.0, 30.0)
    dji = clip.find("asset-clip")
    assert clip.get("start") == fcpxml.rational(snapped, 30)
    assert dji.get("offset") == clip.get("start")
    assert dji.get("start") == fcpxml.rational(snapped, 59.94)

def test_camera_start_before_its_asset_is_trimmed_not_negative(tmp_path):
    sync = {**SYNC, "youtube": {**SYNC["youtube"], "fps": 25.0},
            "sources": {"dji": {**SYNC["sources"]["dji"], "fps": 60.0, "intercept_s": -10.019, "slope": 1.0}}}
    early = [moment("0-00-12_hit", "hit", 12.019, 30.0)]
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, early, config.FcpxmlCfg(per_service=False))
                         .xml.split("<!DOCTYPE fcpxml>\n")[1])
    dji = root.find(".//project/sequence/spine/asset-clip/asset-clip")
    assert dji.get("start") == "0s"
    assert not dji.get("offset").startswith("-") and not dji.get("duration").startswith("-")

def test_asset_duration_is_floored_to_whole_frames(tmp_path):
    sync = {**SYNC, "sources": {"x5": {**SYNC["sources"]["x5"], "duration_s": 100.02}}}
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, [], config.FcpxmlCfg()).xml
                         .split("<!DOCTYPE fcpxml>\n")[1])
    asset = next(a for a in root.iter("asset") if a.get("name") == "x5")
    assert asset.get("duration") == "3000/30s"



def test_spine_clip_never_runs_past_the_floored_youtube_asset(tmp_path):
    sync = {**SYNC, "youtube": {**SYNC["youtube"], "duration_s": 100.02}, "sources": {}}
    late = [moment("0-01-33_tail", "tail", 93.0 + 2.0, 100.02)]
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, late, config.FcpxmlCfg()).xml
                         .split("<!DOCTYPE fcpxml>\n")[1])
    asset = next(a for a in root.iter("asset") if a.get("name") == "youtube")
    assert asset.get("duration") == "3000/30s"
    for clip in root.iter("asset-clip"):
        start = 0 if clip.get("start") == "0s" else int(clip.get("start").split("/")[0])
        assert start + int(clip.get("duration").split("/")[0]) <= 3000
    for seq in root.iter("sequence"):
        assert int(seq.get("duration").split("/")[0]) <= 3000

def test_mixed_frame_rate_camera_clip_ends_inside_its_asset(tmp_path):
    sync = {**SYNC, "youtube": {**SYNC["youtube"], "fps": 25.0, "duration_s": 100.0},
            "sources": {"dji": {**SYNC["sources"]["dji"], "fps": 59.94, "duration_s": 30.03, "intercept_s": -10.0,
                                "slope": 1.0}}}
    root = ET.fromstring(fcpxml.build(Service(tmp_path / "s"), sync, [], config.FcpxmlCfg()).xml
                         .split("<!DOCTYPE fcpxml>\n")[1])
    asset = next(a for a in root.iter("asset") if a.get("name") == "dji")
    clip = next(c for c in root.iter("asset-clip") if c.get("name") == "dji")
    asset_s = int(asset.get("duration").split("/")[0]) / 60000
    start_s = 0 if clip.get("start") == "0s" else int(clip.get("start").split("/")[0]) / 60000
    assert clip.get("duration").endswith("/25s")
    assert start_s + int(clip.get("duration").split("/")[0]) / 25 <= asset_s + 1e-9
