import json
import logging
import xml.etree.ElementTree as ET

import pytest

from reels import config
from reels.service import Service, StageError, write_json
from reels.stages import render
from tests.timeline_fixtures import MOMENTS, SYNC, write_moments, write_sync


def args(root, mode="fcpxml", force=False):
    return render.Options(service=str(root), mode=mode, force=force)




def test_run_writes_and_skips_when_up_to_date(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    service = Service(tmp_path)
    write_sync(service)
    write_moments(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {m["slug"]: {} for m in MOMENTS}})
    cfg = config.Config()
    assert render.run(args(tmp_path), cfg)["ok"]
    out = service.out / "timeline.fcpxml"
    assert (service.work / "render.json").exists()
    assert not (service.out / "timeline.json").exists()
    root = ET.fromstring(out.read_text().split("<!DOCTYPE fcpxml>\n")[1])
    assert len(list(root.iter("project"))) == 3
    assert "drift error" in caplog.text
    caplog.clear()
    render.run(args(tmp_path), cfg)
    assert "up to date" in caplog.text


def test_run_without_moments_warns(tmp_path, caplog):
    service = Service(tmp_path)
    write_sync(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {}})
    render.run(args(tmp_path), config.Config())
    assert "no moments" in caplog.text
    assert (service.out / "timeline.fcpxml").exists()


def test_run_errors(tmp_path):
    with pytest.raises(StageError, match="mode mp4 is not built yet"):
        render.run(args(tmp_path, mode="mp4"), config.Config())
    with pytest.raises(StageError, match="run reels sync first"):
        render.run(args(tmp_path), config.Config())


def test_render_requires_beats(tmp_path):
    service = Service(tmp_path)
    write_sync(service)
    with pytest.raises(StageError, match="run reels cut first"):
        render.run(args(tmp_path), config.Config())


def test_render_skips_moments_missing_from_beats(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    service = Service(tmp_path)
    write_sync(service)
    write_moments(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {MOMENTS[0]["slug"]: {}}})
    assert render.run(args(tmp_path), config.Config())["ok"]
    root = ET.fromstring((service.out / "timeline.fcpxml").read_text().split("<!DOCTYPE fcpxml>\n")[1])
    assert [p.get("name") for p in root.iter("project")][1:] == [MOMENTS[0]["slug"]]
    assert "cuts/1-30-00_solo not in work/beats.json" in caplog.text


def test_render_requires_source_media_and_reruns_when_it_changes(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    service = Service(tmp_path)
    write_sync(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {}})
    assert render.run(args(tmp_path), config.Config())["ok"]
    (service.root / "work/dji.joined.mp4").write_bytes(b"different media")
    caplog.clear()
    assert render.run(args(tmp_path), config.Config())["ok"]
    assert "up to date" not in caplog.text
    (service.root / "sources/youtube.mp4").unlink()
    with pytest.raises(StageError, match="youtube.mp4"):
        render.run(args(tmp_path), config.Config())


def test_render_refuses_moments_cut_against_an_older_sync(tmp_path):
    service = Service(tmp_path)
    write_sync(service)
    write_moments(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {m["slug"]: {} for m in MOMENTS}})
    edited = {**SYNC, "sources": {**SYNC["sources"], "dji": {**SYNC["sources"]["dji"], "intercept_s": -50.0}}}
    write_json(service.work / "sync.json", edited)
    with pytest.raises(StageError, match=r"cuts/1-00-10_fill was cut against an older work/sync.json"):
        render.run(args(tmp_path), config.Config())


def test_render_tolerates_a_failed_camera_without_media(tmp_path):
    service = Service(tmp_path)
    sync = {**SYNC, "sources": {**SYNC["sources"], "x5": {"chunks": ["sources/x5.mp4"], "failed": True, "error": "x"}}}
    write_sync(service, sync={**sync, "sources": {"dji": sync["sources"]["dji"]}})
    write_json(service.work / "sync.json", sync)
    write_json(service.work / "beats.json", {"version": 1, "moments": {}})
    assert render.run(args(tmp_path), config.Config())["ok"]






def test_render_rejects_a_sidecar_path_outside_the_service(tmp_path):
    service = Service(tmp_path)
    write_sync(service)
    escaped = {**SYNC, "sources": {"dji": {**SYNC["sources"]["dji"], "path": "../outside.mp4"}}}
    write_json(service.work / "sync.json", escaped)
    write_json(service.work / "beats.json", {"version": 1, "moments": {}})
    with pytest.raises(StageError, match="outside the service folder"):
        render.run(args(tmp_path), config.Config())


def test_render_rejects_a_malformed_moment_and_an_interrupted_cut(tmp_path):
    service = Service(tmp_path)
    write_sync(service)
    write_moments(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {m["slug"]: {} for m in MOMENTS}})
    broken = service.cuts / MOMENTS[0]["slug"] / "moment.json"
    data = json.loads(broken.read_text())
    write_json(broken, {**data, "inputs": None})
    with pytest.raises(StageError, match="last cut was interrupted"):
        render.run(args(tmp_path), config.Config())
    del data["files"]
    write_json(broken, data)
    with pytest.raises(StageError, match="is missing files"):
        render.run(args(tmp_path), config.Config())


def test_render_fingerprint_includes_resolved_root(tmp_path):
    service = Service(tmp_path)
    write_sync(service)
    write_json(service.work / "beats.json", {"version": 1, "moments": {}})
    render.run(args(tmp_path), config.Config())
    sidecar = json.loads((service.work / "render.json").read_text())
    assert sidecar["inputs"]["params"]["root"] == str(tmp_path.resolve())
