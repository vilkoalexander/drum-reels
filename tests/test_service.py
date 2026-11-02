import pytest

from reels.service import Service, StageError, read_json, up_to_date, write_json


def make_service(tmp_path):
    service = Service(tmp_path)
    service.sources.mkdir()
    return service


def test_camera_files_case_insensitive_sorted(tmp_path):
    service = make_service(tmp_path)
    (service.sources / "DJI_0002.mp4").touch()
    (service.sources / "dji_0001.mp4").touch()
    (service.sources / "youtube.mp4").touch()
    cams = service.camera_files()
    assert [p.name for p in cams["dji"]] == ["dji_0001.mp4", "DJI_0002.mp4"]
    assert cams["x5"] == []


def test_ensure_creates_working_folders(tmp_path):
    service = Service(tmp_path)
    service.ensure()
    assert service.audio.is_dir() and service.cuts.is_dir() and service.out.is_dir()


def test_fingerprint_stable_then_changes_with_size_or_params(tmp_path):
    service = make_service(tmp_path)
    video = service.youtube_path
    video.write_bytes(b"abc")
    before = service.fingerprint([video], {"a": 1})
    assert before == service.fingerprint([video], {"a": 1})
    assert list(before["files"]) == ["sources/youtube.mp4"]
    assert before != service.fingerprint([video], {"a": 2})
    video.write_bytes(b"abcdef")
    assert before != service.fingerprint([video], {"a": 1})


def test_json_roundtrip_is_sorted_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": 2})
    assert read_json(path) == {"a": 2, "b": 1}
    assert path.read_text().index('"a"') < path.read_text().index('"b"')
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
    assert read_json(tmp_path / "none.json") is None


def test_up_to_date_requires_matching_inputs_and_existing_outputs(tmp_path):
    out = tmp_path / "out.txt"
    fp = {"files": {}, "params": {}}
    sidecar = {"inputs": fp}
    assert not up_to_date(sidecar, fp, [out])
    out.touch()
    assert up_to_date(sidecar, fp, [out])
    assert not up_to_date(sidecar, {"files": {}, "params": {"x": 1}}, [out])
    assert not up_to_date(None, fp, [out])


def test_read_json_rejects_non_objects_and_garbage(tmp_path):
    (tmp_path / "list.json").write_text("[]")
    with pytest.raises(StageError, match="JSON object"):
        read_json(tmp_path / "list.json")
    (tmp_path / "bad.json").write_text("{nope")
    with pytest.raises(StageError, match="not valid JSON"):
        read_json(tmp_path / "bad.json")


def test_path_keeps_symlinks_and_rejects_escapes(tmp_path):
    service = make_service(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside.write_bytes(b"x")
    (service.sources / "dji.mp4").symlink_to(outside)
    assert service.path("sources/dji.mp4").read_bytes() == b"x"
    for bad in ("../x.mp4", "/etc/passwd", "sources/../../x", "", 3, None):
        with pytest.raises(StageError, match="outside the service folder"):
            service.path(bad)
