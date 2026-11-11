import json
import logging
from pathlib import Path

import pytest

from reels.cli import main
from reels.service import Service, write_json
from reels.stages import reel


def result(capsys) -> dict:
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_no_arguments_is_usage_error(capsys):
    assert main([]) == 2
    assert result(capsys) == {"ok": False, "command": None, "error": "reels: the following arguments are required: command", "usage": True}


def test_help_lists_all_subcommands(capsys):
    assert main(["--help"]) == 0
    assert "{fetch,sync,cut,render,looks,reel,run,doctor,synth,eval,short,subs,translate}" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["fetch", "sync", "cut", "render", "looks", "reel", "run", "doctor", "synth", "eval", "short", "subs", "translate"])
def test_each_subcommand_has_help(name, capsys):
    assert main([name, "--help"]) == 0


@pytest.mark.parametrize("argv", [["sync", "{d}"], ["run", "{d}"]])
def test_nonzero_stage_code_maps_to_exit_1(argv, tmp_path, monkeypatch):
    monkeypatch.setattr("reels.stages.sync.run", lambda opts, cfg: {"ok": False, "sources": {}})
    assert main([a.format(d=tmp_path) for a in argv]) == 1


def test_render_without_sync_exits_1_with_message_on_stderr(tmp_path, capsys):
    assert main(["render", str(tmp_path), "--mode", "fcpxml"]) == 1
    assert result(capsys) == {"ok": False, "command": "render", "error": "run reels sync first"}


def test_run_stops_at_first_failing_stage(tmp_path, capsys):
    assert main(["run", str(tmp_path)]) == 1
    assert "youtube.mp4" in result(capsys)["error"]


@pytest.mark.parametrize("argv", [["render", "x"], ["fetch", "url"]])
def test_missing_required_flag_is_usage_error(argv, capsys):
    assert main(argv) == 2
    assert result(capsys)["usage"] is True


def test_run_calls_stages_in_order_forwarding_flags(tmp_path, monkeypatch, capsys):
    calls = []
    for name in ("fetch", "sync", "cut", "render"):
        monkeypatch.setattr(f"reels.stages.{name}.run",
                            lambda opts, cfg, name=name: calls.append((name, opts)) or {"ok": True, "stage": name})
    argv = ["run", str(tmp_path), "--url", "https://example.test/v", "--notes", "n.txt", "--force"]
    assert main(argv) == 0
    assert [name for name, _ in calls] == ["fetch", "sync", "cut", "render"]
    opts = dict(calls)
    assert all(o.force is True and o.service == str(tmp_path) for o in opts.values())
    assert opts["fetch"].url == "https://example.test/v"
    assert opts["cut"].notes == Path("n.txt")
    assert opts["render"].mode == "fcpxml" and opts["render"].dry_run is False
    assert list(result(capsys)["stages"]) == ["fetch", "sync", "cut", "render"]


def test_omitted_flags_take_the_options_defaults(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("reels.stages.reel.run", lambda opts, cfg: seen.append(opts) or {"ok": True})
    assert main(["reel", str(tmp_path), "fill", "--look-text", "dji=hi"]) == 0
    assert seen[0] == reel.Options(service=str(tmp_path), slug="fill", look_text=["dji=hi"])


def test_config_flag_reaches_stage(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("reels.stages.sync.run", lambda opts, cfg: seen.append(cfg) or {"ok": True})
    path = tmp_path / "custom.toml"
    path.write_text("[sync]\nmin_confidence = 7.5\n")
    assert main(["--config", str(path), "sync", str(tmp_path)]) == 0
    assert seen[0].sync.min_confidence == 7.5


def test_missing_config_file_exits_1(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.toml"), "sync", str(tmp_path)]) == 1
    assert "nope.toml" in result(capsys)["error"]


def test_verbose_sets_debug_on_root_logger(tmp_path, monkeypatch):
    monkeypatch.setattr("reels.stages.sync.run", lambda opts, cfg: {"ok": True})
    root = logging.getLogger()
    level = root.level
    try:
        assert main(["-v", "sync", str(tmp_path)]) == 0
        assert root.level == logging.DEBUG
        assert main(["sync", str(tmp_path)]) == 0
        assert root.level == logging.INFO
    finally:
        root.setLevel(level)


def test_malformed_notes_line_exits_2_with_message(tmp_path, capsys):
    service = Service(tmp_path)
    service.ensure()
    service.notes.mkdir()
    write_json(service.work / "sync.json", {
        "version": 1, "sources": {},
        "youtube": {"path": "sources/youtube.mp4", "fps": 30.0, "width": 1, "height": 1, "duration_s": 60.0},
    })
    (service.notes / "moments.txt").write_text("nonsense\n")
    assert main(["cut", str(tmp_path)]) == 2
    assert "line 1" in result(capsys)["error"]


def stage_logs(name):
    logging.getLogger(name).info("hi %s", name)
    return {"ok": True}


def test_run_sets_up_a_log_per_stage(tmp_path, monkeypatch):
    for name in ("sync", "cut", "render"):
        monkeypatch.setattr(f"reels.stages.{name}.run", lambda opts, cfg, name=name: stage_logs(name))
    assert main(["run", str(tmp_path)]) == 0
    for name in ("sync", "cut", "render"):
        text = (tmp_path / "work" / f"{name}.log").read_text()
        assert f"hi {name}" in text
        assert all(other not in text for other in {"sync", "cut", "render"} - {name})
