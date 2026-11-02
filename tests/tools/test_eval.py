import pytest

from reels import cli, config
from reels.service import Service, StageError, write_json
from reels.tools import eval as evaluate

SYNC = {"sources": {"dji": {"intercept_s": 37.302, "slope": 1.000085, "fit": "windows"}, "x5": {"failed": True, "error": "weak"}}}
TRUTH = {"dji": {"intercept_s": 37.3, "slope": 1 + 80e-6}, "x5": {"intercept_s": -12.1, "slope": 1.0}}


def test_compare_reports_errors_per_camera():
    report = evaluate.compare(SYNC, TRUTH)
    assert report["dji"]["intercept_error_ms"] == pytest.approx(2.0)
    assert report["dji"]["drift_error_ppm"] == pytest.approx(5.0)
    assert report["x5"] == {"error": "weak"}


def test_eval_command_exits_1_when_a_camera_is_out_of_tolerance(tmp_path, capsys):
    service = Service(tmp_path)
    service.ensure()
    write_json(service.work / "sync.json", SYNC)
    write_json(service.root / "truth.json", TRUTH)
    assert cli.main(["eval", str(tmp_path)]) == 1
    good = {"sources": {"dji": SYNC["sources"]["dji"], "x5": {"intercept_s": -12.1, "slope": 1.0, "fit": "windows"}}}
    write_json(service.work / "sync.json", good)
    assert cli.main(["eval", str(tmp_path), "--max-ms", "5", "--max-ppm", "10"]) == 0
    args = evaluate.Options(service=str(tmp_path), truth=None, max_ms=1.0, max_ppm=10.0)
    assert not evaluate.run(args, config.Config())["ok"]


def test_eval_needs_sync_and_truth(tmp_path):
    with pytest.raises(StageError, match="run reels sync first"):
        evaluate.run(evaluate.Options(service=str(tmp_path), truth=None, max_ms=5, max_ppm=10), config.Config())
    service = Service(tmp_path)
    service.ensure()
    write_json(service.work / "sync.json", SYNC)
    with pytest.raises(StageError, match="truth.json"):
        evaluate.run(evaluate.Options(service=str(tmp_path), truth=None, max_ms=5, max_ppm=10), config.Config())
