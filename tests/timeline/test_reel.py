import pytest

from reels.service import StageError
from reels.subtitles import Caption
from reels.timeline import reel
from reels.timeline.looks import MASTER, SRC, Look

MOMENT = {"slug": "0-34-27_keys-solo", "start_s": 100.0, "end_s": 130.0, "handle_s": 2.0, "bpm": 120.0}
ALL = {"youtube", "dji", "x5"}
LOOKS = {
    "master": MASTER,
    "top": Look("dji", rotate=90),
    "keys": Look("x5", yaw=-160, fov=58, move=True),
    "bass": Look("x5", yaw=-95, fov=34, move=True),
}
PATTERN = ["keys", "top", "keys", "bass", "top", "master"]


def test_plan_cuts_every_two_bars_without_gaps():
    plan = reel.plan(MOMENT, PATTERN, LOOKS, 0, 0, ALL)
    assert plan.frames == 900
    assert [s.frames for s in plan.shots[:3]] == [120, 120, 120]
    assert sum(s.frames for s in plan.shots) == plan.frames
    assert all(a.from_frame + a.frames == b.from_frame for a, b in zip(plan.shots, plan.shots[1:]))
    assert [s.look for s in plan.shots[:6]] == PATTERN


def test_plan_skips_and_limits_bars():
    plan = reel.plan(MOMENT, PATTERN, LOOKS, 4, 8, ALL)
    assert plan.start_s == pytest.approx(10.0)
    assert plan.length_s == pytest.approx(16.0)
    assert len(plan.shots) == 4


def test_plan_falls_back_to_master_for_a_missing_camera():
    plan = reel.plan(MOMENT, PATTERN, LOOKS, 0, 0, {"youtube", "dji"})
    assert {s.look for s in plan.shots} == {"master", "top"}


def test_plan_rejects_missing_grid_and_too_many_skipped_bars():
    with pytest.raises(StageError, match="no beat grid"):
        reel.plan({**MOMENT, "bpm": 0}, PATTERN, LOOKS, 0, 0, ALL)
    with pytest.raises(StageError, match="leaves nothing"):
        reel.plan(MOMENT, PATTERN, LOOKS, 100, 0, ALL)


def test_plan_starts_at_moment_when_handle_was_clipped_at_file_start():
    plan = reel.plan({**MOMENT, "start_s": 0.5, "end_s": 30.5}, PATTERN, LOOKS, 1, 0, ALL)
    assert plan.start_s == pytest.approx(0.5 + 2.0)


def test_props_place_stickers_on_their_look_and_alternate_moves():
    plan = reel.plan(MOMENT, PATTERN, LOOKS, 0, 8, ALL)
    props = reel.props(plan, LOOKS, "Keys", "KJMC", {"bass": "Joe Dart"})
    assert [s["move"] for s in props["shots"]] == ["in", None, "out", "in"]
    sticker = next(t for t in props["texts"] if t["kind"] == "sticker")
    bass = props["shots"][3]
    assert sticker["text"] == "Joe Dart"
    assert bass["from"] < sticker["from"] < bass["from"] + bass["duration"]
    assert {t["kind"] for t in props["texts"]} == {"sticker", "title", "caption"}


def test_single_plan_cuts_one_source_every_two_bars():
    plan = reel.single_plan(3.0, 8, 2.0)
    assert plan.frames == 480
    assert [s.look for s in plan.shots] == ["src"] * 4
    props = reel.props(plan, {"src": SRC}, None, None, {})
    assert [s["move"] for s in props["shots"]] == ["in", "out", "in", "out"]


def test_props_drop_texts_too_short_to_show():
    plan = reel.single_plan(0.0, 1, 0.5)
    props = reel.props(plan, {"src": SRC}, "Title", "Caption", {})
    assert plan.frames == 15
    assert [t["kind"] for t in props["texts"]] == ["title"]


def test_window_subtitles_shift_clip_and_drop_slivers():
    caps = [Caption(1.0, 4.0, "before"), Caption(8.0, 12.0, "first"), Caption(14.0, 16.0, "inside"),
            Caption(18.0, 25.0, "last"), Caption(30.0, 31.0, "after")]
    got = reel.window_subtitles(caps, 10.0, 10.0)
    assert [(c.text, c.start, c.end) for c in got] == [("first", 0.0, 2.0), ("inside", 4.0, 6.0), ("last", 8.0, 10.0)]
    assert reel.window_subtitles([Caption(19.9, 25.0, "sliver")], 10.0, 10.0) == []


def test_window_subtitles_show_one_at_a_time():
    got = reel.window_subtitles([Caption(0.0, 5.0, "a"), Caption(3.0, 6.0, "b")], 0.0, 10.0)
    assert [(c.text, c.start, c.end) for c in got] == [("a", 0.0, 3.0), ("b", 3.0, 6.0)]


def test_props_turn_subtitles_into_subtitle_texts():
    plan = reel.single_plan(0.0, 4, 2.0)
    props = reel.props(plan, {"src": SRC}, None, None, {}, [Caption(1.0, 2.5, "line")])
    assert props["texts"] == [{"text": "line", "from": 30, "duration": 45, "kind": "subtitle"}]
