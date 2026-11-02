import pytest

from reels.analysis import people
from reels.analysis.people import Detection, Person


def seen(yaw, frames, height=20.0):
    return [Detection(f, yaw, -3.0, height) for f in range(frames)]


def test_detections_map_pixels_to_angles_and_wrap_the_seam():
    found = people.detections(0, [(1940, 840, 1980, 960), (3900, 900, 3940, 1020)], 3840, 1920)
    assert found[0].yaw == pytest.approx(3.75)
    assert found[0].pitch == pytest.approx(5.625)
    assert found[0].height == pytest.approx(11.25)
    assert found[1].yaw == pytest.approx(-172.5)


def test_people_groups_by_yaw_and_drops_people_passing_through():
    found = seen(-83, 5) + seen(-80, 5) + seen(92, 4) + seen(10, 1)
    result = people.people(found, 5)
    assert [round(p.yaw) for p in result] == [-82, 92]
    assert [p.share for p in result] == [1.0, 0.8]


def test_people_joins_one_person_split_across_the_seam():
    found = seen(178, 3) + seen(-178, 3)
    result = people.people(found, 3)
    assert len(result) == 1
    assert abs(result[0].yaw) == pytest.approx(180, abs=0.5)


def test_fov_follows_person_size_within_limits():
    near, far, tiny = (people.people(seen(0, 2, h), 2)[0].fov for h in (40.0, 20.0, 1.0))
    assert near > far
    assert tiny == people.MIN_FOV


def test_wide_holds_everyone_with_margin_even_across_the_seam():
    assert people.wide([]) == (0.0, 90.0)
    yaw, fov = people.wide([Person(170, 0, 30, 1), Person(-170, 0, 30, 1)])
    assert abs(yaw) == pytest.approx(180)
    assert fov == pytest.approx(20 + 2 * people.WIDE_MARGIN_DEG)
    yaw, fov = people.wide([Person(-15, 0, 30, 1), Person(15, 0, 30, 1)])
    assert (yaw, fov) == (pytest.approx(0), pytest.approx(30 + 2 * people.WIDE_MARGIN_DEG))
    assert people.wide([Person(-90, 0, 30, 1), Person(90, 0, 30, 1)])[1] == people.MAX_FOV


def test_best_rotation_prefers_highest_score_then_no_rotation():
    assert people.best_rotation({0: 0.4, 90: 2.5, 180: 0.0, 270: 0.3}) == 90
    assert people.best_rotation({0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}) == 0


def test_default_pattern_alternates_flat_cameras_with_views():
    assert people.default_pattern(["dji"], ["person1", "person2", "wide"]) == [
        "dji", "person1", "dji", "person2", "dji", "wide", "master"]
    assert people.default_pattern([], ["person1", "wide"]) == ["person1", "wide", "master"]
    assert people.default_pattern(["dji"], []) == ["dji", "master"]
