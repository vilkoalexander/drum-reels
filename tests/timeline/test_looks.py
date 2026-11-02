import pytest

from reels.service import StageError
from reels.timeline import looks
from reels.timeline.looks import MASTER, Look

TEXT = """
[look.top]
camera = "dji"
rotate = 90

[look.drums]
camera = "x5"
yaw = -83
pitch = -3
fov = 36

[pattern]
default = ["top", "drums", "master"]
short = ["drums"]
"""


def test_parse_reads_looks_patterns_and_adds_master():
    found, patterns = looks.parse(TEXT, "looks.toml")
    assert found == {"master": MASTER, "top": Look("dji", rotate=90),
                     "drums": Look("x5", yaw=-83.0, pitch=-3.0, fov=36.0, move=True)}
    assert patterns == {"default": ["top", "drums", "master"], "short": ["drums"]}


def test_to_toml_round_trips():
    found, patterns = looks.parse(TEXT, "looks.toml")
    assert looks.parse(looks.to_toml(found, patterns, ["hi"]), "again") == (found, patterns)


@pytest.mark.parametrize("bad, message", [
    (TEXT.replace('"top", "drums"', '"top", "keys"'), "unknown look 'keys'"),
    (TEXT.replace("rotate = 90", "rotate = 45"), "rotate must be one of"),
    (TEXT.replace("fov = 36", "fov = 0"), "fov between 0 and 180"),
    (TEXT.replace("[look.top]", "[look.Top]"), "lower case"),
    (TEXT.replace("pitch = -3", "tilt = -3"), "unknown key tilt"),
    (TEXT.split("[pattern]")[0], "no \\[pattern\\]"),
    ("[look.x\n", "looks.toml"),
])
def test_parse_rejects_bad_files_naming_the_problem(bad, message):
    with pytest.raises(StageError, match=message):
        looks.parse(bad, "looks.toml")


def test_flat_look_rotates_then_crops_to_9_16():
    vf = looks.look_filter(Look("dji", rotate=90))
    assert vf.startswith("transpose=1,crop=")
    assert looks.look_filter(MASTER).startswith("crop=")


def test_view_wraps_yaw_and_keeps_pixels_square():
    vf = looks.look_filter(Look("x5", yaw=200, fov=36))
    assert "yaw=-160" in vf
    # flat projection: tan(v/2) / tan(h/2) must equal HEIGHT / WIDTH
    assert "v_fov=60.02:" in vf
    assert "v_fov=102.45:" in looks.look_filter(Look("x5", yaw=0, fov=70))


def test_look_filter_appends_grade_and_rejects_unknown():
    assert looks.look_filter(MASTER, "film").endswith(looks.GRADES["film"])
    with pytest.raises(StageError, match="unknown grade"):
        looks.look_filter(MASTER, "sepia")


def test_h_fov_inverts_v_fov():
    assert looks.h_fov(looks.v_fov(40.0)) == pytest.approx(40.0)
