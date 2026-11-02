from pathlib import Path

import pytest

from reels import config


def test_defaults_load_without_path():
    cfg = config.load()
    assert cfg.sync.sample_rate == 16000
    assert cfg.cut.people[0] == "guitar"
    assert cfg.fcpxml.per_moment is True


def test_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.toml"):
        config.load(tmp_path / "missing.toml")


def test_default_path_is_the_repo_toml():
    assert config.DEFAULT_PATH == Path(__file__).resolve().parents[1] / "reels.toml"


def test_repo_toml_matches_dataclass_defaults():
    assert config.load() == config.Config()


def test_override_changes_one_value_and_keeps_the_rest(tmp_path):
    path = tmp_path / "reels.toml"
    path.write_text("[sync]\nmin_confidence = 6.5\n")
    cfg = config.load(path)
    assert cfg.sync.min_confidence == 6.5
    assert cfg.sync.warn_confidence == config.Config().sync.warn_confidence
    assert cfg.cut == config.Config().cut


def test_list_override_becomes_tuple(tmp_path):
    path = tmp_path / "reels.toml"
    path.write_text('[cut]\npeople = ["drums"]\n')
    assert config.load(path).cut.people == ("drums",)


def test_unknown_key_names_section_and_key(tmp_path):
    path = tmp_path / "reels.toml"
    path.write_text("[sync]\nbogus = 1\n")
    with pytest.raises(ValueError, match=r"sync\.bogus"):
        config.load(path)


def test_unknown_section_raises(tmp_path):
    path = tmp_path / "reels.toml"
    path.write_text("[nope]\na = 1\n")
    with pytest.raises(ValueError, match="nope"):
        config.load(path)


@pytest.mark.parametrize("text, match", [
    ("sync = 3\n", "section sync"),
    ('[cut]\npeople = "guitar"\n', r"cut\.people must be a list of strings"),
    ('[cut]\nhandle_s = "2"\n', r"cut\.handle_s must be a float"),
    ("[sync]\nfine_every_s = 0\n", r"sync\.fine_every_s must be positive"),
    ("[cut]\nsnap_to_bars = 1\n", r"cut\.snap_to_bars must be a bool"),
    ('[cut]\ncodec = "vp9"\n', "cut.codec must be one of"),
    ("[sync]\nfine_window_s = inf\n", r"sync\.fine_window_s must be a finite number"),
    ("[sync]\nmax_drift_ppm = nan\n", r"sync\.max_drift_ppm must be a finite number"),
    ("[cut]\nhandle_s = -1\n", r"cut\.handle_s must not be negative"),
    ("[sync]\nmax_window_residual_ms = 0\n", r"sync\.max_window_residual_ms must be positive"),
    ("[cut]\ndefault_len_s = 3\n", r"cut\.default_len_s \(3\) must be at least cut\.min_len_s \(5\)"),
])
def test_wrong_value_types_are_rejected(tmp_path, text, match):
    path = tmp_path / "reels.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        config.load(path)


def test_int_is_accepted_for_a_float_key(tmp_path):
    path = tmp_path / "reels.toml"
    path.write_text("[cut]\nhandle_s = 3\n")
    assert config.load(path).cut.handle_s == 3


@pytest.mark.parametrize("where", sorted(config.NON_NEGATIVE))
def test_negative_values_are_rejected_for_every_non_negative_key(tmp_path, where):
    section, key = where.split(".")
    path = tmp_path / "reels.toml"
    path.write_text(f"[{section}]\n{key} = -1\n")
    with pytest.raises(ValueError, match="must not be negative"):
        config.load(path)


@pytest.mark.parametrize("where", sorted(config.POSITIVE))
def test_zero_is_rejected_for_every_positive_key(tmp_path, where):
    section, key = where.split(".")
    path = tmp_path / "reels.toml"
    path.write_text(f"[{section}]\n{key} = 0\n")
    with pytest.raises(ValueError, match="must be positive"):
        config.load(path)
