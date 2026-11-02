import dataclasses
import math
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path

from reels.service import CODECS


@dataclass(frozen=True)
class FetchCfg:
    format: str = "bv*[vcodec^=avc1][height<=1080]+ba[ext=m4a]/b[ext=mp4]"


@dataclass(frozen=True)
class SyncCfg:
    sample_rate: int = 16000
    envelope_hop: int = 160
    fine_window_s: float = 20
    fine_every_s: float = 600
    fine_search_s: float = 2.0
    min_confidence: float = 4.0
    warn_confidence: float = 8.0
    max_window_residual_ms: float = 50
    max_drift_ppm: float = 300


@dataclass(frozen=True)
class CutCfg:
    default_len_s: float = 30
    min_len_s: float = 5
    handle_s: float = 2
    beats_per_bar: int = 4
    snap_to_bars: bool = True
    codec: str = "h264"
    bitrate_mbps: float = 50
    people: tuple[str, ...] = ("guitar", "keys", "bass", "vocals", "vocal", "singer", "piano")


@dataclass(frozen=True)
class FcpxmlCfg:
    per_service: bool = True
    per_moment: bool = True


@dataclass(frozen=True)
class Config:
    fetch: FetchCfg = field(default_factory=FetchCfg)
    sync: SyncCfg = field(default_factory=SyncCfg)
    cut: CutCfg = field(default_factory=CutCfg)
    fcpxml: FcpxmlCfg = field(default_factory=FcpxmlCfg)


DEFAULT_PATH = Path(__file__).resolve().parents[2] / "reels.toml"
POSITIVE = {
    "sync.sample_rate", "sync.envelope_hop", "sync.fine_window_s", "sync.fine_every_s", "sync.fine_search_s",
    "sync.max_window_residual_ms", "sync.max_drift_ppm", "cut.default_len_s", "cut.min_len_s", "cut.beats_per_bar",
    "cut.bitrate_mbps",
}
NON_NEGATIVE = {"sync.min_confidence", "sync.warn_confidence", "cut.handle_s"}


def _check(name: str, key: str, value, kind):
    where = f"{name}.{key}"
    if kind is bool:
        ok = isinstance(value, bool)
    elif kind is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind is str:
        ok = isinstance(value, str)
    else:
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        value = tuple(value) if ok else value
    if not ok:
        wanted = "list of strings" if kind not in (bool, int, float, str) else kind.__name__
        raise ValueError(f"config {where} must be a {wanted}, got {value!r}")
    if kind in (int, float) and not math.isfinite(value):
        raise ValueError(f"config {where} must be a finite number, got {value!r}")
    if where in POSITIVE and value <= 0:
        raise ValueError(f"config {where} must be positive, got {value!r}")
    if where in NON_NEGATIVE and value < 0:
        raise ValueError(f"config {where} must not be negative, got {value!r}")
    if where == "cut.codec" and value not in CODECS:
        raise ValueError(f"config cut.codec must be one of {CODECS}, got {value!r}")
    return value


def load(path: Path | None = None) -> Config:
    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_PATH
    if explicit and not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    defaults = Config()
    sections = {f.name for f in dataclasses.fields(Config)}
    for name in data:
        if name not in sections:
            raise ValueError(f"unknown config section: {name}")
    merged = {}
    for name, values in data.items():
        section = getattr(defaults, name)
        if not hasattr(values, "items"):
            raise ValueError(f"config section {name} must be a table, got {values!r}")
        known = {f.name for f in dataclasses.fields(section)}
        for key in values:
            if key not in known:
                raise ValueError(f"unknown config key: {name}.{key}")
        kinds = typing.get_type_hints(type(section))
        values = {k: _check(name, k, v, kinds[k]) for k, v in values.items()}
        merged[name] = dataclasses.replace(section, **values)
    cfg = dataclasses.replace(defaults, **merged)
    _check_relations(cfg)
    return cfg


def _check_relations(cfg: Config) -> None:
    if cfg.cut.default_len_s < cfg.cut.min_len_s:
        raise ValueError(f"config cut.default_len_s ({cfg.cut.default_len_s:g}) must be at least cut.min_len_s "
                         f"({cfg.cut.min_len_s:g})")
