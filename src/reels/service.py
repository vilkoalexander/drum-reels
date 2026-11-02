import json
import os
import posixpath
import tempfile
from collections.abc import Iterable
from pathlib import Path

CAMERAS = ("dji", "x5")
CODECS = ("h264", "prores")


class StageError(Exception):
    pass


class UsageError(Exception):
    pass


class Service:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def notes(self) -> Path:
        return self.root / "notes"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def audio(self) -> Path:
        return self.work / "audio"

    @property
    def cuts(self) -> Path:
        return self.root / "cuts"

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def youtube_path(self) -> Path:
        return self.sources / "youtube.mp4"

    def path(self, rel) -> Path:
        if not isinstance(rel, str) or not rel or rel.startswith("/") or posixpath.normpath(rel).startswith(".."):
            raise StageError(f"sidecar path {rel!r} points outside the service folder")
        return self.root / rel

    def ensure(self) -> None:
        for folder in (self.sources, self.work, self.audio, self.cuts, self.out):
            folder.mkdir(parents=True, exist_ok=True)

    def camera_files(self) -> dict[str, list[Path]]:
        found: dict[str, list[Path]] = {camera: [] for camera in CAMERAS}
        if not self.sources.is_dir():
            return found
        for path in sorted(self.sources.iterdir(), key=lambda p: p.name.lower()):
            name = path.name.lower()
            for camera in CAMERAS:
                if name.startswith(camera) and name.endswith(".mp4"):
                    found[camera].append(path)
        return found

    def fingerprint(self, paths: Iterable[Path], params: dict) -> dict:
        files = {}
        for path in paths:
            stat = Path(path).stat()
            files[Path(path).relative_to(self.root).as_posix()] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {"files": files, "params": json.loads(json.dumps(params))}


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        raise StageError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise StageError(f"{path} must hold a JSON object, got {type(data).__name__}")
    return data


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def up_to_date(sidecar: dict | None, fingerprint: dict, outputs: Iterable[Path]) -> bool:
    if sidecar is None or sidecar.get("inputs") != fingerprint:
        return False
    return all(Path(p).exists() for p in outputs)
