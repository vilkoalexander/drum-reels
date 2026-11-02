from pathlib import Path

from reels.service import read_json, write_json


def stamp(sources: list[Path], params: dict) -> dict:
    files = []
    for path in sources:
        stat = path.stat()
        files.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"sources": files, "params": params}


def stamp_path(dst: Path) -> Path:
    return dst.with_name(dst.name + ".src.json")


def fresh(dst: Path, expected: dict) -> bool:
    return dst.exists() and read_json(stamp_path(dst)) == expected


def publish(dst: Path, expected: dict) -> None:
    write_json(stamp_path(dst), expected)


def audio_stamp(src: Path, sample_rate: int) -> dict:
    return stamp([src], {"sample_rate": sample_rate})


def join_stamp(chunks: list[Path]) -> dict:
    return stamp(chunks, {})


def audio_fresh(src: Path, dst: Path, sample_rate: int) -> bool:
    return src.exists() and fresh(dst, audio_stamp(src, sample_rate))


def join_fresh(chunks: list[Path], dst: Path) -> bool:
    return all(c.exists() for c in chunks) and fresh(dst, join_stamp(chunks))


def temp_name(dst: Path) -> Path:
    return dst.with_name(f"{dst.stem}.part{dst.suffix}")
