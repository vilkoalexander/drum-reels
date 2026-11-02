import xml.etree.ElementTree as ET

from reels.service import write_json

SYNC = {
    "youtube": {"path": "sources/youtube.mp4", "fps": 30.0, "width": 1920, "height": 1080, "duration_s": 7200.0},
    "sources": {
        "dji": {"chunks": ["sources/dji.mp4"], "path": "work/dji.joined.mp4", "fps": 59.94, "width": 3840,
                "height": 2160, "duration_s": 6900.0, "intercept_s": 37.3, "slope": 1.00008},
        "x5": {"chunks": ["sources/x5.mp4"], "path": "sources/x5.mp4", "fps": 30.0, "width": 1920, "height": 1080,
               "duration_s": 7000.0, "intercept_s": -12.1, "slope": 1.0},
    },
}


def moment(slug, label, start_s, end_s, skipped=None):
    files = {"youtube": "youtube.mov", "dji": "dji.mov", "x5": "x5.mov"}
    for cam in skipped or {}:
        del files[cam]
    return {"slug": slug, "label": label, "start_s": start_s, "end_s": end_s, "handle_s": 2.0,
            "files": files, "skipped": skipped or {}}


MOMENTS = [moment("1-00-10_fill", "fill", 610.0, 640.0), moment("1-30-00_solo", "solo", 5400.0, 5430.0, {"x5": "late"})]


def body(xml: str):
    return ET.fromstring(xml.split("<!DOCTYPE fcpxml>\n")[1])


def write_sync(service, sync=SYNC):
    service.ensure()
    for path in [sync["youtube"]["path"], *(s["path"] for s in sync["sources"].values())]:
        (service.root / path).parent.mkdir(parents=True, exist_ok=True)
        (service.root / path).write_bytes(b"media")
    write_json(service.work / "sync.json", sync)


def write_moments(service, moments=MOMENTS, sync=SYNC):
    files = [service.work / "sync.json", service.root / sync["youtube"]["path"],
             *(service.root / s["path"] for s in sync["sources"].values() if not s.get("failed"))]
    inputs = service.fingerprint(files, {})
    for m in moments:
        (service.cuts / m["slug"]).mkdir(exist_ok=True)
        write_json(service.cuts / m["slug"] / "moment.json", {**m, "inputs": inputs})
