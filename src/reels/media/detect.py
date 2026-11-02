import logging
from pathlib import Path

from reels.media.ffmpeg import run_ffmpeg
from reels.service import StageError

log = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parents[3] / "models" / "yolo11n.pt"
PERSON = 0
MIN_CONFIDENCE = 0.35
# share of the equirect width copied past the right edge so a person on the seam is seen whole
SEAM_PAD = 0.125

Box = tuple[float, float, float, float]


def grab(path: Path, times_s: list[float], width: int, folder: Path) -> list[Path]:
    """One jpg per time, scaled to width; the file's rotation tag is applied, as when rendering."""
    out = []
    for i, t in enumerate(times_s):
        dst = folder / f"frame_{i}.jpg"
        run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3",
                    str(dst)])
        out.append(dst)
    return out


def _libs():
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as error:
        raise StageError("person detection needs the people extra: pip install -e '.[people]'") from error
    if not MODEL.exists():
        raise StageError(f"missing {MODEL}, download yolo11n.pt into models/")
    return cv2, np, YOLO


class Detector:
    def __init__(self):
        self.cv2, self.np, yolo = _libs()
        self.model = yolo(str(MODEL))
        try:
            import torch
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

    def _boxes(self, image) -> tuple[list[Box], list[float]]:
        result = self.model.predict(image, classes=[PERSON], conf=MIN_CONFIDENCE, verbose=False, device=self.device,
                                    imgsz=1280)[0]
        return ([tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()],
                [float(c) for c in result.boxes.conf.cpu().numpy()])

    def equirect(self, path: Path) -> tuple[list[Box], int, int]:
        """Person boxes in an equirect frame, x measured on the unpadded width."""
        image = self.cv2.imread(str(path))
        height, width = image.shape[:2]
        pad = image[:, : int(width * SEAM_PAD)]
        boxes, _ = self._boxes(self.np.concatenate([image, pad], axis=1))
        return boxes, width, height

    def upright_score(self, path: Path, rotate: int) -> float:
        """Sum of person confidences after turning the frame clockwise by rotate degrees."""
        image = self.cv2.imread(str(path))
        turns = {0: None, 90: self.cv2.ROTATE_90_CLOCKWISE, 180: self.cv2.ROTATE_180,
                 270: self.cv2.ROTATE_90_COUNTERCLOCKWISE}
        if turns[rotate] is not None:
            image = self.cv2.rotate(image, turns[rotate])
        return sum(self._boxes(image)[1])

    def preview(self, frame: Path, labels: list[tuple[str, float]], boxes: list[Box], dst: Path) -> None:
        """The equirect frame with every box, each named view at its yaw, and a degree scale along the top."""
        cv2 = self.cv2
        image = cv2.imread(str(frame))
        height, width = image.shape[:2]
        for x0, y0, x1, y1 in boxes:
            for shift in (0, -width):
                cv2.rectangle(image, (int(x0 + shift), int(y0)), (int(x1 + shift), int(y1)), (0, 200, 255), 2)
        for deg in range(-180, 181, 15):
            x = int((deg + 180) / 360 * (width - 1))
            cv2.line(image, (x, 0), (x, 24 if deg % 45 else 40), (255, 255, 255), 2)
            if deg % 45 == 0:
                cv2.putText(image, str(deg), (x + 4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for name, yaw in labels:
            x = int((yaw + 180) / 360 * (width - 1))
            cv2.line(image, (x, 40), (x, height), (80, 255, 80), 1)
            cv2.putText(image, f"{name} {yaw:+.0f}", (x + 6, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 80), 2)
        cv2.imwrite(str(dst), image)
