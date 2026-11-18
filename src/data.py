"""Dataset indexing and identical preprocessing for training and inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


HEIGHT, WIDTH = 66, 200
OUTPUT_SCALE_DEGREES = 45.0


@dataclass(frozen=True)
class DrivingSample:
    image_path: Path
    angle_degrees: float


def read_index(root: str | Path) -> list[DrivingSample]:
    """Read filename + steering degrees records in Sully Chen's text format.

    The 2018 version optionally puts a timestamp after a comma in column two.
    Files remain in index order so that training can split chronologically.
    """
    root = Path(root)
    index = root / "data.txt"
    if not index.is_file():
        raise FileNotFoundError(f"Missing dataset index: {index}")

    samples: list[DrivingSample] = []
    for line_no, raw in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 2:
            raise ValueError(f"{index}:{line_no}: expected 'filename angle'")
        relative = Path(columns[0])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{index}:{line_no}: filename must be inside dataset folder")
        try:
            angle = float(columns[1].split(",", 1)[0])
        except ValueError as exc:
            raise ValueError(f"{index}:{line_no}: invalid steering angle") from exc
        if not np.isfinite(angle):
            raise ValueError(f"{index}:{line_no}: non-finite steering angle")
        image_path = root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Referenced image does not exist: {image_path}")
        samples.append(DrivingSample(image_path, angle))
    if not samples:
        raise ValueError("Dataset index has no valid samples")
    return samples


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Return 1x3x66x200 float32 tensor, scaled to [0,1]."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    w, h = rgb.size
    if w < 20 or h < 20:
        raise ValueError("Image is too small for steering prediction")
    road = rgb.crop((0, max(0, h - 150), w, h))
    resized = road.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()
