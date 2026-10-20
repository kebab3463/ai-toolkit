"""A 2D body-width measure for weight sliders: silhouette width in torso lengths.

From a person matte (Sapiens2) and COCO keypoints (Keypoint R-CNN): the torso
length T is the distance from the shoulder midpoint to the hip midpoint, and
the score is the mean silhouette width over rows from just below the
shoulders (0.15 T) to the upper thighs (1.5 T below the shoulders), divided by
T. Heavier bodies are wider at the same torso length, so the score rises with
weight; dividing by T makes it independent of image size and camera distance.

What it is good for: comparisons where pose and framing are (nearly) fixed,
i.e. a sweep image against its own m = 0 render, or a tuple's members against
each other. What it is not: an absolute weight estimate across different
photos. Rows are horizontal, so it assumes an upright person; arms beside the
torso count as width (they thicken with weight too); a side or back view
measures depth or back width instead of front width. Only the connected
person region around the torso is counted, so other people don't add width.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

SHOULDERS, HIPS = (5, 6), (11, 12)
BAND = (0.15, 1.5)  # rows measured, in torso lengths below the shoulder midpoint
LEVELS = {"chest": 0.35, "waist": 0.7, "hips": 1.0, "thighs": 1.35}  # reported widths, same units
MIN_TORSO_PX = 16


@dataclass
class BodyMeasurement:
    score: Optional[float]  # mean band width / torso length; None if not measurable
    reason: str = ""  # why score is None
    torso_px: float = 0.0
    band: Tuple[int, int] = (0, 0)  # rows [y0, y1)
    widths: Dict[str, float] = field(default_factory=dict)  # LEVELS rows, in torso lengths


def _person_region(matte: np.ndarray, center: Tuple[float, float], threshold: float = 0.5) -> np.ndarray:
    """The matte's connected component containing ``center`` (else the largest one), as a 0/1 array."""
    import cv2

    binary = (matte >= threshold).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    if n <= 1:
        return binary.astype(np.float32)
    h, w = binary.shape
    cx, cy = int(round(min(max(center[0], 0), w - 1))), int(round(min(max(center[1], 0), h - 1)))
    label = labels[cy, cx]
    if label == 0:  # torso center outside the matte: take the largest component
        label = 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    return (labels == label).astype(np.float32)


def measure_body(matte: np.ndarray, keypoints: Optional[np.ndarray], min_score: float = 1.0) -> BodyMeasurement:
    """Body width in torso lengths from an H x W matte in [0, 1] and (17, 3) COCO keypoints (x, y, score)."""
    if keypoints is None:
        return BodyMeasurement(None, "no person detected")
    kps = np.asarray(keypoints, dtype=np.float64)
    needed = SHOULDERS + HIPS
    missing = [i for i in needed if kps[i, 2] < min_score]
    if missing:
        return BodyMeasurement(None, f"shoulder/hip keypoints not visible: {missing}")
    shoulder = kps[list(SHOULDERS), :2].mean(axis=0)
    hip = kps[list(HIPS), :2].mean(axis=0)
    torso = float(np.linalg.norm(hip - shoulder))
    if torso < MIN_TORSO_PX:
        return BodyMeasurement(None, f"torso too short ({torso:.0f} px)", torso_px=torso)

    h = matte.shape[0]
    region = _person_region(matte, tuple((shoulder + hip) / 2))
    row_width = (np.clip(matte, 0, 1) * region).sum(axis=1)  # soft width per row, px
    y0 = int(round(shoulder[1] + BAND[0] * torso))
    y1 = int(round(shoulder[1] + BAND[1] * torso))
    y0, y1 = max(y0, 0), min(y1, h)
    if y1 - y0 < 4:
        return BodyMeasurement(None, "torso band outside the image", torso_px=torso, band=(y0, y1))
    widths = {}
    for name, frac in LEVELS.items():
        y = int(round(shoulder[1] + frac * torso))
        if 0 <= y < h:
            widths[name] = float(row_width[max(y - 2, 0):y + 3].mean() / torso)
    return BodyMeasurement(float(row_width[y0:y1].mean() / torso), torso_px=torso, band=(y0, y1), widths=widths)


def draw_measurement(image: np.ndarray, m: BodyMeasurement) -> np.ndarray:
    """RGB uint8 copy of ``image`` with the measured band shaded and the score written on it."""
    import cv2

    out = image.copy()
    if m.score is not None:
        y0, y1 = m.band
        tint = out[y0:y1].astype(np.float32)
        tint[..., 1] = tint[..., 1] * 0.6 + 255 * 0.4
        out[y0:y1] = tint.astype(np.uint8)
        cv2.line(out, (0, y0), (out.shape[1] - 1, y0), (0, 255, 0), 2)
        cv2.line(out, (0, y1 - 1), (out.shape[1] - 1, y1 - 1), (0, 255, 0), 2)
    text = f"{m.score:.3f}" if m.score is not None else m.reason[:40]
    scale = max(out.shape[1] / 600, 0.6)
    cv2.putText(out, text, (8, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale))
    cv2.putText(out, text, (8, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), int(2 * scale))
    return out


_MODELS = {}


def body_width(image, device: str = "cuda:0") -> Optional[float]:
    """PIL image -> score or None, loading Sapiens2 + Keypoint R-CNN once (for --measure module:function use).
    slider_sweep.py's built-in ``--measure body`` reuses its own models instead."""
    from .tuple_masks import load_mask_models

    if device not in _MODELS:
        _MODELS[device] = load_mask_models(device)
    matte_fn, keypoints_fn = _MODELS[device]
    image = image.convert("RGB")
    return measure_body(matte_fn(image), keypoints_fn(image)).score
