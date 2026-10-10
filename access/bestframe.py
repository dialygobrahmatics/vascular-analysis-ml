"""Best-frame picker (PRD Step 4, requirement R3): the frame where the vessel is most
filled with dye, skipping frames where the patient moved.

Dye shows dark on X-ray. Per pixel, the brightest value over the run (a high percentile,
robust to noise) is what that pixel looks like without dye; how much darker each frame is
than that is the dye at that pixel. Averaging the dye over the pixels that ever fill gives
the dye-density curve; its peak is the best frame. Static things (bone, devices, burned-in
text) cancel out because they are in the "no dye" reference too.

Motion is the frame-to-frame displacement measured by phase correlation; in DSA a moving
arm also smears the subtraction, so frames that moved are excluded from the choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

WORK_SIDE = 256
REF_PERCENTILE = 90
MOTION_FRAC = 0.012  # displacement vs previous frame, as a fraction of the frame width
FILL_MIN = 0.04  # dye level (fraction of the value range) for a pixel to count as "fills"


@dataclass
class BestFrame:
    index: int
    curve: list[float]
    moved: list[bool]
    confidence: str
    reason: str
    dye_dark: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _small(frames: np.ndarray) -> np.ndarray:
    n, h, w = frames.shape
    s = min(1.0, WORK_SIDE / max(h, w))
    if s == 1.0:
        return frames.astype(np.float32)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)


def motion_flags(small: np.ndarray) -> list[bool]:
    win = cv2.createHanningWindow(small.shape[2:0:-1], cv2.CV_32F)
    flags = [False]
    for a, b in zip(small[:-1], small[1:]):
        # phaseCorrelate windows its inputs in place, so hand it copies
        (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy(), win)
        flags.append(bool(np.hypot(dx, dy) / small.shape[2] > MOTION_FRAC))
    return flags


def dye_maps(frames: np.ndarray, dye_dark: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(per-frame dye maps, no-dye reference). Frames scaled to [0, 1]."""
    f = frames if dye_dark else 1.0 - frames
    ref = np.percentile(f, REF_PERCENTILE, axis=0)
    dye = np.clip(ref[None] - f, 0, None)
    return dye, ref


def pick_best_frame(frames: np.ndarray, dye_dark: bool = True) -> BestFrame:
    n = frames.shape[0]
    if n == 1:
        return BestFrame(0, [0.0], [False], "high", "single-frame image", dye_dark)
    small = _small(frames)
    dye, _ = dye_maps(small, dye_dark)
    dye = ndi.gaussian_filter(dye, (0, 1, 1))
    fills = dye.max(axis=0) > FILL_MIN
    if fills.sum() < 20:
        mid = n // 2
        return BestFrame(mid, [0.0] * n, [False] * n, "low",
                         "no frame shows a clear dye increase; middle frame chosen, doctor should pick", dye_dark)
    curve = dye[:, fills].mean(axis=1)
    moved = motion_flags(small)
    usable = [i for i in range(n) if not moved[i]] or list(range(n))
    best = max(usable, key=lambda i: curve[i])
    peak, base = float(curve.max()), float(np.percentile(curve, 10))
    prominence = (curve[best] - base) / max(peak, 1e-9)
    near_best = sum(1 for i in usable if curve[i] >= 0.97 * curve[best])
    if moved[best] or prominence < 0.3:
        conf, reason = "low", "weak or unclear dye peak"
    elif near_best > max(3, n // 5):
        conf, reason = "medium", f"{near_best} frames are within 3% of the peak; any of them may be equally good"
    else:
        conf, reason = "high", "clear dye peak"
    if any(moved):
        reason += f"; {sum(moved)} frame(s) with patient motion excluded"
    return BestFrame(int(best), [round(float(c), 5) for c in curve], moved, conf, reason, dye_dark)
