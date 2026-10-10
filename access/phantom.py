"""Phantom test (PRD Step 6): X-rayed, dye-filled plastic tubes of known size (2, 3, 4, 6,
8, 10 mm) must be read within 0.1 mm at the machine's centre before patient images are
measured.

The tubes are found automatically (long straight dark bands), each is measured along its
middle 60% with the same width engine used for patients, and matched to the known
diameters by size order. Usage:

    python -m access.phantom <dicom-or-image> --diameters 2 3 4 6 8 10 [--mm-per-px 0.24]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import ndimage as ndi

from . import calibration as cal
from . import rules as rule_tables
from .measure import centreline_path, refine, width_profile

TOLERANCE_MM = 0.1


def tube_masks(frame: np.ndarray, max_diameter_px: float) -> tuple[list[np.ndarray], np.ndarray]:
    """(one mask per tube, dye map) for a phantom frame scaled to [0, 1], dye dark."""
    size = int(2.5 * max_diameter_px) | 1
    bg = ndi.grey_closing(frame, size=(size, size))
    dye = np.clip(bg - frame, 0, None)
    sm = ndi.gaussian_filter(dye, 1.0)
    thr = 0.3 * float(np.percentile(sm, 99.5))
    lbl, n = ndi.label(sm > thr)
    masks = []
    for i, sl in enumerate(ndi.find_objects(lbl), 1):
        m = lbl == i
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if m.sum() > 200 and extent > 60:
            masks.append(m)
    return masks, dye


def measure_phantom(frame: np.ndarray, diameters_mm: list[float], calibration: cal.Calibration, rules: dict) -> dict:
    if not calibration.reliable or not calibration.mm_per_px:
        return {"ok": False, "error": "A reliable calibration is needed to test millimetre accuracy: " + calibration.detail}
    mmpx = calibration.mm_per_px
    masks, dye = tube_masks(frame, max(diameters_mm) / mmpx)
    tubes = []
    for m in masks:
        path = centreline_path(m)
        if path is None or len(path) < 30:
            continue
        prof = width_profile(m, path, rules, dye)
        n = len(prof["width_px"])
        mid = list(range(int(0.2 * n), int(0.8 * n)))
        refine(prof, mid)
        w = prof["width_px"][mid]
        w = w[np.isfinite(w) & prof["fitted"][mid]]
        if len(w) < 10:
            continue
        tubes.append({"measured_px": float(np.median(w)), "spread_px": float(np.std(w)),
                      "row": float(prof["centres"][n // 2][0]), "col": float(prof["centres"][n // 2][1])})
    tubes.sort(key=lambda t: t["measured_px"])
    expected = sorted(diameters_mm)
    rows = []
    for t, d in zip(tubes, expected):
        mm = t["measured_px"] * mmpx
        rows.append({"known_mm": d, "measured_mm": round(mm, 3), "error_mm": round(mm - d, 3),
                     "spread_mm": round(t["spread_px"] * mmpx, 3), "pass": abs(mm - d) <= TOLERANCE_MM,
                     "position_rc": [round(t["row"]), round(t["col"])]})
    found_all = len(tubes) == len(expected)
    return {"ok": True, "mm_per_px": mmpx, "calibration": calibration.to_dict(), "tubes": rows,
            "found": len(tubes), "expected": len(expected), "tolerance_mm": TOLERANCE_MM,
            "passed": bool(found_all and all(r["pass"] for r in rows)),
            "note": None if found_all else f"Found {len(tubes)} tubes but {len(expected)} diameters were given; check the image."}


def main(argv=None) -> int:
    from pathlib import Path

    import cv2

    from .dicom_io import is_dicom, load_frames, read_run_info

    ap = argparse.ArgumentParser(description="Measure a phantom of dye-filled tubes of known size.")
    ap.add_argument("image")
    ap.add_argument("--diameters", type=float, nargs="+", required=True)
    ap.add_argument("--mm-per-px", type=float, help="use instead of the DICOM geometry (e.g. from a ruler)")
    ap.add_argument("--frame", type=int, help="frame to use from a multi-frame run (default: middle)")
    a = ap.parse_args(argv)
    rules = rule_tables.load("measurement_rules")
    p = Path(a.image)
    if is_dicom(p):
        frames = load_frames(p)
        frame = frames[a.frame if a.frame is not None else len(frames) // 2]
        c = cal.from_dicom(read_run_info(p), rules)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frame = img.astype(np.float32) / 255.0
        c = cal.NONE
    if a.mm_per_px:
        c = cal.Calibration(a.mm_per_px, 0.0, "given", True, f"Given on the command line: {a.mm_per_px} mm/px.")
    print(json.dumps(measure_phantom(frame, a.diameters, c, rules), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
