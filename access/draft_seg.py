"""Draft vessel and device outlines for the best frame (R4) -- a starting point for the
doctor to correct, until the trained segmentation model (PRD Step 5, nnU-Net on
doctor-labelled frames) exists.

Vessel: dye is what changes during the run. Subtracting each pixel's no-dye appearance
(see bestframe.dye_maps) from the best frame leaves the dye-filled lumen and removes
bone, soft tissue and devices, which are there before and after the dye: the same idea
as the scanner's own subtraction (DSA), done in software.

Devices (sheath, catheter, wire): thin dark lines that are already visible in the no-dye
reference. They are drawn as their own class so they are never measured as vessel.

A single frame (spot image) has no no-dye reference: the vessel draft then falls back to
a morphological background estimate and devices cannot be separated; the result says so.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import disk

from .bestframe import dye_maps

MIN_VESSEL_PX = 150
DEVICE_SIGMAS = (1.0, 1.5, 2.0)
DEVICE_MAX_WIDTH_PX = 9  # mean width (area / length) above this is not a wire or catheter
DEVICE_MIN_LENGTH_PX = 40


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lbl, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lbl]


def _noise_sigma(x: np.ndarray) -> float:
    d = x - ndi.median_filter(x, 3)
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) + 1e-6


def draft_masks(frames: np.ndarray, best: int, dye_dark: bool = True) -> dict:
    notes: list[str] = []
    frame = frames[best] if dye_dark else 1.0 - frames[best]
    if frames.shape[0] >= 3:
        dye_all, ref = dye_maps(frames, dye_dark)
        dye_raw = dye_all[best]
        dye = ndi.gaussian_filter(dye_raw, 1.0)
        method = "temporal subtraction"
    else:
        # no no-dye frame: estimate background as the frame with thin dark structures removed
        bg = ndi.grey_closing(frame, footprint=disk(15))
        dye_raw = np.clip(bg - frame, 0, None)
        dye = ndi.gaussian_filter(dye_raw, 1.0)
        ref = bg
        method = "single-frame background estimate"
        notes.append("Single frame: devices cannot be separated from vessels automatically; check the outline carefully.")

    noise = _noise_sigma(dye)
    hi = max(6 * noise, 0.3 * float(np.percentile(dye, 99.5)))
    vessel = apply_hysteresis_threshold(dye, 0.5 * hi, hi)
    vessel = _drop_small(vessel, MIN_VESSEL_PX)
    vessel = vessel | (_drop_small(~vessel, 200) ^ ~vessel)  # fill holes under 200 px

    device = np.zeros_like(vessel)
    if frames.shape[0] >= 3:
        ridge = np.max([frangi(ref, sigmas=[s], black_ridges=True) for s in DEVICE_SIGMAS], axis=0)
        # blacked-out regions (anonymised text, collimator) have the sharpest edges in the
        # image and would otherwise set the threshold and be read as devices
        blank = ndi.binary_dilation(ref <= float(ref.min()) + 1e-4, iterations=4)
        ridge[blank] = 0
        thr = max(float(np.percentile(ridge[~blank], 99.5)) if (~blank).any() else 0.0, 1e-6)
        line = ridge > 0.5 * thr
        # a device is long and thin: keep components whose mean width (area / length
        # along their longest extent) is wire- or catheter-like
        # ... and it must be clearly darker than its surroundings, not ridge-shaped noise
        contrast = ndi.grey_closing(ref, footprint=disk(5)) - ref
        min_contrast = 5 * _noise_sigma(ref)
        lbl, n = ndi.label(line, structure=np.ones((3, 3)))
        for i, sl in enumerate(ndi.find_objects(lbl), 1):
            comp = lbl[sl] == i
            extent = float(np.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
            if (extent >= DEVICE_MIN_LENGTH_PX and comp.sum() / extent <= DEVICE_MAX_WIDTH_PX
                    and float(contrast[sl][comp].mean()) >= min_contrast):
                device[sl] |= comp
        # a device lying inside a filled vessel is both; vessel pixels stay vessel
        device &= ~ndi.binary_erosion(vessel, iterations=2)
        device = ndi.binary_dilation(device, iterations=1)
    if not vessel.any():
        notes.append("No dye-filled vessel found on this frame.")
    # dye_raw (unsmoothed) is what the measurement engine fits: extra smoothing would bias widths
    return {"vessel": vessel, "device": device, "dye": dye, "dye_raw": dye_raw, "reference": ref, "method": method, "notes": notes,
            "noise": noise}
