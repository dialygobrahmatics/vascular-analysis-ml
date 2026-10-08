"""Contrast enhancement, field-of-view detection, and image-quality scoring."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import laplace


def enhance(img: np.ndarray, clahe: bool = True, clahe_clip: float = 0.01, denoise: bool = True) -> np.ndarray:
    out = img.astype(np.float32)
    if clahe:
        out = equalize_adapthist(out, clip_limit=clahe_clip).astype(np.float32)
    if denoise:
        out = ndi.median_filter(out, size=3)
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(float(hi - lo), 1e-6), 0, 1)


def field_of_view(img: np.ndarray) -> np.ndarray:
    """Mask of the collimated imaging field; excludes black borders and letterboxing."""
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    blur = ndi.gaussian_filter(img, 8)
    thr = max(0.05, 0.5 * float(threshold_otsu(img)))
    fov = ndi.binary_closing(blur > thr, structure=disk(10))
    lbl, n = ndi.label(fov)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        fov = lbl == int(np.argmax(sizes))
    fov = ndi.binary_fill_holes(fov)
    if fov.mean() < 0.02:
        fov = np.ones_like(fov, dtype=bool)
    return fov


def quality(img: np.ndarray, fov: np.ndarray) -> dict:
    vals = img[fov]
    contrast = float(vals.std())
    sharpness = float(laplace(ndi.gaussian_filter(img, 1.0))[fov].var())
    if contrast >= 0.15 and sharpness >= 1.5e-4:
        label = "adequate"
    elif contrast >= 0.08:
        label = "limited"
    else:
        label = "poor"
    return {"contrast": round(contrast, 4), "sharpness": round(sharpness, 6), "label": label}



POLARITIES = ("bright", "dark")


def polarity_scores(img: np.ndarray, max_side: int = 256, top_frac: float = 0.005, local_sigma: float = 8.0) -> dict:
    """How far the strongest ridges of each polarity stand out from their surroundings.

    For each polarity, take the pixels with the top `top_frac` Frangi response and
    measure their mean signed intensity offset from the local background (a wide
    Gaussian blur): brighter for "bright", darker for "dark". Real vessels stand out from
    what is around them; "ridges" of the opposite polarity are mostly the gaps between
    vessels or texture, which do not. A local rather than global background matters on
    non-subtracted X-ray, where bright lung fields and bone would otherwise make the
    dark-vessel frame look like a bright-ridge one.
    """
    from skimage.filters import frangi
    from skimage.transform import rescale

    scale = min(1.0, max_side / max(img.shape))
    small = enhance(rescale(img, scale, anti_aliasing=True) if scale < 1.0 else img)
    local = small - ndi.gaussian_filter(small, local_sigma)
    scores = {}
    for p in POLARITIES:
        v = frangi(small, sigmas=[1.0, 2.0, 3.0], black_ridges=(p == "dark"))
        offset = local[v >= np.quantile(v, 1.0 - top_frac)].mean()
        scores[p] = float(offset if p == "bright" else -offset)
    return scores


# gap = dark score - bright score. Bright-vessel projections (MRA/CTA MIP) give a large
# negative gap (about -0.2 to -0.5: the vessels are by far the most conspicuous thing in
# the image), while real non-subtracted X-ray angiograms land near zero to slightly
# positive (dark vessels compete with bone and lung texture); DSA is strongly positive.
# So "bright" needs clear evidence and everything else is treated as X-ray.
BRIGHT_GAP = -0.10
CONFIDENT_BRIGHT_GAP = -0.15
CONFIDENT_DARK_GAP = 0.10


def polarity_from_scores(scores: dict) -> dict:
    gap = scores["dark"] - scores["bright"]
    polarity = "bright" if gap < BRIGHT_GAP else "dark"
    return {
        "polarity": polarity,
        "source": "auto",
        "confident": bool(gap < CONFIDENT_BRIGHT_GAP or gap > CONFIDENT_DARK_GAP),
        "scores": {p: round(v, 4) for p, v in scores.items()},
    }


def detect_polarity(img: np.ndarray) -> dict:
    """Decide whether vessels appear bright (MR/CT angiography MIP) or dark (X-ray /
    DSA angiography); a borderline score is reported as not confident."""
    return polarity_from_scores(polarity_scores(img))
