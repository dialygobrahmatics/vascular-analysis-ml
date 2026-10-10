"""Measurement engine (PRD section 6, Step 6, requirement R6): turn a vessel outline into
MLD, RVD, %DS and length, with an error range on every number.

1. Centreline: skeleton of the outline, then the path between the doctor's start and end
   points (or the longest path), smoothed and resampled every `sample_step_px`.
2. Width at right angles to the centreline at every sample -- never straight across the
   screen, which overstates width on a curve.
   - from the outline: where the (lightly smoothed) mask crosses 0.5, to sub-pixel
   - refined from the image when the dye map is available: the dye profile across a
     round vessel is its projected chord, so a blurred cylinder model is fitted to it.
     This uses all the pixels across the vessel instead of two edge pixels, which is what
     makes sub-pixel width (and the phantom target) reachable.
3. Width profile -> MLD (narrowest), span of the narrowing, RVD from the healthy vein next
   to it following the doctor's rule table (never an aneurysmal bulge), %DS and length.

Golden rule 4: mm values are only produced from a reliable calibration; otherwise only
percentages (which do not need calibration) are given.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

from .calibration import Calibration
from .rules import reference_rule

T_STEP = 0.25  # px, sampling along each normal


# ---------------------------------------------------------------- centreline ----

def _skeleton_graph(skel: np.ndarray):
    coords = np.argwhere(skel)
    index = -np.ones(skel.shape, int)
    index[tuple(coords.T)] = np.arange(len(coords))
    rows, cols, weights = [], [], []
    h, w = skel.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = coords[:, 0] + dy, coords[:, 1] + dx
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        j = np.full(len(coords), -1)
        j[ok] = index[r2[ok], c2[ok]]
        m = j >= 0
        rows.append(np.where(m)[0])
        cols.append(j[m])
        weights.append(np.full(m.sum(), math.hypot(dy, dx)))
    r, c, wt = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    g = coo_matrix((wt, (r, c)), shape=(len(coords), len(coords))).tocsr()
    return coords, g + g.T


def centreline_path(mask: np.ndarray, start=None, end=None) -> np.ndarray | None:
    """Ordered (row, col) skeleton pixels from start to end (nearest skeleton pixels to
    the given points), or the longest path through the largest connected piece."""
    skel = skeletonize(mask)
    if skel.sum() < 3:
        return None
    coords, g = _skeleton_graph(skel)

    def nearest(p):
        return int(np.argmin(np.hypot(coords[:, 0] - p[0], coords[:, 1] - p[1])))

    if start is not None and end is not None:
        s, e = nearest(start), nearest(end)
    else:
        _, labels = connected_components(g, directed=False)
        comp = np.bincount(labels).argmax() if start is None else labels[nearest(start)]
        node = nearest(start) if start is not None else int(np.where(labels == comp)[0][0])
        d = dijkstra(g, indices=node)
        d[~np.isfinite(d)] = -1
        s = node if start is not None else int(np.argmax(d))
        d2 = dijkstra(g, indices=s)
        d2[~np.isfinite(d2)] = -1
        e = int(np.argmax(d2))
    dist, pred = dijkstra(g, indices=s, return_predecessors=True)
    if not np.isfinite(dist[e]):
        return None
    path = [e]
    while path[-1] != s:
        path.append(pred[path[-1]])
    return coords[path[::-1]].astype(float)


def resample(path: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centres, unit normals, arc length) every `step` px along a smoothed path."""
    d = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], n)
    r = ndi.gaussian_filter1d(np.interp(t, s, path[:, 0]), 2.0, mode="nearest")
    c = ndi.gaussian_filter1d(np.interp(t, s, path[:, 1]), 2.0, mode="nearest")
    tr = np.gradient(ndi.gaussian_filter1d(r, 3.0, mode="nearest"))
    tc = np.gradient(ndi.gaussian_filter1d(c, 3.0, mode="nearest"))
    norm = np.hypot(tr, tc) + 1e-12
    normals = np.stack([-tc / norm, tr / norm], axis=1)
    centres = np.stack([r, c], axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(centres, axis=0).T))])
    return centres, normals, arc


# ------------------------------------------------------------ width profile ----

def _crossings(values: np.ndarray, centre: int, level: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel offsets (in samples, relative to `centre`) of the first crossing below
    `level` on each side, per row. NaN where the row never crosses."""
    n, m = values.shape
    out = []
    for side in (1, -1):
        seg = values[:, centre::side] if side == 1 else values[:, centre::-1]
        below = seg < level
        k = np.argmax(below, axis=1)
        ok = below[np.arange(n), k] & (k > 0)
        a = seg[np.arange(n), np.maximum(k - 1, 0)]
        b = seg[np.arange(n), k]
        frac = np.where(ok, (a - level) / np.maximum(a - b, 1e-9), np.nan)
        out.append(np.where(ok, (k - 1 + frac), np.nan))
    return out[1], out[0]  # left (negative side), right


def _cylinder(t: np.ndarray, c: float, r: float, a: float, b: float, psf_samples: float) -> np.ndarray:
    chord = 2.0 * np.sqrt(np.clip(r * r - (t - c) ** 2, 0.0, None))
    return b + a * ndi.gaussian_filter1d(chord, psf_samples, mode="nearest")


def _fit_profile(t: np.ndarray, p: np.ndarray, c0: float, r0: float, psf_px: float):
    """Least-squares blurred-cylinder fit; returns (centre, radius, radius sigma) or None."""
    win = np.abs(t - c0) <= 2.0 * r0 + 4.0
    tt, pp = t[win], p[win]
    if len(tt) < 12:
        return None
    b0 = float(np.median(np.concatenate([pp[:4], pp[-4:]])))
    a0 = max(float(pp.max() - b0), 1e-6) / max(2 * r0, 1.0)
    psf_samples = psf_px / T_STEP
    # the model is evaluated on the same regular t grid, so the PSF is a fixed-width blur
    x0 = np.array([c0, max(r0, 0.6), a0, b0])
    lo = np.array([c0 - max(r0, 2.0), 0.3, 0.0, b0 - abs(a0) * 4 * r0 - 1])
    hi = np.array([c0 + max(r0, 2.0), 3.0 * r0 + 4.0, a0 * 20 + 1, b0 + abs(a0) * 4 * r0 + 1])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    try:
        res = least_squares(lambda x: _cylinder(tt, *x, psf_samples) - pp, x0, bounds=(lo, hi), method="trf")
    except ValueError:
        return None
    if not res.success:
        return None
    dof = max(len(tt) - 4, 1)
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac) * s2
        sr = float(math.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        return None
    return float(res.x[0]), float(res.x[1]), sr


def width_profile(mask: np.ndarray, path: np.ndarray, rules: dict, dye: np.ndarray | None = None) -> dict:
    m = rules["measurement"]
    centres, normals, arc = resample(path, m["sample_step_px"])
    edt = ndi.distance_transform_edt(mask)
    R = float(max(6.0, 2.5 * edt.max() + 4.0))
    t = np.arange(-R, R + T_STEP / 2, T_STEP)
    ci = len(t) // 2
    pts = centres[:, None, :] + t[None, :, None] * normals[:, None, :]
    coords = [pts[..., 0].ravel(), pts[..., 1].ravel()]
    smooth = ndi.gaussian_filter(mask.astype(np.float32), 0.7)
    mv = ndi.map_coordinates(smooth, coords, order=1, cval=0.0).reshape(pts.shape[:2])
    left, right = _crossings(mv, ci)
    mask_w = (left + right) * T_STEP
    centre_off = (right - left) / 2 * T_STEP
    width = mask_w.copy()
    err = np.full(len(width), m["edge_error_px"] * math.sqrt(2))
    centre_t = centre_off.copy()
    dv = None
    if dye is not None:
        dv = ndi.map_coordinates(dye.astype(np.float32), coords, order=1, cval=0.0).reshape(pts.shape[:2])
        mw = _moment_widths(dv, t, centre_off, mask_w, m["psf_sigma_px"])
        use = np.isfinite(mw) & np.isfinite(mask_w) & _plausible(mw, mask_w)
        width[use] = mw[use]
    prof = {
        "arc_px": arc, "centres": centres, "normals": normals, "width_px": width, "err_px": err,
        "fitted": np.zeros(len(width), bool), "centre_t": centre_t, "mask_w": mask_w,
        "method": "outline edges" if dv is None else "dye profile (moments), cylinder fit at reported points",
        "_dv": dv, "_t": t, "_psf": m["psf_sigma_px"],
    }
    _update_edges(prof)
    return prof


def _plausible(width, mask_w):
    """Image-based width is trusted over the outline unless wildly different. The outline is
    a threshold and under-traces thin parts next to much wider ones, which is exactly what
    the image-based width corrects, so the accepted range is wide and proportional."""
    return (width >= 0.4 * mask_w - 1.0) & (width <= 2.5 * mask_w + 3.0)


def _moment_widths(dv, t, centre, mask_w, psf):
    """Fast width from the dye profile: for a round vessel the projected chord profile has
    second moment r^2/4 about its centre, and blur adds psf^2, so width = 4 sqrt(m2 - psf^2).
    Used for the whole profile; the reported points are refined by the full model fit."""
    out = np.full(len(dv), np.nan)
    for i, (p, c, mw) in enumerate(zip(dv, centre, mask_w)):
        if not np.isfinite(mw) or mw < 0.5:
            continue
        half = 0.8 * mw + 4
        win = np.abs(t - c) <= half
        ring = (np.abs(t - c) > half) & (np.abs(t - c) <= half + 4)
        b = float(np.median(p[ring])) if ring.any() else 0.0
        q = np.clip(p[win] - b, 0, None)
        total = q.sum()
        if total <= 0:
            continue
        tt = t[win]
        cc = (q * tt).sum() / total
        m2 = (q * (tt - cc) ** 2).sum() / total
        out[i] = 4 * math.sqrt(max(m2 - psf * psf, 0.0))
    return out


def _update_edges(prof):
    w = prof["width_px"]
    ok = np.isfinite(w)
    half = np.where(ok, w / 2, 0)
    ct = np.where(ok & np.isfinite(prof["centre_t"]), prof["centre_t"], 0)
    prof["edge_a"] = prof["centres"] + (ct - half)[:, None] * prof["normals"]
    prof["edge_b"] = prof["centres"] + (ct + half)[:, None] * prof["normals"]
    prof["width_px"] = np.where(ok, w, np.nan)


def attach_dye(prof: dict, dye: np.ndarray, rules: dict) -> None:
    """Re-sample the dye map along a stored profile's normals (needed for refine())."""
    mw = prof["mask_w"][np.isfinite(prof["mask_w"])]
    R = float(max(6.0, 1.25 * (mw.max() if len(mw) else 8.0) + 4.0))
    t = np.arange(-R, R + T_STEP / 2, T_STEP)
    pts = prof["centres"][:, None, :] + t[None, :, None] * prof["normals"][:, None, :]
    prof["_dv"] = ndi.map_coordinates(dye.astype(np.float32), [pts[..., 0].ravel(), pts[..., 1].ravel()],
                                      order=1, cval=0.0).reshape(pts.shape[:2])
    prof["_t"] = t
    prof["_psf"] = rules["measurement"]["psf_sigma_px"]


def refine(prof: dict, indices) -> None:
    """Full blurred-cylinder fit at the given samples (the ones the report quotes)."""
    dv, t = prof.get("_dv"), prof.get("_t")
    if dv is None:
        return
    for i in sorted(set(int(i) for i in indices)):
        if not (0 <= i < len(dv)) or prof["fitted"][i]:
            continue
        mw = prof["mask_w"][i]
        if not np.isfinite(mw) or mw < 0.5:
            continue
        fit = _fit_profile(t, dv[i], prof["centre_t"][i], mw / 2, prof["_psf"])
        if fit is None:
            continue
        c, r, sr = fit
        if _plausible(2 * r, mw):
            prof["width_px"][i], prof["err_px"][i], prof["centre_t"][i], prof["fitted"][i] = 2 * r, max(2 * sr, 0.1), c, True
    _update_edges(prof)


# ----------------------------------------------------------------- stenosis ----

def _span(w: np.ndarray, i: int, limit: float) -> tuple[int, int]:
    a = i
    while a > 0 and w[a - 1] < limit:
        a -= 1
    b = i
    while b < len(w) - 1 and w[b + 1] < limit:
        b += 1
    return a, b


def _side_samples(w, valid, start, direction, gap, window, aneurysm_limit):
    """Indices of healthy samples beyond `start` (exclusive) in `direction`, skipping the
    gap and any aneurysmal bulge ("use the healthy vein just beyond any bulge")."""
    i, got, skipped_aneurysm = start + direction * (gap + 1), [], False
    while 0 <= i < len(w) and len(got) < window:
        if valid[i]:
            if w[i] > aneurysm_limit:
                skipped_aneurysm = True
                if got:  # a bulge after healthy samples ends this reference window
                    break
            else:
                got.append(i)
        i += direction
    return got, skipped_aneurysm


def analyse_profile(profile: dict, rules: dict, calibration: Calibration, segment=None,
                    mld_index: int | None = None, ref_indices: list[int] | None = None,
                    device_mask: np.ndarray | None = None) -> dict:
    """Two passes: locate the narrowing and reference on the fast profile, refine exactly
    the samples the report quotes with the full model fit, then measure again."""
    first = _analyse(profile, rules, calibration, segment, mld_index, ref_indices, device_mask)
    if not first.get("ok") or profile.get("_dv") is None:
        return first
    m = first["mld_index"]
    pts = list(range(m - 3, m + 4)) + [i for a, b in first["ref_ranges"] for i in range(a, b + 1, 2)]
    if ref_indices:
        pts += list(ref_indices)
    refine(profile, pts)
    return _analyse(profile, rules, calibration, segment, mld_index, ref_indices, device_mask)


def _analyse(profile: dict, rules: dict, calibration: Calibration, segment=None,
             mld_index: int | None = None, ref_indices: list[int] | None = None,
             device_mask: np.ndarray | None = None) -> dict:
    st, ref_rule = rules["stenosis"], reference_rule(rules, segment)
    w, err, arc = profile["width_px"], profile["err_px"], profile["arc_px"]
    n = len(w)
    valid = np.isfinite(w) & (w > 0)
    if valid.sum() < 10:
        return {"ok": False, "notes": ["Too little of the vessel could be measured along this path."]}
    step = float(np.median(np.diff(arc))) if n > 1 else 1.0
    mmpx = calibration.mm_per_px if calibration.reliable else None
    to_samples = (lambda mm: max(1, int(round(mm / (mmpx or 0.25) / step))))  # 0.25 mm/px fallback for rule distances only
    ws = np.where(valid, ndi.gaussian_filter1d(np.where(valid, w, np.nanmedian(w)), 1.0), np.nan)
    typical = float(np.nanmedian(ws[valid]))
    # the skeleton ends inside the vessel's rounded ends, where width tapers to nothing:
    # keep at least about one vessel width away from both ends of the path
    margin = max(2, int(st["end_margin_fraction"] * n), int(math.ceil(1.2 * typical / step)))
    interior = np.zeros(n, bool)
    interior[margin:n - margin] = True
    aneurysm_limit = ref_rule["aneurysm_ratio"] * typical
    notes, reasons = [], []

    if mld_index is None:
        cand = np.where(valid & interior)[0]
        mld = int(cand[np.argmin(ws[cand])]) if len(cand) else int(np.nanargmin(ws))
        mld_source = "automatic"
    else:
        mld, mld_source = int(np.clip(mld_index, 0, n - 1)), "doctor"
    # report the MLD from the unsmoothed width at the narrowest point near the smoothed minimum
    lo_i, hi_i = max(0, mld - 2), min(n, mld + 3)
    if mld_source == "automatic":
        mld = lo_i + int(np.nanargmin(np.where(valid[lo_i:hi_i], w[lo_i:hi_i], np.inf)))

    rvd, ref_ranges, side_used = typical, [], ref_rule["side"]
    span = (mld, mld)
    gap, window = to_samples(ref_rule["gap_mm"]), to_samples(ref_rule["window_mm"])
    skipped = False
    if ref_indices:
        idx = [int(i) for i in ref_indices if 0 <= int(i) < n and valid[int(i)]]
        rvd = float(np.median(w[idx])) if idx else None
        ref_ranges = [[min(idx), max(idx)]] if idx else []
        side_used = "doctor"
        span = _span(ws, mld, st["normal_fraction"] * rvd) if rvd else span
    else:
        for _ in range(3):  # span and reference depend on each other: iterate to agreement
            span = _span(ws, mld, st["normal_fraction"] * rvd)
            sides = {"upstream": [-1], "downstream": [1], "both": [-1, 1]}[ref_rule["side"]]
            picks, ref_ranges, meds = [], [], []
            for d in sides:
                got, sk = _side_samples(ws, valid, span[0] if d < 0 else span[1], d, gap, window, aneurysm_limit)
                skipped |= sk
                if len(got) >= ref_rule["min_samples"]:
                    meds.append(float(np.median(w[got])))
                    ref_ranges.append([min(got), max(got)])
                    picks += got
            if not meds:
                rvd = None
                break
            new = float(np.mean(meds))
            if abs(new - rvd) < 0.05:
                rvd = new
                break
            rvd = new
        if rvd is not None and len(ref_ranges) < len({"upstream": [-1], "downstream": [1], "both": [-1, 1]}[ref_rule["side"]]):
            notes.append("Healthy reference found on one side only.")
            reasons.append("reference on one side only")
    if skipped:
        notes.append("An aneurysmal (ballooned) part was skipped when choosing the normal reference width.")

    mld_px, mld_err = float(w[mld]), float(err[mld])
    depth = 1 - mld_px / typical if typical > 0 else 0.0
    if mld_source == "automatic" and depth < st["candidate_min_depth"]:
        notes.append(f"No narrowing deeper than {st['candidate_min_depth'] * 100:.0f}% of the typical width along this path; "
                     "the narrowest point is shown for reference.")
    out = {
        "narrowing_found": bool(mld_source == "doctor" or depth >= st["candidate_min_depth"]),
        "ok": True, "segment": segment, "mld_index": mld, "mld_source": mld_source, "span": [int(span[0]), int(span[1])],
        "ref_ranges": ref_ranges, "ref_side": side_used, "mld_px": round(mld_px, 3), "mld_err_px": round(2 * mld_err, 3),
        "rvd_px": None, "ds_percent": None, "ds_err": None, "length_px": None,
        "mld_mm": None, "mld_mm_err": None, "rvd_mm": None, "rvd_mm_err": None, "length_mm": None, "length_mm_err": None,
        "percentage_available": rvd is not None, "mm_available": mmpx is not None,
        "calibration": calibration.to_dict(), "width_method": profile["method"], "notes": notes,
        "mld_points": [profile["edge_a"][mld].round(2).tolist(), profile["edge_b"][mld].round(2).tolist()],
    }
    if rvd is not None:
        ridx = [i for a, b in ref_ranges for i in range(a, b + 1) if valid[i]] or [mld]
        rvd_err = float(np.median(err[ridx])) / math.sqrt(max(len(ridx), 1)) + float(np.std(w[ridx])) / math.sqrt(max(len(ridx), 1))
        ratio = mld_px / rvd
        ds = (1 - ratio) * 100
        ds_err = 100 * ratio * math.hypot(mld_err / max(mld_px, 1e-6), rvd_err / rvd)
        length_px = float(arc[span[1]] - arc[span[0]]) + step if span[1] > span[0] else 0.0
        mid_ref = ridx[len(ridx) // 2]
        out.update(rvd_px=round(rvd, 3), rvd_err_px=round(2 * rvd_err, 3), ds_percent=round(ds, 1), ds_err=round(2 * ds_err, 1),
                   length_px=round(length_px, 2),
                   ref_points=[profile["edge_a"][mid_ref].round(2).tolist(), profile["edge_b"][mid_ref].round(2).tolist()],
                   significant=bool(ds > st["significant_ds_percent"]))
    else:
        notes.append("No healthy vessel next to the narrowing to compare with: percentage not available, MLD only.")
        reasons.append("no healthy reference")
        out["significant"] = None

    if mmpx is not None:
        rel = calibration.rel_error or 0.0

        def mm(v_px, e_px):
            v = v_px * mmpx
            return round(v, 2), round(2 * math.hypot(e_px * mmpx, v * rel), 2)  # ~95% range

        out["mld_mm"], out["mld_mm_err"] = mm(mld_px, mld_err)
        if rvd is not None:
            out["rvd_mm"], out["rvd_mm_err"] = mm(rvd, out["rvd_err_px"] / 2)
            out["length_mm"], out["length_mm_err"] = mm(out["length_px"], 2 * step)
    else:
        notes.append("Millimetres not available: " + calibration.detail)

    # confidence (golden rule 5)
    if mld_px < rules["measurement"]["min_reliable_width_px"]:
        reasons.append(f"narrowest width is only {mld_px:.1f} px across")
    if not profile["fitted"][mld]:
        reasons.append("width taken from the outline, not refined from the image")
    if device_mask is not None:
        pts = np.array(out["mld_points"] + [profile["centres"][mld].tolist()]).round().astype(int)
        h, w_img = device_mask.shape
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w_img)]
        if len(pts) and ndi.binary_dilation(device_mask, iterations=3)[pts[:, 0], pts[:, 1]].any():
            reasons.append("a device (catheter / wire / sheath) overlaps the narrowing")
            notes.append("A device overlaps this narrowing: it may be a catheter or a fibrin sheath rather than a true narrowing.")
    if mld < margin or mld > n - 1 - margin:
        reasons.append("narrowest point is at the end of the measured path")
    out["confidence"] = "high" if not reasons else ("medium" if len(reasons) == 1 else "low")
    out["confidence_reasons"] = reasons
    return out


def candidate_narrowings(profile: dict, rules: dict, max_candidates: int = 5) -> list[int]:
    """Sample indices of distinct local width minima deep enough to be worth showing."""
    w = profile["width_px"]
    valid = np.isfinite(w)
    if valid.sum() < 10:
        return []
    ws = ndi.gaussian_filter1d(np.where(valid, w, np.nanmedian(w)), 2.0)
    typical = float(np.median(ws))
    depth = 1 - ws / typical
    mins = [i for i in range(1, len(ws) - 1) if ws[i] <= ws[i - 1] and ws[i] <= ws[i + 1]
            and depth[i] >= rules["stenosis"]["candidate_min_depth"]]
    picked = []
    for i in sorted(mins, key=lambda i: ws[i]):
        if all(abs(i - j) > 15 for j in picked):
            picked.append(i)
    return picked[:max_candidates]


def measure_vessel(mask: np.ndarray, rules: dict, calibration: Calibration, dye: np.ndarray | None = None,
                   start=None, end=None, segment=None, device_mask: np.ndarray | None = None,
                   mld_index: int | None = None, ref_indices: list[int] | None = None) -> dict:
    path = centreline_path(mask, start, end)
    if path is None or len(path) < 10:
        return {"ok": False, "notes": ["No continuous vessel centreline between the chosen points."]}
    prof = width_profile(mask, path, rules, dye)
    finding = analyse_profile(prof, rules, calibration, segment, mld_index, ref_indices, device_mask)
    return {"ok": finding.get("ok", False), "profile": prof, "finding": finding,
            "candidates": candidate_narrowings(prof, rules)}


def profile_to_json(profile: dict) -> dict:
    """Plain lists for storage / the browser."""
    return {
        "arc_px": np.round(profile["arc_px"], 2).tolist(),
        "width_px": [None if not np.isfinite(v) else round(float(v), 3) for v in profile["width_px"]],
        "err_px": np.round(profile["err_px"], 3).tolist(),
        "centres": np.round(profile["centres"], 2).tolist(),
        "normals": np.round(profile["normals"], 4).tolist(),
        "edge_a": np.round(profile["edge_a"], 2).tolist(),
        "edge_b": np.round(profile["edge_b"], 2).tolist(),
        "fitted": profile["fitted"].tolist(),
        "centre_t": [None if not np.isfinite(v) else round(float(v), 3) for v in profile["centre_t"]],
        "mask_w": [None if not np.isfinite(v) else round(float(v), 3) for v in profile["mask_w"]],
        "method": profile["method"],
    }


def profile_from_json(d: dict) -> dict:
    return {
        "arc_px": np.array(d["arc_px"]), "width_px": np.array([np.nan if v is None else v for v in d["width_px"]]),
        "err_px": np.array(d["err_px"]), "centres": np.array(d["centres"]), "normals": np.array(d["normals"]),
        "edge_a": np.array(d["edge_a"]), "edge_b": np.array(d["edge_b"]), "fitted": np.array(d["fitted"], bool),
        "centre_t": np.array([np.nan if v is None else v for v in d["centre_t"]]),
        "mask_w": np.array([np.nan if v is None else v for v in d["mask_w"]]),
        "method": d["method"],
    }
