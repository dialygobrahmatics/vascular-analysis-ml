"""Measurement accuracy on synthetic geometry with exact answers (PRD Step 6)."""

import numpy as np
from scipy import ndimage as ndi

from access import calibration as cal
from access import rules as rule_tables
from access.measure import analyse_profile, measure_vessel
from access.phantom import measure_phantom

MMPX = 0.24
RULES = rule_tables.load("measurement_rules")
CAL = cal.Calibration(MMPX, 0.0, "given", True, "test")


def tube_image(diameters_mm, h=520, w=700, psf=0.8, noise=0.004, seed=1, mu=0.012):
    """Horizontal dye-filled tubes: intensity drops with projected chord length."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = np.full((h, w), 0.8)
    for k, d in enumerate(diameters_mm):
        r = d / 2 / MMPX
        cy = 50 + k * 80 + 0.37 * k  # off-pixel centres
        chord = 2 * np.sqrt(np.clip(r * r - (yy - cy) ** 2, 0, None))
        chord[(xx < 60) | (xx > w - 60)] = 0
        img -= mu * chord
    img = ndi.gaussian_filter(img, psf) + rng.normal(0, noise, (h, w))
    return np.clip(img, 0, 1)


def test_phantom_tubes_within_0_1_mm():
    diam = [2, 3, 4, 6]
    res = measure_phantom(tube_image(diam), diam, CAL, RULES)
    assert res["ok"] and res["found"] == len(diam), res
    for row in res["tubes"]:
        assert abs(row["error_mm"]) <= 0.1, row
    assert res["passed"]


def test_phantom_requires_reliable_calibration():
    res = measure_phantom(tube_image([2, 3]), [2, 3], cal.NONE, RULES)
    assert not res["ok"]


def _narrowed_vessel(rvd_mm=6.0, mld_mm=2.1, length_mm=12.0, h=200, w=620, aneurysm=None):
    """Straight vessel with a cosine narrowing in the middle; returns (mask, dye)."""
    xx = np.arange(w, dtype=float)
    s_mm = (xx - 40) * MMPX
    d = np.full(w, rvd_mm)
    c = (w / 2 - 40) * MMPX
    inside = np.abs(s_mm - c) < length_mm / 2
    d[inside] = rvd_mm - (rvd_mm - mld_mm) * 0.5 * (1 + np.cos(np.pi * (s_mm[inside] - c) / (length_mm / 2)))
    if aneurysm:
        a_c, a_len, a_d = aneurysm
        ins = np.abs(s_mm - a_c) < a_len / 2
        d[ins] = np.maximum(d[ins], rvd_mm + (a_d - rvd_mm) * 0.5 * (1 + np.cos(np.pi * (s_mm[ins] - a_c) / (a_len / 2))))
    r = d / 2 / MMPX
    yy = np.arange(h, dtype=float)[:, None] - h / 2 - 0.3
    chord = 2 * np.sqrt(np.clip(r[None, :] ** 2 - yy ** 2, 0, None))
    chord[:, :40] = chord[:, -40:] = 0
    dye = ndi.gaussian_filter(0.012 * chord, 0.8) + np.random.default_rng(3).normal(0, 0.004, chord.shape)
    mask = ndi.gaussian_filter(dye, 1.0) > 0.15 * dye.max()
    return mask, dye


def test_stenosis_measurement_matches_truth():
    mask, dye = _narrowed_vessel()
    res = measure_vessel(mask, RULES, CAL, dye=dye)
    f = res["finding"]
    assert abs(f["mld_mm"] - 2.1) <= 0.15
    assert abs(f["rvd_mm"] - 6.0) <= 0.15
    assert abs(f["ds_percent"] - 65.0) <= 3.0
    assert f["narrowing_found"] and f["significant"]
    assert f["mld_mm_err"] > 0 and f["ds_err"] > 0


def test_aneurysm_never_used_as_reference():
    # a 12 mm bulge right next to the narrowing would make the narrowing look far worse
    mask, dye = _narrowed_vessel(aneurysm=(85 * 0.24 * 4.4, 15.0, 12.0))
    f = measure_vessel(mask, RULES, CAL, dye=dye)["finding"]
    assert abs(f["rvd_mm"] - 6.0) <= 0.3, f
    assert abs(f["ds_percent"] - 65.0) <= 5.0, f


def test_no_mm_without_reliable_calibration():
    mask, dye = _narrowed_vessel()
    f = measure_vessel(mask, RULES, cal.NONE, dye=dye)["finding"]
    assert f["mld_mm"] is None and f["rvd_mm"] is None and not f["mm_available"]
    assert f["ds_percent"] is not None  # percentages do not need calibration


def test_doctor_moving_points_recalculates():
    mask, dye = _narrowed_vessel()
    res = measure_vessel(mask, RULES, CAL, dye=dye)
    f = res["finding"]
    moved = analyse_profile(res["profile"], RULES, CAL, mld_index=f["mld_index"] + 15)
    assert moved["mld_source"] == "doctor" and moved["mld_px"] > f["mld_px"]
    ref = analyse_profile(res["profile"], RULES, CAL, ref_indices=[20, 21, 22])
    assert ref["ref_side"] == "doctor"


def test_no_healthy_reference_gives_mld_only():
    """Shoulder bend (segment 6) compares with the vein just before it; if the narrowing
    reaches the start of the measured vessel there is nothing healthy to compare with."""
    n = 200
    w = np.full(n, 25.0)
    w[:36] = 12.0
    w[20] = 9.0
    prof = {"width_px": w, "err_px": np.full(n, 0.5), "arc_px": np.arange(n, dtype=float), "fitted": np.ones(n, bool),
            "centres": np.stack([np.full(n, 50.0), np.arange(n, dtype=float)], 1), "method": "test", "_dv": None}
    prof["edge_a"] = prof["centres"] - [w[0] / 2, 0]
    prof["edge_b"] = prof["centres"] + [w[0] / 2, 0]
    f = analyse_profile(prof, RULES, CAL, segment=6, mld_index=20)
    assert f["rvd_mm"] is None and not f["percentage_available"] and f["ds_percent"] is None
    assert abs(f["mld_mm"] - 9 * MMPX) < 1e-6
    assert any("percentage not available" in n for n in f["notes"])
