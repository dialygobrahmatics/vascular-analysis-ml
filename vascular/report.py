"""
Single-paragraph Vascular Observation Report.

Deterministic template over the measured metrics: no generative model is involved, so
every number in the paragraph is traceable to the JSON metrics file.
Wording follows the vessel polarity the image was analysed with: "bright" for MR/CT
angiography projections (MIP), "dark" for X-ray / DSA angiography.
"""

from __future__ import annotations
from pathlib import Path

MODALITY = {
    "bright": {
        "name": "MR/CT angiographic",
        "no_vessel_causes": (
            "poor signal-to-noise ratio, out-of-volume cropping, or insufficient flow-velocity "
            "or contrast enhancement"
        ),
        "review": "the study should be reviewed manually on the original 3D multi-planar slices",
        "tree": "contrast/flow-enhanced 3D arterial tree",
        "tortuosity_note": "clinical thresholds for carotid/vertebral tortuosity or kinking are definition-dependent",
        "caveat": (
            "These are automated deterministic measurements calculated across a multi-planar or maximum intensity "
            "reconstruction and do not constitute a clinical diagnosis; overlapping anatomical structures, "
            "magnetic susceptibility artifacts, turbulent flow signal loss, and patient movement can skew diameter calculations. "
            "Direct clinical correlation with expert neuroradiological interpretation and hemodynamic velocity assessment is required."
        ),
    },
    "dark": {
        "name": "X-ray angiographic",
        "no_vessel_causes": (
            "inadequate contrast opacification, a frame captured before contrast arrival or after washout, "
            "over- or under-exposure, or patient motion"
        ),
        "review": "the run should be reviewed manually frame by frame",
        "tree": "contrast-opacified vessel tree",
        "tortuosity_note": "clinical thresholds for tortuosity are territory- and definition-dependent",
        "caveat": (
            "These are automated deterministic measurements on a single 2D projection and do not constitute a "
            "clinical diagnosis; vessel overlap, foreshortening, incomplete contrast filling, catheter or guidewire "
            "shadows, and cardiac, respiratory or patient motion can skew diameter calculations. "
            "Direct clinical correlation with expert interventional or radiological interpretation is required."
        ),
    },
}

def build_observation(
    metrics: dict,
    quality: dict,
    meta: dict,
    unit: str = "mm",
    density: float = 0.0,
    calibrated: bool = True, # Set default to True as MRA datasets include physical spacing
    polarity: str = "bright",
) -> str:
    src = meta.get("label") or Path(meta.get("source", "input")).name
    kind = meta.get("format", "image")
    q = quality.get("label", "adequate")
    mod = MODALITY.get(polarity, MODALITY["bright"])

    if metrics["n_branches"] == 0:
        return (
            f"Automated {mod['name']} analysis of {src} ({kind}; image quality graded {q}) "
            f"did not segment any analysable vessel structure above the detection threshold, so no "
            f"quantitative vascular observation can be issued; this pattern is typically seen with "
            f"{mod['no_vessel_causes']}, and {mod['review']}."
        )

    calib = (
        f"pixel spacing calibrated via DICOM metadata, physical units in {unit}"
        if calibrated
        else "no volume calibration, all measurements computed in voxels/pixels"
    )

    s1 = (
        f"Automated {mod['name']} analysis of {src} ({kind}; image quality graded {q}; {calib}) "
        f"demonstrates a {mod['tree']} occupying {density * 100:.1f}% of the "
        f"analysable field of view, comprising {metrics['n_branches']} analysable centerline "
        f"branch(es) with a total length of {metrics['total_length']:.1f} {unit}, "
        f"{metrics['n_bifurcations']} bifurcation point(s) and {metrics['n_endpoints']} free "
        f"termination(s)."
    )

    s2 = (
        f"Vessel calibre profiling yields a length-weighted mean branch diameter of "
        f"{metrics['mean_branch_diam']:.2f} {unit}, the widest analysable branch averaging "
        f"{metrics['max_branch_diam']:.2f} {unit} against {metrics['min_branch_diam']:.2f} {unit} "
        f"for the narrowest peripheral branch."
    )

    if metrics["mean_tortuosity"] is not None:
        s3 = (
            f"Branch tortuosity, expressed as the centerline arc-to-chord ratio, has a median of "
            f"{metrics['mean_tortuosity']:.2f} and a maximum of {metrics['max_tortuosity']:.2f} "
            f"(a value of 1.0 corresponds to a perfectly straight segment; {mod['tortuosity_note']})."
        )
    else:
        s3 = "Branch tortuosity could not be computed because no branch yielded a finite arc-to-chord ratio."

    thr_pct = int(metrics["stenosis_ratio_threshold"] * 100)
    if metrics["candidate_stenoses"]:
        top = metrics["candidate_stenoses"][0]
        n = len(metrics["candidate_stenoses"])
        s4 = (
            f"Diameter profiling along the vessel lumens identifies {n} candidate focal narrowing(s), "
            f"defined as a local calibre at or below {thr_pct}% of the localized branch reference diameter, "
            f"the most prominent located approximately {top['min_pos_frac'] * 100:.0f}% along the most affected segment "
            f"({top['min_diam']:.2f} {unit} versus {top['ref_diam']:.2f} {unit} reference path)."
        )
    else:
        s4 = (
            f"No focal lumen reduction or severe signal drop below the {thr_pct}% screening threshold "
            f"was identified along the tracked tracking branches."
        )

    s5 = mod["caveat"]
    if q != "adequate":
        s5 += f" Image quality was graded {q}, which further limits measurement confidence."

    return " ".join([s1, s2, s3, s4, s5])


def build_recommendations(checks: dict, metrics: dict, quality: dict) -> str:
    """Deterministic caveats derived from the image sanity checks in vascular.sanity and
    from filters applied during quantification (vascular.quantify). Every sentence here
    is traceable to a specific check result, not a generic disclaimer.
    """
    notes: list[str] = []

    pol = checks.get("polarity") or {}
    if pol.get("source") == "auto" and not pol.get("confident", True):
        appearance = "bright (MR/CT angiography)" if pol.get("polarity") == "bright" else "dark (X-ray/DSA angiography)"
        notes.append(
            f"Vessel appearance could not be determined confidently and the image was analysed assuming "
            f"{appearance} vessels; if that is wrong, re-run with the vessel appearance option set manually."
        )

    panels = checks.get("panels", {})
    artifacts = checks.get("artifacts", [])
    img_h = checks.get("image_height", 0)
    divider = next(
        (a for a in artifacts if a["orientation"] == "vertical" and a["length_px"] >= 0.5 * img_h),
        None,
    )
    if panels.get("multi_panel") or divider is not None:
        if divider:
            where = f"a full-height divider was detected around column {divider['bbox'][1]}"
        elif panels.get("gap"):
            where = f"an empty vertical corridor (columns {panels['gap'][0]}-{panels['gap'][1]}) separates two populated regions"
        else:
            where = "the analysable field splits into separate populated regions"
        notes.append(
            f"This image appears to contain more than one panel or sub-figure ({where}); "
            "analysing a multi-panel figure as a single field of view mixes independent vessel "
            "trees into one set of measurements. Crop each panel to its own image and re-run the "
            "analysis separately for a reliable report."
        )

    if artifacts:
        notes.append(
            f"{len(artifacts)} straight, flat-intensity graphical element(s) consistent with a panel "
            "divider, scale bar, or annotation rule were detected and excluded from the vessel mask "
            "before measurement; confirm none of the reported branches correspond to image furniture "
            "rather than vasculature."
        )

    seams = checks.get("seams", {})
    if seams.get("n_seams"):
        rows = ", ".join(str(r) for r in seams["rows"][:5])
        notes.append(
            f"{seams['n_seams']} near-complete straight line(s) crossing most of the field width were "
            f"detected (row(s) ~{rows}); these are typical of station-junction seams in stitched "
            "multi-station run-off acquisitions and can fragment centerlines or register as spurious "
            "calibre steps."
        )

    labels = checks.get("labels", {})
    if labels.get("n_candidates"):
        notes.append(
            f"{labels['n_candidates']} small, solid, letter- or number-like bright region(s) outside "
            "the vessel mask were detected (e.g. panel labels, orientation markers, or scale "
            "annotations); if any sit on or near the vessel tree they can bias local contrast or be "
            "picked up as a spurious vessel fragment -- consider masking burned-in annotations before "
            "re-analysis."
        )

    n_sup = metrics.get("n_suppressed_small_caliber", 0)
    if n_sup:
        unit = "mm" if metrics.get("spacing_mm") else "px"
        notes.append(
            f"{n_sup} candidate narrowing(s) were suppressed because the local reference calibre was "
            f"below {metrics.get('min_ref_diam', 0):.1f} {unit}: at that scale, pixel noise and "
            "anti-aliasing are not reliably distinguishable from a true stenosis."
        )

    aspect = checks.get("vessel_aspect")
    if aspect and aspect >= 1.6:
        notes.append(
            "The segmented vessel tree spans a tall, elongated field more consistent with a peripheral "
            "run-off (e.g. lower-limb) acquisition than a cervicocranial one; the tortuosity/stenosis "
            "language above uses generic carotid/vertebral phrasing and should be reinterpreted for the "
            "actual anatomical territory, which this pipeline does not identify."
        )

    if quality.get("label") != "adequate":
        notes.append(
            f"Image quality was graded {quality.get('label')}, which independently reduces confidence "
            "in every measurement above."
        )

    if not notes:
        return ""
    return "Recommendations: " + " ".join(notes)
