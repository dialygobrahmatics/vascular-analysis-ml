"""Completeness rules that gate signing (golden rules 2, 3, 6 and 7).

- Every segment [0]-[7] has a status (open / narrowed / blocked / not visible). A segment
  the dye did not reach is "not visible", never "open" by default (rule 3), so nothing is
  pre-filled.
- Every danger sign is answered one by one (rule 6), and every automatic warning is
  acknowledged one by one.
- Side is confirmed (rule 1).
Only then can the doctor sign (rule 7), and only a signed study produces a report.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import rules as rule_tables

SEGMENTS = rule_tables.load("segments")
DANGER = rule_tables.load("danger_signs")
SEGMENT_IDS = [s["id"] for s in SEGMENTS["segments"]]
SEGMENT_STATUS_IDS = {s["id"] for s in SEGMENTS["statuses"]}
DANGER_IDS = [d["id"] for d in DANGER["items"]]
DANGER_ANSWER_IDS = {a["id"] for a in DANGER["answers"]}


def signing_blockers(laterality_status: str, segment_status: dict, danger_answers: dict,
                     warnings: list[dict], findings: list[dict]) -> list[str]:
    """Everything still preventing signature; empty list = ready to sign."""
    out = []
    if laterality_status != "confirmed":
        out.append("Side (left / right) is not confirmed.")
    missing = [str(i) for i in SEGMENT_IDS if segment_status.get(str(i)) not in SEGMENT_STATUS_IDS]
    if missing:
        out.append(f"Segments without a status: [{'], ['.join(missing)}].")
    unanswered = [d["name"] for d in DANGER["items"] if danger_answers.get(d["id"]) not in DANGER_ANSWER_IDS]
    if unanswered:
        out.append("Danger signs not answered: " + "; ".join(unanswered) + ".")
    unack = [w for w in warnings if not w.get("acknowledged")]
    if unack:
        out.append(f"{len(unack)} automatic warning(s) not acknowledged individually.")
    drafts = [f for f in findings if f.get("status") == "draft"]
    if drafts:
        out.append(f"{len(drafts)} draft measurement(s) not yet accepted or rejected.")
    accepted = [f for f in findings if f.get("status", "accepted") == "accepted"]
    unassigned = [f for f in accepted if f.get("segment") in (None, "")]
    if unassigned:
        out.append(f"{len(unassigned)} accepted narrowing(s) not assigned to a segment.")
    narrowed_segments = {str(f["segment"]) for f in accepted if f.get("segment") not in (None, "")}
    contradict = [s for s in narrowed_segments if segment_status.get(s) in ("open", "not_visible")]
    if contradict:
        out.append(f"Segment(s) [{'], ['.join(sorted(contradict))}] have a measured narrowing but are marked open / not visible.")
    return out


def persistent_dye_hint(frames: np.ndarray, best: int, vessel_mask: np.ndarray, dye_dark: bool = True) -> dict | None:
    """Experimental hint for possible dye leak (extravasation), PRD section 10: a dark
    blob that stays after the dye has washed out of the vessels. Compares the last frames
    of the run with the peak frame; never used to clear the checklist item, only to
    raise a warning the doctor must look at."""
    from .bestframe import dye_maps

    n = frames.shape[0]
    if n < 8 or best > n - 4:
        return None
    dye, _ = dye_maps(frames, dye_dark)
    late = dye[max(best + 3, n - max(3, n // 6)):].mean(axis=0)
    peak = dye[best]
    vessel_peak = float(np.percentile(peak[vessel_mask], 75)) if vessel_mask.any() else float(peak.max())
    if vessel_peak <= 0:
        return None
    washed = float(np.median(late[vessel_mask])) / vessel_peak if vessel_mask.any() else 1.0
    if washed > 0.5:
        return None  # dye has not washed out by the end of the run: persistence cannot be judged
    stay = ndi.gaussian_filter(late, 2) > 0.5 * vessel_peak
    lbl, k = ndi.label(stay)
    if k == 0:
        return None
    sizes = ndi.sum(stay, lbl, range(1, k + 1))
    i = int(np.argmax(sizes)) + 1
    if sizes[i - 1] < 30:
        return None
    rr, cc = np.argwhere(lbl == i).mean(axis=0)
    return {"kind": "extravasation", "confidence": "low", "frame": int(n - 1), "position_rc": [round(float(rr)), round(float(cc))],
            "message": "Possible dye persisting after washout (could be a dye leak, or a slow-filling side vessel). Check this run."}
