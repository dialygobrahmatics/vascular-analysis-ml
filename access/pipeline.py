"""Import a study and prepare the doctor's drafts.

Import (any time): read every run, cache its frames, pick the best frame, draft the
vessel and device outlines, work out the calibration, collect the side from the headers
and raise automatic warnings.

Measurement (only after the side is confirmed, golden rule 1): measure the main vessel of
every run and store the narrowings as *draft* findings for the doctor to accept, correct
or reject. Nothing is final until the doctor signs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import calibration as cal
from . import rules as rule_tables
from .bestframe import pick_best_frame
from .checklist import persistent_dye_hint
from .db import DB
from .dicom_io import load_frames, load_study_runs
from .draft_seg import draft_masks
from .laterality import header_sources, resolve
from .measure import analyse_profile, attach_dye, measure_vessel, profile_from_json, profile_to_json

SYSTEM = "system"
MAX_CACHE_SIDE = 1024


def run_cache(cache_dir: Path) -> dict:
    """Arrays of one analysed run (frames, outlines, dye map) from its cache folder."""
    d = np.load(cache_dir / "analysis.npz")
    return {k: d[k] for k in d.files}


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def analyse_run(info, cache_dir: Path, rules: dict) -> dict:
    frames = load_frames(info.path)
    n, h, w = frames.shape
    scale = min(1.0, MAX_CACHE_SIDE / max(h, w))
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))
        frames = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
    best = pick_best_frame(frames)
    seg = draft_masks(frames, best.index)
    calib = cal.from_dicom(info, rules)
    if scale < 1.0 and calib.mm_per_px:
        calib.mm_per_px /= scale
        calib.detail += f" (image shown at {scale:.2f}x)"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "analysis.npz", frames=(frames * 255).astype(np.uint8), vessel=seg["vessel"],
                        device=seg["device"], dye=seg["dye_raw"].astype(np.float32))
    _save_mask_png(cache_dir / "vessel.png", seg["vessel"])
    _save_mask_png(cache_dir / "device.png", seg["device"])
    hint = persistent_dye_hint(frames, best.index, seg["vessel"])
    return {"best": best.to_dict(), "calibration": calib.to_dict(), "scale": scale, "notes": seg["notes"],
            "method": seg["method"], "hint": hint}


def import_study(db: DB, folder: str | Path, data_dir: Path, progress=lambda msg, frac: None) -> list[int]:
    """Import every study found under `folder`; returns study ids."""
    rules = rule_tables.load("measurement_rules")
    studies = load_study_runs(folder)
    if not studies:
        raise ValueError("No DICOM files found in that folder.")
    import pydicom

    not_anon = [r.path for runs in studies.values() for r in runs
                if str(getattr(pydicom.dcmread(r.path, stop_before_pixels=True, force=True), "PatientIdentityRemoved", "")).upper() != "YES"]
    if not_anon:
        # PRD section 3: work only on the anonymised copy
        raise ValueError(f"{len(not_anon)} file(s) are not anonymised (PatientIdentityRemoved is not YES). "
                         "Anonymise them first with: python -m access.anonymise <in> <out> --secret-file <key>")
    ids = []
    for uid, runs in studies.items():
        existing = db.one("SELECT id FROM studies WHERE uid = ?", (uid,))
        if existing:
            ids.append(existing["id"])
            continue
        pseudo = str(getattr(pydicom.dcmread(runs[0].path, stop_before_pixels=True, force=True), "PatientID", "") or "")
        sid = db.execute("INSERT INTO studies (uid, pseudonym, folder, imported_at, state) VALUES (?,?,?,?,?)",
                         (uid, pseudo, str(folder), time.time(), "importing"))
        db.audit(SYSTEM, sid, "studies", sid, "import", None, {"folder": str(folder), "runs": len(runs)})
        ids.append(sid)
        for k, info in enumerate(runs):
            progress(f"Analysing run {k + 1} of {len(runs)}", k / len(runs))
            cache = data_dir / "cache" / f"study{sid}" / f"run{k + 1}"
            res = analyse_run(info, cache, rules)
            rid = db.execute(
                "INSERT INTO runs (study_id, idx, path, info, best, calibration, cache_dir, draft_notes) VALUES (?,?,?,?,?,?,?,?)",
                (sid, k + 1, info.path, info.to_dict() | {"display_scale": res["scale"]}, res["best"], res["calibration"],
                 str(cache), {"segmentation": res["method"], "notes": res["notes"]}))
            if res["hint"]:
                db.execute("INSERT INTO warnings (study_id, run_id, kind, message, data) VALUES (?,?,?,?,?)",
                           (sid, rid, res["hint"]["kind"], f"Run {k + 1}: " + res["hint"]["message"], res["hint"]))
            if any(res["best"]["moved"]):
                db.execute("INSERT INTO warnings (study_id, run_id, kind, message, data) VALUES (?,?,?,?,?)",
                           (sid, rid, "motion", f"Run {k + 1}: patient motion in {sum(res['best']['moved'])} frame(s)"
                            + (" - DSA subtraction may show false shapes there." if info.is_dsa else "."), None))
        lat = resolve(header_sources(runs))
        db.execute("UPDATE studies SET laterality = ?, state = ? WHERE id = ?", (lat, "ready", sid))
        db.audit(SYSTEM, sid, "studies", sid, "laterality", None, lat)
    progress("Done", 1.0)
    return ids


def study_runs_info(db: DB, study_id: int):
    from .dicom_io import RunInfo

    runs = db.query("SELECT * FROM runs WHERE study_id = ? ORDER BY idx", (study_id,))
    infos = []
    for r in runs:
        d = {k: v for k, v in r["info"].items() if k in RunInfo.__dataclass_fields__}
        if d.get("pixel_spacing"):
            d["pixel_spacing"] = tuple(d["pixel_spacing"])
        if d.get("imager_pixel_spacing"):
            d["imager_pixel_spacing"] = tuple(d["imager_pixel_spacing"])
        infos.append(RunInfo(**d))
    return runs, infos


def current_laterality(db: DB, study: dict) -> dict:
    runs, infos = study_runs_info(db, study["id"])
    return resolve(header_sources(infos), study.get("note_side"), study.get("doctor_side"), study.get("override_reason"))


def calibration_of(run: dict) -> cal.Calibration:
    c = run["calibration"] or {}
    return cal.Calibration(c.get("mm_per_px"), c.get("rel_error"), c.get("method", "none"), bool(c.get("reliable")),
                           c.get("detail", ""))


def measure(db: DB, run: dict, username: str, start=None, end=None, segment=None, mld_index=None, ref_indices=None,
            finding_id: int | None = None, status: str = "draft") -> dict:
    """Measure along a path and store / update the finding."""
    rules = rule_tables.load("measurement_rules")
    arrays = run_cache(Path(run["cache_dir"]))
    vessel = arrays["vessel"].astype(bool)
    res = measure_vessel(vessel, rules, calibration_of(run), dye=arrays["dye"], start=start, end=end,
                         segment=segment, device_mask=arrays["device"].astype(bool), mld_index=mld_index, ref_indices=ref_indices)
    if not res["ok"]:
        return {"ok": False, "notes": res.get("notes") or res.get("finding", {}).get("notes", [])}
    data = res["finding"] | {"profile": profile_to_json(res["profile"]), "frame": run["best"]["index"], "run_idx": run["idx"],
                             "candidates": res["candidates"]}
    inputs = {"start": start, "end": end, "mld_index": mld_index, "ref_indices": ref_indices}
    seg = None if segment in (None, "") else str(segment)
    if finding_id is None:
        fid = db.execute("INSERT INTO findings (study_id, run_id, segment, status, data, inputs, created_by, updated_at) "
                         "VALUES (?,?,?,?,?,?,?,?)", (run["study_id"], run["id"], seg, status, data, inputs, username, time.time()))
        db.audit(username, run["study_id"], "findings", fid, "created", None, _summary(data))
    else:
        fid = finding_id
        old = db.one("SELECT data FROM findings WHERE id = ?", (fid,))
        db.execute("UPDATE findings SET data = ?, inputs = ?, updated_at = ? WHERE id = ?", (data, inputs, time.time(), fid))
        db.audit(username, run["study_id"], "findings", fid, "measurement", _summary(old["data"]) if old else None, _summary(data))
    return {"ok": True, "id": fid, "finding": data}


def remeasure_points(db: DB, finding: dict, run: dict, username: str, mld_index=None, ref_indices=None) -> dict:
    """Doctor moved the MLD or reference points: recompute from the stored profile (fast)."""
    rules = rule_tables.load("measurement_rules")
    prof = profile_from_json(finding["data"]["profile"])
    arrays = run_cache(Path(run["cache_dir"]))
    attach_dye(prof, arrays["dye"], rules)  # so moved points get the full model fit too
    out = analyse_profile(prof, rules, calibration_of(run), finding["segment"], mld_index, ref_indices,
                          arrays["device"].astype(bool))
    data = finding["data"] | out | {"profile": profile_to_json(prof)}
    inputs = finding["inputs"] | {"mld_index": mld_index, "ref_indices": ref_indices}
    db.execute("UPDATE findings SET data = ?, inputs = ?, updated_at = ? WHERE id = ?", (data, inputs, time.time(), finding["id"]))
    db.audit(username, run["study_id"], "findings", finding["id"], "points", _summary(finding["data"]), _summary(data))
    return {"ok": True, "id": finding["id"], "finding": data}


def _summary(d: dict | None) -> dict | None:
    if not d:
        return None
    return {k: d.get(k) for k in ("mld_mm", "rvd_mm", "ds_percent", "length_mm", "mld_px", "rvd_px", "mld_index", "segment", "confidence")}


def draft_measurements(db: DB, study_id: int) -> int:
    """Automatic first measurements, run once the side is confirmed."""
    n = 0
    for run in db.query("SELECT * FROM runs WHERE study_id = ? ORDER BY idx", (study_id,)):
        res = measure(db, run, SYSTEM)
        if res.get("ok"):
            n += 1
            f = res["finding"]
            if f.get("ok") and not f.get("narrowing_found"):
                db.update_field("findings", res["id"], "status", "no_narrowing", SYSTEM, study_id)
    db.execute("UPDATE studies SET measured = 1 WHERE id = ?", (study_id,))
    return n
