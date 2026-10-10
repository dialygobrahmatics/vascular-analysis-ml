"""Anonymise DICOM studies and build the run inventory (PRD Step 1, requirement D1).

Two layers, because removing the header is not enough:

1. Header: patient / institution / staff identifiers removed, private tags and overlay
   planes dropped, UIDs replaced by stable pseudonyms, dates shifted by a per-patient
   offset (times kept, so runs stay in order). Geometry tags (pixel spacing, SID, SOD,
   angles) and laterality are kept: later steps need them.
2. Pixels: text burned into the image (scanner prints the name in a corner) is found and
   blacked out on every frame. Burned-in text is static across the run and drawn at an
   extreme intensity, which anatomy and dye are not; optional offline OCR (a local
   Tesseract install, never a web service) adds any text regions the rule missed.

The doctor still checks a random sample of runs by eye before the data is used (PRD
Step 1 "done when"); every run gets `needs_visual_check` in the inventory.

Usage:
    python -m access.anonymise <in_dir> <out_dir> --secret-file <path> [--mapping <csv>] [--ocr]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import shutil
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .dicom_io import find_dicom_files, read_run_info

# Header elements removed outright (subset of the DICOM PS3.15 basic confidentiality
# profile that covers what X-ray angiography scanners write).
REMOVE_KEYWORDS = [
    "PatientBirthDate", "PatientBirthTime", "PatientBirthName", "PatientMotherBirthName", "OtherPatientIDs",
    "OtherPatientNames", "OtherPatientIDsSequence", "PatientAddress", "PatientTelephoneNumbers", "PatientInsurancePlanCodeSequence",
    "MilitaryRank", "BranchOfService", "MedicalRecordLocator", "EthnicGroup", "Occupation", "PatientComments",
    "AdditionalPatientHistory", "PatientReligiousPreference", "ResponsiblePerson", "ResponsibleOrganization",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "InstitutionCodeSequence",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord", "OperatorsName",
    "RequestingPhysician", "ScheduledPerformingPhysicianName", "AccessionNumber", "StudyID", "StationName",
    "DeviceSerialNumber", "RequestAttributesSequence", "AdmissionID", "IssuerOfPatientID", "CurrentPatientLocation",
    "PatientState", "ImageComments", "RequestedProcedureID", "PerformedProcedureStepID", "ScheduledProcedureStepID",
    "FillerOrderNumberImagingServiceRequest", "PlacerOrderNumberImagingServiceRequest", "PersonName",
    "ContentCreatorName", "ReviewerName", "VerifyingObserverName", "TextComments", "DerivationDescription",
]
DATE_KEYWORDS = ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "PerformedProcedureStepStartDate", "InstanceCreationDate"]
UID_KEYWORDS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID", "MediaStorageSOPInstanceUID"]

TEXT_STATIC_FRAC = 0.02  # temporal range below this share of the value range = static pixel
TEXT_EXTREME_PCT = 99.7  # burned-in text is drawn at (near) full intensity
TEXT_MAX_HEIGHT_FRAC = 0.12
TEXT_PAD_PX = 3
DEIDENT_METHOD = "Dialygo-Access v1: basic profile, UIDs, dates, pixel text"  # LO: max 64 chars


def _key(secret: bytes, value: str) -> str:
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


def pseudonym(secret: bytes, patient_id: str) -> str:
    return "DA-" + _key(secret, "patient:" + patient_id)[:10].upper()


def pseudo_uid(secret: bytes, uid: str) -> str:
    # 2.25.<integer> is the standard UUID-derived UID form; deterministic per input
    return "2.25." + str(int(_key(secret, "uid:" + uid)[:30], 16))


def date_offset_days(secret: bytes, patient_id: str) -> int:
    """Same shift for every date of one patient, so intervals between visits survive."""
    return 30 + int(_key(secret, "date:" + patient_id)[:8], 16) % 3000


def _shift_date(value: str, days: int) -> str:
    try:
        d = date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except (ValueError, TypeError):
        return ""
    return (d - timedelta(days=days)).strftime("%Y%m%d")


def anonymise_header(ds, secret: bytes) -> dict:
    """Edit `ds` in place; returns the pseudonyms used (for the private mapping file)."""
    original_pid = str(getattr(ds, "PatientID", "") or getattr(ds, "PatientName", "") or "unknown")
    pid = pseudonym(secret, original_pid)
    shift = date_offset_days(secret, original_pid)
    old_study = str(getattr(ds, "StudyInstanceUID", ""))

    ds.remove_private_tags()
    for group in range(0x6000, 0x6020, 2):  # overlay planes can carry burned-in style text
        for elem in [e for e in ds if e.tag.group == group]:
            del ds[elem.tag]

    def walk(dataset):
        for elem in list(dataset):
            if elem.VR == "SQ":
                for item in elem.value:
                    walk(item)
            elif elem.VR == "PN" and elem.keyword not in ("PatientName",):
                del dataset[elem.tag]
        for kw in REMOVE_KEYWORDS:
            if kw in dataset:
                delattr(dataset, kw)

    walk(ds)
    ds.PatientName = pid
    ds.PatientID = pid
    for kw in DATE_KEYWORDS:
        if kw in ds and ds.data_element(kw).value:
            setattr(ds, kw, _shift_date(str(ds.data_element(kw).value), shift))
    if "AcquisitionDateTime" in ds and ds.AcquisitionDateTime:
        v = str(ds.AcquisitionDateTime)
        ds.AcquisitionDateTime = _shift_date(v[:8], shift) + v[8:]
    for kw in UID_KEYWORDS:
        if kw in ds and ds.data_element(kw).value:
            setattr(ds, kw, pseudo_uid(secret, str(ds.data_element(kw).value)))
    if hasattr(ds, "file_meta") and "MediaStorageSOPInstanceUID" in ds.file_meta:
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = DEIDENT_METHOD
    return {"original_patient_id": original_pid, "pseudonym": pid, "original_study_uid": old_study,
            "study_uid": str(getattr(ds, "StudyInstanceUID", "")), "date_shift_days": shift}


def _ocr_boxes(frame_u8: np.ndarray, tesseract: str) -> list[tuple[int, int, int, int]]:
    """Word boxes from a local Tesseract executable (offline). Only the box positions are
    kept; the recognised text itself is discarded so no identifier is written anywhere."""
    with tempfile.TemporaryDirectory() as tmp:
        import cv2

        img = Path(tmp) / "f.png"
        cv2.imwrite(str(img), frame_u8)
        try:
            out = subprocess.run([tesseract, str(img), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return []
    boxes = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 12 and parts[11].strip() and float(parts[10] or -1) >= 50:
            x, y, w, h = map(int, parts[6:10])
            boxes.append((y, x, y + h, x + w))
    return boxes


def find_text_regions(arr: np.ndarray, use_ocr: str | None = None) -> list[tuple[int, int, int, int]]:
    """Boxes (r0, c0, r1, c1) of burned-in text in a run, shape (n, rows, cols)."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    rng = max(hi - lo, 1e-6)
    med = np.median(arr, axis=0) if arr.shape[0] > 1 else arr[0]
    static = (arr.max(axis=0) - arr.min(axis=0)) <= TEXT_STATIC_FRAC * rng if arr.shape[0] > 2 else np.ones(med.shape, bool)
    thr = max(float(np.percentile(med, TEXT_EXTREME_PCT)), lo + 0.9 * rng)
    cand = static & (med >= thr)
    h, w = med.shape
    # join letters of one word/line: text sits in horizontal runs
    joined = ndi.binary_closing(ndi.binary_dilation(cand, structure=np.ones((3, 3))), structure=np.ones((3, 11)))
    lbl, n = ndi.label(joined)
    boxes = []
    for sl in ndi.find_objects(lbl):
        if sl is None:
            continue
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        strokes = cand[sl].mean()
        if 4 <= bh <= TEXT_MAX_HEIGHT_FRAC * h and bw >= 6 and bw >= 0.8 * bh and 0.03 <= strokes <= 0.8 and bh * bw < 0.05 * h * w:
            boxes.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
    if use_ocr:
        u8 = ((med - lo) / rng * 255).astype(np.uint8)
        boxes += _ocr_boxes(u8, use_ocr)
    return [(max(0, r0 - TEXT_PAD_PX), max(0, c0 - TEXT_PAD_PX), min(h, r1 + TEXT_PAD_PX), min(w, c1 + TEXT_PAD_PX)) for r0, c0, r1, c1 in boxes]


def blackout(arr: np.ndarray, boxes) -> np.ndarray:
    out = arr.copy()
    fill = arr.min()
    for r0, c0, r1, c1 in boxes:
        out[..., r0:r1, c0:c1] = fill if out.ndim <= 3 else fill
    return out


def anonymise_file(src: Path, dst: Path, secret: bytes, use_ocr: str | None = None) -> dict:
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian

    ds = pydicom.dcmread(str(src), force=True)
    arr = ds.pixel_array
    color = arr.ndim == 4 or (arr.ndim == 3 and int(getattr(ds, "SamplesPerPixel", 1)) > 1)
    stack = arr if arr.ndim == (4 if color else 3) else arr[None]
    gray = stack.max(axis=-1) if color else stack
    boxes = find_text_regions(gray, use_ocr)
    cleaned = blackout(stack, boxes)
    cleaned = cleaned if arr.ndim == stack.ndim else cleaned[0]
    mapping = anonymise_header(ds, secret)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric.startswith("YBR"):
        photometric = "RGB"  # pixel_array is already converted to RGB on read
    ds.set_pixel_data(cleaned, photometric, int(getattr(ds, "BitsStored", cleaned.dtype.itemsize * 8)))
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dst), enforce_file_format=True)
    return {**mapping, "text_regions": len(boxes), "boxes": boxes}


INVENTORY_FIELDS = [
    "study_pseudonym", "study_uid", "series_number", "file", "frames", "fps", "dsa_or_cine", "rows", "cols",
    "pixel_spacing_present", "geometry_present", "side_header", "text_regions_blacked", "needs_visual_check",
]


def inventory_row(path: Path, out_root: Path, text_regions: int, pseudonym_id: str) -> dict:
    info = read_run_info(path)
    side = sorted(set(info.laterality_tags.values()))
    return {
        "study_pseudonym": pseudonym_id,
        "study_uid": info.study_uid,
        "series_number": info.series_number,
        "file": str(path.relative_to(out_root)),
        "frames": info.n_frames,
        "fps": info.fps or "",
        "dsa_or_cine": "DSA" if info.is_dsa else ("spot" if info.n_frames == 1 else "cine"),
        "rows": info.rows,
        "cols": info.cols,
        "pixel_spacing_present": "yes" if (info.pixel_spacing or info.imager_pixel_spacing) else "no",
        "geometry_present": "yes" if (info.sid_mm and info.sod_mm) else "no",
        "side_header": "/".join(side) if side else "",
        "text_regions_blacked": text_regions,
        "needs_visual_check": "yes",
    }


def anonymise_folder(src_dir: str | Path, out_dir: str | Path, secret: bytes, mapping_csv: str | Path | None = None,
                     use_ocr: str | None = None, progress=print) -> list[dict]:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    files = find_dicom_files(src_dir)
    rows, mapping_rows, failed = [], [], []
    for i, f in enumerate(files, 1):
        try:
            pre = read_run_info(f)
            study_pseudo = pseudo_uid(secret, pre.study_uid)
            dst = out_dir / study_pseudo / f"series{pre.series_number or 0:03d}_{_key(secret, pre.sop_uid)[:12]}.dcm"
            res = anonymise_file(f, dst, secret, use_ocr)
            rows.append(inventory_row(dst, out_dir, res["text_regions"], res["pseudonym"]))
            mapping_rows.append({k: res[k] for k in ("original_patient_id", "pseudonym", "original_study_uid", "study_uid", "date_shift_days")}
                                | {"source_file": str(f)})
        except Exception as exc:  # keep going; the failure list is reported
            failed.append((str(f), str(exc)))
        progress(f"[{i}/{len(files)}] {f.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "inventory.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    if mapping_csv:
        # links pseudonyms back to real patients: stays on the hospital disk, never with the data
        with open(mapping_csv, "w", newline="", encoding="utf-8") as fh:
            uniq = {(r["original_patient_id"], r["original_study_uid"]): r for r in mapping_rows}.values()
            wr = csv.DictWriter(fh, fieldnames=list(next(iter(uniq)).keys()) if mapping_rows else ["pseudonym"])
            wr.writeheader()
            wr.writerows(uniq)
    if failed:
        with open(out_dir / "failed.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([("file", "error"), *failed])
    return rows


def load_or_create_secret(path: str | Path) -> bytes:
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    import secrets

    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    return key


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Anonymise a folder of DICOM studies and write inventory.csv.")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--secret-file", required=True, help="key for pseudonyms; created if missing. Keep it private and keep it the same between batches")
    ap.add_argument("--mapping", help="private CSV linking pseudonyms to original IDs (keep on the hospital disk)")
    ap.add_argument("--ocr", action="store_true", help="also use a local Tesseract install to find text regions")
    a = ap.parse_args(argv)
    tesseract = None
    if a.ocr:
        tesseract = shutil.which("tesseract") or next((p for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",) if Path(p).exists()), None)
        if not tesseract:
            print("Tesseract not found; continuing with the rule-based text detector only.")
    rows = anonymise_folder(a.src, a.out, load_or_create_secret(a.secret_file), a.mapping, tesseract)
    print(f"Done: {len(rows)} runs written to {a.out}; inventory at {Path(a.out) / 'inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
