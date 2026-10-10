"""Generate a synthetic dialysis-access DICOM study with a known answer key.

No real patient data is available to develop against, so this builds studies that look
enough like the real thing to exercise every step, with exact ground truth:

- vessels drawn from cylinder geometry (projected chord length), so MLD / RVD / %DS /
  length are known exactly
- a static catheter (a device that must not be read as vessel)
- dye arriving and washing out over the run (known peak frame)
- fake patient name/ID burned into the pixels and in the header (anonymiser must remove)
- X-ray geometry tags (ImagerPixelSpacing, SID, SOD), laterality tags and acquisition times

Usage:
    python tools/make_synthetic_study.py <out_dir> [--seed 0]
Writes <out_dir>/run1_cine.dcm, run2_dsa.dcm, run3_post.dcm and <out_dir>/truth.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

H, W = 512, 512
IMAGER_SPACING_MM = 0.30
SID_MM, SOD_MM = 1000.0, 800.0
MM_PER_PX = IMAGER_SPACING_MM * SOD_MM / SID_MM  # 0.24 mm/px at the arm


def centreline(kind: str) -> np.ndarray:
    """Dense (row, col) centreline points, ~0.25 px apart."""
    t = np.linspace(0, 1, 4000)
    if kind == "arm":
        rows = 40 + t * 430
        cols = 180 + 90 * np.sin(t * np.pi * 1.1) + 30 * t
    else:
        rows = 470 - t * 420
        cols = 120 + 260 * t + 25 * np.sin(t * np.pi * 3)
    return np.stack([rows, cols], axis=1)


def radius_profile(n: int, rvd_mm: float, mld_mm: float, sten_center: float, sten_len_mm: float,
                   arc_len_px: float, aneurysm: tuple[float, float, float] | None = None) -> np.ndarray:
    """Radius (px) along the centreline: constant RVD, a smooth cosine-shaped narrowing
    of total length `sten_len_mm` down to MLD, and optionally an aneurysm bulge
    (center fraction, length mm, diameter mm)."""
    s_mm = np.linspace(0, arc_len_px * MM_PER_PX, n)
    d = np.full(n, rvd_mm)
    c_mm = sten_center * s_mm[-1]
    half = sten_len_mm / 2
    inside = np.abs(s_mm - c_mm) < half
    d[inside] = rvd_mm - (rvd_mm - mld_mm) * 0.5 * (1 + np.cos(np.pi * (s_mm[inside] - c_mm) / half))
    if aneurysm:
        ac, alen, adiam = aneurysm
        a_mm = ac * s_mm[-1]
        ins = np.abs(s_mm - a_mm) < alen / 2
        d[ins] = np.maximum(d[ins], rvd_mm + (adiam - rvd_mm) * 0.5 * (1 + np.cos(np.pi * (s_mm[ins] - a_mm) / (alen / 2))))
    return d / 2 / MM_PER_PX


def chord_map(pts: np.ndarray, radius_px: np.ndarray) -> np.ndarray:
    """Projected path length (px) through the vessel at every pixel."""
    yy, xx = np.mgrid[0:H, 0:W]
    tree = cKDTree(pts)
    dist, idx = tree.query(np.stack([yy.ravel(), xx.ravel()], axis=1), distance_upper_bound=60)
    chord = np.zeros(H * W)
    ok = np.isfinite(dist)
    r = radius_px[np.clip(idx[ok], 0, len(radius_px) - 1)]
    chord[ok] = 2 * np.sqrt(np.clip(r ** 2 - dist[ok] ** 2, 0, None))
    return chord.reshape(H, W)


def arc_length(pts: np.ndarray) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def render_run(kind: str, n_frames: int, peak: int, rvd: float, mld: float, sten_len: float,
               sten_center: float, dsa: bool, rng, aneurysm=None, catheter=True):
    pts = centreline(kind)
    L = arc_length(pts)
    rad = radius_profile(len(pts), rvd, mld, sten_center, sten_len, L, aneurysm)
    chord = chord_map(pts, rad)
    bg = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 30)
    bg = 0.55 + 0.12 * (bg - bg.min()) / (np.ptp(bg) + 1e-9)
    bone = np.exp(-((np.arange(W) - 400) / 35.0) ** 2)[None, :] * 0.18
    static = bg if dsa else bg + bone
    if catheter and not dsa:
        cat = np.zeros((H, W), np.float32)
        cv2.line(cat, (60, 500), (300, 90), 1.0, 3)
        static = static - 0.25 * ndi.gaussian_filter(cat, 0.8)
    # dye curve: arrives, peaks at `peak`, washes out
    t = np.arange(n_frames)
    dye = np.exp(-0.5 * ((t - peak) / (n_frames / 6)) ** 2)
    dye[t < peak - n_frames / 3] = 0
    mu = 0.012  # intensity drop per px of contrast path
    frames = []
    for i in range(n_frames):
        base = np.full((H, W), 0.85) if dsa else static
        img = base - mu * dye[i] * chord
        img = ndi.gaussian_filter(img, 0.7) + rng.normal(0, 0.008, (H, W))
        frames.append(np.clip(img, 0, 1))
    return np.stack(frames), pts, rad, L


TEXT = [("RAMESH KUMAR", (12, 24), 0.6), ("ID 4471823  DOB 12/03/1961", (12, 46), 0.45), ("ST JOHNS HOSPITAL", (300, 500), 0.5)]


def text_mask() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)  # cv2.putText only draws on 8-bit images
    for s, org, scale in TEXT:
        cv2.putText(m, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    return m.astype(np.float32) / 255.0


def burn_text(frames: np.ndarray) -> np.ndarray:
    a = text_mask()
    return np.stack([((f * (1 - a) + a) * 4095).astype(np.uint16) for f in frames])


def write_dicom(path: Path, pixels: np.ndarray, *, series: int, time: str, desc: str, dsa: bool, laterality: str, study_uid: str):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"  # X-Ray Angiographic Image
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, generate_uid()
    ds.PatientName, ds.PatientID, ds.PatientBirthDate, ds.PatientSex = "KUMAR^RAMESH", "4471823", "19610312", "M"
    ds.InstitutionName, ds.ReferringPhysicianName, ds.AccessionNumber = "St Johns Hospital", "DR^SHARMA", "ACC99812"
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20260901"
    ds.StudyTime, ds.SeriesTime, ds.AcquisitionTime = "101500", time, time
    ds.Modality, ds.SeriesNumber, ds.InstanceNumber = "XA", series, 1
    ds.SeriesDescription, ds.StudyDescription = desc, "FISTULOGRAM"
    ds.BodyPartExamined = "ARM"
    if laterality:
        ds.Laterality = laterality
    ds.ImageType = ["ORIGINAL", "PRIMARY", "SINGLE PLANE", "SUBTRACTION"] if dsa else ["ORIGINAL", "PRIMARY", "SINGLE PLANE"]
    ds.ImagerPixelSpacing = [IMAGER_SPACING_MM, IMAGER_SPACING_MM]
    ds.DistanceSourceToDetector, ds.DistanceSourceToPatient = SID_MM, SOD_MM
    ds.PositionerPrimaryAngle, ds.PositionerSecondaryAngle = 0, 0
    ds.BurnedInAnnotation = "YES"
    ds.CineRate, ds.FrameTime = 15, 1000 / 15
    ds.Rows, ds.Columns = pixels.shape[1:]
    ds.NumberOfFrames = pixels.shape[0]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def make_study(out: str | Path, seed: int = 0) -> dict:
    from pydicom.uid import generate_uid

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    study_uid = generate_uid()
    specs = [
        # name, kind, frames, peak, RVD, MLD, length, centre, dsa, aneurysm, time, desc
        ("run1_cine", "arm", 30, 14, 6.0, 2.1, 12.0, 0.55, False, None, "102000", "LEFT ARM FISTULOGRAM"),
        ("run2_dsa", "upper", 24, 10, 7.0, 3.5, 10.0, 0.70, True, (0.30, 14.0, 11.0), "102300", "LEFT UPPER ARM DSA"),
        ("run3_post", "arm", 30, 15, 6.0, 5.0, 12.0, 0.55, False, None, "103500", "LEFT ARM POST PTA"),
    ]
    truth = {"mm_per_px": MM_PER_PX, "laterality": "LEFT", "runs": {}}
    for i, (name, kind, n, peak, rvd, mld, length, centre, dsa, aneur, time, desc) in enumerate(specs, 1):
        frames, pts, rad, L = render_run(kind, n, peak, rvd, mld, length, centre, dsa, rng, aneurysm=aneur)
        write_dicom(out / f"{name}.dcm", burn_text(frames), series=i, time=time, desc=desc, dsa=dsa,
                    laterality="L", study_uid=study_uid)
        truth["runs"][name] = {
            "peak_frame": peak, "rvd_mm": rvd, "mld_mm": mld, "length_mm": length,
            "ds_percent": round((1 - mld / rvd) * 100, 1), "is_dsa": dsa,
            "aneurysm": None if aneur is None else {"diameter_mm": aneur[2], "length_mm": aneur[1]},
        }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(make_study(a.out, a.seed), indent=2))
