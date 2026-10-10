"""Reading DICOM studies: group files into runs, order them in time, and pull out the
header tags the rest of the pipeline needs (PRD section 3, requirement R1).

A "run" is one DICOM file holding one X-ray video (or a single spot image). A study is
all runs sharing a StudyInstanceUID.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DICOM_SUFFIXES = {".dcm", ".dicom", ""}


def _get(ds, name, default=None):
    v = getattr(ds, name, None)
    if v is None or v == "":
        return default
    return v


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pair(v):
    if v is None:
        return None
    try:
        a, b = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (a, b) if a > 0 and b > 0 else None


def _time_key(ds) -> str:
    """Sortable acquisition timestamp; falls back through the time tags scanners use."""
    for date_tag, time_tag in (
        ("AcquisitionDate", "AcquisitionTime"),
        ("ContentDate", "ContentTime"),
        ("SeriesDate", "SeriesTime"),
        ("StudyDate", "StudyTime"),
    ):
        t = _get(ds, time_tag)
        if t:
            return f"{_get(ds, date_tag, '')}{str(t).split('.')[0].ljust(6, '0')}{str(t).partition('.')[2].ljust(6, '0')}"
    dt = _get(ds, "AcquisitionDateTime")
    return str(dt) if dt else ""


def is_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


@dataclass
class RunInfo:
    """Header facts about one run. Everything a later step needs is copied out here so the
    pixel data only has to be read when a run is actually analysed."""

    path: str
    study_uid: str
    series_uid: str
    sop_uid: str
    series_number: int | None
    instance_number: int | None
    time_key: str
    modality: str
    n_frames: int
    fps: float | None
    rows: int
    cols: int
    is_dsa: bool
    pixel_spacing: tuple[float, float] | None
    imager_pixel_spacing: tuple[float, float] | None
    pixel_spacing_calibration_type: str
    sid_mm: float | None
    sod_mm: float | None
    magnification_factor: float | None
    primary_angle: float | None
    secondary_angle: float | None
    laterality_tags: dict = field(default_factory=dict)
    description_text: str = ""
    burned_in_annotation: str = ""
    photometric: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _is_dsa(ds) -> bool:
    image_type = [str(x).upper() for x in (_get(ds, "ImageType") or [])]
    if any("SUBTRACT" in x for x in image_type):
        return True
    if _get(ds, "MaskSubtractionSequence") is not None:
        return True
    text = " ".join(str(_get(ds, t, "")) for t in ("SeriesDescription", "ProtocolName", "AcquisitionDeviceProcessingDescription"))
    return bool(re.search(r"\b(DSA|SUBTRACT)", text.upper()))


def read_run_info(path: str | Path) -> RunInfo:
    import pydicom

    path = Path(path)
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    fps = None
    for tag in ("CineRate", "RecommendedDisplayFrameRate"):
        if _float(_get(ds, tag)):
            fps = _float(_get(ds, tag))
            break
    if fps is None and _float(_get(ds, "FrameTime")):
        fps = 1000.0 / _float(_get(ds, "FrameTime"))
    texts = [str(_get(ds, t, "")) for t in ("SeriesDescription", "StudyDescription", "ProtocolName", "PerformedProcedureStepDescription", "BodyPartExamined")]
    return RunInfo(
        path=str(path),
        study_uid=str(_get(ds, "StudyInstanceUID", "unknown-study")),
        series_uid=str(_get(ds, "SeriesInstanceUID", path.stem)),
        sop_uid=str(_get(ds, "SOPInstanceUID", path.stem)),
        series_number=int(_get(ds, "SeriesNumber")) if _get(ds, "SeriesNumber") is not None else None,
        instance_number=int(_get(ds, "InstanceNumber")) if _get(ds, "InstanceNumber") is not None else None,
        time_key=_time_key(ds),
        modality=str(_get(ds, "Modality", "")),
        n_frames=int(_get(ds, "NumberOfFrames", 1) or 1),
        fps=fps,
        rows=int(_get(ds, "Rows", 0) or 0),
        cols=int(_get(ds, "Columns", 0) or 0),
        is_dsa=_is_dsa(ds),
        pixel_spacing=_pair(_get(ds, "PixelSpacing")),
        imager_pixel_spacing=_pair(_get(ds, "ImagerPixelSpacing")),
        pixel_spacing_calibration_type=str(_get(ds, "PixelSpacingCalibrationType", "")).upper(),
        sid_mm=_float(_get(ds, "DistanceSourceToDetector")),
        sod_mm=_float(_get(ds, "DistanceSourceToPatient")),
        magnification_factor=_float(_get(ds, "EstimatedRadiographicMagnificationFactor")),
        primary_angle=_float(_get(ds, "PositionerPrimaryAngle")),
        secondary_angle=_float(_get(ds, "PositionerSecondaryAngle")),
        laterality_tags={
            k: str(_get(ds, k)).strip().upper()
            for k in ("Laterality", "ImageLaterality", "FrameLaterality")
            if _get(ds, k)
        },
        description_text=" | ".join(t for t in texts if t),
        burned_in_annotation=str(_get(ds, "BurnedInAnnotation", "")).upper(),
        photometric=str(_get(ds, "PhotometricInterpretation", "")).upper(),
    )


def find_dicom_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES and is_dicom(p))


def load_study_runs(folder: str | Path) -> dict[str, list[RunInfo]]:
    """All DICOM runs under `folder`, grouped by study and in acquisition order."""
    studies: dict[str, list[RunInfo]] = {}
    for p in find_dicom_files(folder):
        try:
            info = read_run_info(p)
        except Exception:
            continue
        studies.setdefault(info.study_uid, []).append(info)
    for runs in studies.values():
        runs.sort(key=lambda r: (r.time_key or "~", r.series_number or 0, r.instance_number or 0, r.path))
    return studies


def load_frames(path: str | Path) -> np.ndarray:
    """Pixel data of a run as float32 frames scaled to [0, 1], shape (n, rows, cols).

    One scaling for the whole run (not per frame), so a frame before dye arrival keeps
    its real brightness relative to the opacified frames. Display is normalised so that
    higher = brighter, i.e. MONOCHROME1 is inverted.
    """
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(_get(ds, "RescaleSlope", 1.0)) + float(_get(ds, "RescaleIntercept", 0.0))
    spp = int(_get(ds, "SamplesPerPixel", 1) or 1)
    if spp > 1:
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim == 2:
        arr = arr[None]
    if str(_get(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    return np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
