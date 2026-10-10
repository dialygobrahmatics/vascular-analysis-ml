"""Pixels to millimetres (PRD section 6, golden rule 4: "pixels are not millimetres").

Order of preference, fixed by the PRD:
  1. geometry in the DICOM header (pixel spacing corrected for magnification)
  2. a ruler / marker bands of known size in the image (doctor marks its two ends)
  3. a catheter or sheath from OUR OWN micrometer-measured device list
  never: the size printed on a device label (French size is the inside, brands differ)
  none of these: mm are not shown, only percentages.

Every calibration carries a relative error, and is only "reliable" when that error is
within the rule table's limit. Measurements quote mm only from a reliable calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

DEVICES_FILE = Path(__file__).resolve().parent / "rules" / "devices.json"


@dataclass
class Calibration:
    mm_per_px: float | None
    rel_error: float | None
    method: str  # dicom_calibrated_spacing | dicom_geometry | dicom_magnification | marker | device | none
    reliable: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


NONE = Calibration(None, None, "none", False, "No reliable calibration: millimetres not available, percentages only.")


def from_dicom(info, rules: dict) -> Calibration:
    c = rules["calibration"]
    cal_type = (info.pixel_spacing_calibration_type or "").upper()
    if info.pixel_spacing and cal_type in ("GEOMETRY", "FIDUCIAL"):
        rel = c["calibrated_spacing_rel_error"]
        return _limit(Calibration(info.pixel_spacing[0], rel, "dicom_calibrated_spacing", True,
                                  f"Pixel spacing calibrated by the scanner ({cal_type})."), c)
    ips = info.imager_pixel_spacing
    if ips and info.sid_mm and info.sod_mm and info.sod_mm < info.sid_mm:
        mm = ips[0] * info.sod_mm / info.sid_mm
        # the conversion assumes the arm is at the isocentre; the error is how far it may be
        rel = math.hypot(c["isocentre_offset_mm"] / info.sod_mm, 0.01)
        return _limit(Calibration(mm, rel, "dicom_geometry", True,
                                  f"Detector spacing {ips[0]:.3f} mm x SOD {info.sod_mm:.0f} / SID {info.sid_mm:.0f} mm "
                                  f"(assumes the arm is within {c['isocentre_offset_mm']} mm of the isocentre)."), c)
    if ips and info.magnification_factor and info.magnification_factor > 1:
        rel = c["magnification_factor_rel_error"]
        return _limit(Calibration(ips[0] / info.magnification_factor, rel, "dicom_magnification", True,
                                  f"Detector spacing / estimated magnification {info.magnification_factor:.3f}."), c)
    if ips or info.pixel_spacing:
        return Calibration(None, None, "none", False,
                           "The header has a pixel spacing but no magnification geometry (SID/SOD): sizes at the arm "
                           "are unknown, so millimetres are not shown.")
    return NONE


def from_marker(length_px: float, known_mm: float, rules: dict) -> Calibration:
    c = rules["calibration"]
    if length_px <= 0 or known_mm <= 0:
        return NONE
    rel = math.sqrt(2) * c["marker_endpoint_error_px"] / length_px
    return _limit(Calibration(known_mm / length_px, rel, "marker", True,
                              f"Marker of {known_mm:g} mm spans {length_px:.1f} px."), c)


def load_devices(path: Path = DEVICES_FILE) -> list[dict]:
    """Only devices with a micrometer-measured outer diameter are usable."""
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8")).get("devices", [])
    return [d for d in items if d.get("measured_outer_mm") and d.get("measured_by")]


def from_device(device_id: str, width_px: float, width_err_px: float, rules: dict, devices: list[dict] | None = None) -> Calibration:
    devices = load_devices() if devices is None else [d for d in devices if d.get("measured_outer_mm") and d.get("measured_by")]
    dev = next((d for d in devices if d["id"] == device_id), None)
    if dev is None:
        return Calibration(None, None, "none", False,
                           f"Device '{device_id}' is not in the micrometer-measured device list; label sizes are never used.")
    if width_px <= 0:
        return NONE
    rel = math.hypot(width_err_px / width_px, dev.get("measured_tolerance_mm", 0.02) / dev["measured_outer_mm"])
    return _limit(Calibration(dev["measured_outer_mm"] / width_px, rel, "device", True,
                              f"{dev['name']}: measured outer diameter {dev['measured_outer_mm']} mm spans {width_px:.1f} px."),
                  rules["calibration"])


def _limit(cal: Calibration, c: dict) -> Calibration:
    if cal.rel_error is not None and cal.rel_error > c["max_rel_error"]:
        cal.reliable = False
        cal.detail += f" Error {cal.rel_error * 100:.0f}% exceeds the {c['max_rel_error'] * 100:.0f}% limit: millimetres not shown."
    return cal


def choose(candidates: list[Calibration]) -> Calibration:
    """First reliable calibration in PRD order of preference."""
    order = ["dicom_calibrated_spacing", "dicom_geometry", "dicom_magnification", "marker", "device"]
    reliable = sorted((c for c in candidates if c.reliable and c.mm_per_px), key=lambda c: order.index(c.method))
    if reliable:
        return reliable[0]
    return next((c for c in candidates if c.method == "none" and c.detail != NONE.detail), NONE)
