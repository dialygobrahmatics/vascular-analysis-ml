"""Laterality, signing checklist, notes parsing, calibration order, anonymisation."""

from pathlib import Path

import numpy as np
import pytest

from access import calibration as cal
from access import rules as rule_tables
from access.checklist import DANGER_IDS, SEGMENT_IDS, signing_blockers
from access.dicom_io import RunInfo
from access.laterality import decide, header_sources, resolve
from access.notes import parse_note

RULES = rule_tables.load("measurement_rules")


def run(**kw):
    base = dict(path="x.dcm", study_uid="s", series_uid="se", sop_uid="so", series_number=1, instance_number=1,
                time_key="1", modality="XA", n_frames=10, fps=15.0, rows=512, cols=512, is_dsa=False, pixel_spacing=None,
                imager_pixel_spacing=None, pixel_spacing_calibration_type="", sid_mm=None, sod_mm=None,
                magnification_factor=None, primary_angle=None, secondary_angle=None, laterality_tags={}, description_text="")
    base.update(kw)
    return RunInfo(**base)


# ---- laterality (golden rule 1) ----

def test_header_counts_as_one_source():
    votes = header_sources([run(laterality_tags={"Laterality": "L"}, description_text="LEFT ARM FISTULOGRAM")])
    assert len(votes) == 1 and decide(votes)["status"] == "needs_confirmation"


def test_header_and_note_agree_confirm():
    votes = header_sources([run(laterality_tags={"Laterality": "L"}, description_text="LEFT ARM FISTULOGRAM")])
    r = resolve(votes, note_side="LEFT")
    assert r["status"] == "confirmed" and r["side"] == "LEFT"


def test_header_contradicting_itself_is_a_conflict():
    votes = header_sources([run(laterality_tags={"Laterality": "L"}, description_text="RIGHT ARM")])
    assert decide(votes)["status"] == "conflict"


def test_disagreement_stops():
    votes = header_sources([run(laterality_tags={"Laterality": "L"})])
    r = resolve(votes, note_side="RIGHT")
    assert r["status"] == "conflict" and r["side"] is None


def test_single_source_needs_confirmation_then_doctor_confirms():
    votes = header_sources([run(laterality_tags={"Laterality": "R"})])
    assert decide(votes)["status"] == "needs_confirmation"
    assert resolve(votes, doctor_side="RIGHT")["status"] == "confirmed"


def test_conflict_needs_reason_to_override():
    votes = header_sources([run(laterality_tags={"Laterality": "L"}, series_number=1),
                            run(laterality_tags={"Laterality": "R"}, series_number=2)])
    assert resolve(votes, doctor_side="LEFT")["status"] == "conflict"
    r = resolve(votes, doctor_side="LEFT", doctor_override_reason="series 2 mislabelled at the console")
    assert r["status"] == "confirmed" and r["override"]


def test_no_side_is_never_guessed():
    assert decide(header_sources([run()]))["status"] == "unknown"


# ---- signing checklist (golden rules 2, 3, 6, 7) ----

def complete_state():
    return dict(laterality_status="confirmed", segment_status={str(s): "open" for s in SEGMENT_IDS},
                danger_answers={d: "absent" for d in DANGER_IDS}, warnings=[], findings=[])


def test_complete_study_can_sign():
    assert signing_blockers(**complete_state()) == []


def test_missing_segment_blocks():
    st = complete_state()
    del st["segment_status"]["7"]
    assert any("[7]" in b for b in signing_blockers(**st))


def test_unanswered_danger_blocks_and_warnings_need_individual_ack():
    st = complete_state()
    st["danger_answers"].pop("extravasation")
    st["warnings"] = [{"acknowledged": True}, {"acknowledged": False}]
    blockers = signing_blockers(**st)
    assert any("Danger signs" in b for b in blockers) and any("warning" in b for b in blockers)


def test_finding_in_segment_marked_open_blocks():
    st = complete_state()
    st["findings"] = [{"segment": "3"}]
    assert any("[3]" in b for b in signing_blockers(**st))


def test_unconfirmed_side_blocks():
    st = complete_state()
    st["laterality_status"] = "conflict"
    assert signing_blockers(**st)


# ---- calibration order (PRD section 6) ----

def test_geometry_calibration():
    c = cal.from_dicom(run(imager_pixel_spacing=(0.3, 0.3), sid_mm=1000, sod_mm=800), RULES)
    assert c.method == "dicom_geometry" and c.reliable and abs(c.mm_per_px - 0.24) < 1e-9


def test_detector_spacing_without_geometry_is_not_millimetres():
    c = cal.from_dicom(run(imager_pixel_spacing=(0.3, 0.3)), RULES)
    assert not c.reliable and c.mm_per_px is None


def test_geometry_preferred_over_marker():
    geo = cal.from_dicom(run(imager_pixel_spacing=(0.3, 0.3), sid_mm=1000, sod_mm=800), RULES)
    marker = cal.from_marker(400, 100, RULES)
    assert cal.choose([marker, geo]).method == "dicom_geometry"


def test_short_marker_is_unreliable():
    assert not cal.from_marker(8, 2, RULES).reliable


def test_label_sized_device_never_used():
    c = cal.from_device("sheath-7F", 12, 0.5, RULES, devices=[{"id": "sheath-7F", "name": "7F", "label_fr": 7}])
    assert not c.reliable


def test_measured_device_used():
    devices = [{"id": "d1", "name": "measured sheath", "measured_outer_mm": 2.84, "measured_by": "GR"}]
    c = cal.from_device("d1", 11.8, 0.2, RULES, devices=devices)
    assert c.reliable and abs(c.mm_per_px - 2.84 / 11.8) < 1e-9


# ---- notes -> answer key ----

def test_note_parsing():
    r = parse_note("S1", "Left brachiocephalic AVF. 70% stenosis at the cephalic arch. Juxta-anastomotic segment patent. "
                         "PTA with 8 mm balloon, residual 20%. Good result.")
    assert r["side"] == "LEFT" and r["access_type"] == "AVF" and r["segments_narrowed"] == "6"
    assert r["max_stenosis_percent"] == 70 and r["residual_percent"] == 20 and "angioplasty" in r["procedure"]


def test_unclear_note_is_flagged_not_guessed():
    r = parse_note("S2", "Stenosis treated.")
    assert r["needs_review"] == "yes" and r["side"] == ""


def test_occlusion_and_central_vein():
    r = parse_note("S3", "Right AVG. Subclavian vein occluded.")
    assert r["segments_blocked"] == "7" and r["side"] == "RIGHT" and r["access_type"] == "AVG"


# ---- anonymisation ----

@pytest.fixture(scope="module")
def synthetic_study(tmp_path_factory):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_synthetic_study import make_study

    d = tmp_path_factory.mktemp("study")
    truth = make_study(d, seed=3)
    return d, truth


def test_anonymiser_removes_identifiers(synthetic_study, tmp_path):
    from access.anonymise import anonymise_folder

    src, _ = synthetic_study
    rows = anonymise_folder(src, tmp_path / "anon", b"k" * 32, tmp_path / "map.csv", progress=lambda *_: None)
    assert len(rows) == 3 and all(int(r["text_regions_blacked"]) >= 3 for r in rows)
    for f in (tmp_path / "anon").rglob("*.dcm"):
        raw = f.read_bytes()
        for needle in (b"KUMAR", b"4471823", b"Johns", b"SHARMA", b"ACC99812", b"19610312"):
            assert needle not in raw
    assert (tmp_path / "anon" / "inventory.csv").exists()


def test_anonymiser_blacks_out_burned_in_text(synthetic_study, tmp_path):
    import pydicom

    from access.anonymise import anonymise_folder

    src, _ = synthetic_study
    anonymise_folder(src, tmp_path / "anon", b"k" * 32, progress=lambda *_: None)
    from make_synthetic_study import text_mask

    f = sorted((tmp_path / "anon").rglob("series001*.dcm"))[0]
    px = pydicom.dcmread(str(f)).pixel_array
    text = text_mask() > 0.2
    assert (px[:, text] <= px.min() + 1).all()  # every burned-in text pixel is black on every frame


def test_pseudonyms_are_stable(synthetic_study, tmp_path):
    from access.anonymise import anonymise_folder

    src, _ = synthetic_study
    a = anonymise_folder(src, tmp_path / "a", b"k" * 32, progress=lambda *_: None)
    b = anonymise_folder(src, tmp_path / "b", b"k" * 32, progress=lambda *_: None)
    assert [r["study_pseudonym"] for r in a] == [r["study_pseudonym"] for r in b]


def test_anonymiser_handles_compressed_dicom(synthetic_study, tmp_path):
    import pydicom
    from pydicom.uid import RLELossless

    from access.anonymise import anonymise_file

    src, _ = synthetic_study
    ds = pydicom.dcmread(str(sorted(Path(src).glob("run1*.dcm"))[0]))
    ds.compress(RLELossless)
    comp = tmp_path / "compressed.dcm"
    ds.save_as(str(comp))
    res = anonymise_file(comp, tmp_path / "out.dcm", b"k" * 32)
    out = pydicom.dcmread(str(tmp_path / "out.dcm"))
    assert res["text_regions"] >= 3 and out.PatientID.startswith("DA-") and out.pixel_array.shape[0] == ds.NumberOfFrames
