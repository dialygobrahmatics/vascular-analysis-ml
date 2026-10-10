"""Signed report (requirement R11): a PDF for people and a JSON file for systems, built only
from a study the doctor has signed (golden rule 7). The report states findings and
measurements; it never recommends treatment or a balloon size (PRD "must not").
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .checklist import DANGER, SEGMENTS

STATUS_LABEL = {s["id"]: s["label"] for s in SEGMENTS["statuses"]}
ANSWER_LABEL = {a["id"]: a["label"] for a in DANGER["answers"]}
DISCLAIMER = ("Research use only. Measurements are computer-assisted and were checked and signed by the doctor named "
              "below. This report does not recommend treatment.")


def _fmt(v, err=None, unit=""):
    if v is None:
        return "not available"
    s = f"{v:g}{unit}"
    return f"{s} ± {err:g}{unit}" if err not in (None, 0) else s


def build_json(study: dict, runs: list[dict], findings: list[dict], warnings: list[dict], signer: dict) -> dict:
    accepted = [f for f in findings if f["status"] == "accepted" and f["data"].get("ok")]
    return {
        "format": "dialygo-access-report", "version": 1, "study_id": study["pseudonym"] or study["uid"],
        "study_uid": study["uid"], "signed_by": signer["username"], "signed_by_name": signer.get("full_name"),
        "signed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(study["signed_at"])),
        "review_seconds": round(study["signed_at"] - study["review_started_at"], 1) if study.get("review_started_at") else None,
        "laterality": {k: study["laterality"].get(k) for k in ("side", "status", "message", "override")},
        "segment_status": study["segment_status"],
        "danger": study["danger"],
        "warnings": [{"kind": w["kind"], "message": w["message"], "acknowledged_by": w["acknowledged_by"],
                      "response": w["response"]} for w in warnings],
        "findings": [
            {"segment": f["segment"], "run": f["data"].get("run_idx"), "frame": f["data"].get("frame"),
             "status": f["status"], "confidence": f["data"].get("confidence"),
             "confidence_reasons": f["data"].get("confidence_reasons"),
             **{k: f["data"].get(k) for k in ("mld_mm", "mld_mm_err", "rvd_mm", "rvd_mm_err", "ds_percent", "ds_err",
                                             "length_mm", "length_mm_err", "mld_px", "rvd_px", "percentage_available",
                                             "mm_available", "ref_side", "mld_source", "width_method", "notes")},
             "calibration": (f["data"].get("calibration") or {}).get("method")}
            for f in accepted
        ],
        "runs": [{"run": r["idx"], "frames": r["info"]["n_frames"], "dsa": r["info"]["is_dsa"],
                  "best_frame": r["best"]["index"], "calibration": r["calibration"]["method"]} for r in runs],
        "disclaimer": DISCLAIMER,
    }


def _text_page(pdf, lines: list[tuple[str, str]]):
    fig = plt.figure(figsize=(8.27, 11.69))
    y = 0.96
    for style, text in lines:
        size = {"h1": 15, "h2": 11.5, "b": 9.5, "t": 9, "s": 7.5}[style]
        weight = "bold" if style in ("h1", "h2", "b") else "normal"
        for chunk in textwrap.wrap(text, 105 if size < 9 else 95) or [""]:
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                y = 0.96
            fig.text(0.06, y, chunk, fontsize=size, weight=weight, family="DejaVu Sans")
            y -= size / 520 + (0.012 if style in ("h1", "h2") else 0.004)
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf(path: Path, rep: dict, images: list[tuple[str, np.ndarray, dict]]) -> None:
    lines = [("h1", "Dialygo-Access - dialysis access angiogram report"), ("s", DISCLAIMER), ("t", ""),
             ("b", f"Study: {rep['study_id']}"),
             ("b", f"Side: {rep['laterality'].get('side')} ({rep['laterality'].get('message')})"),
             ("t", f"Signed by {rep['signed_by_name'] or rep['signed_by']} on {rep['signed_at']}"), ("t", ""),
             ("h2", "Segments")]
    for s in SEGMENTS["segments"]:
        lines.append(("t", f"[{s['id']}] {s['name']}: {STATUS_LABEL.get(rep['segment_status'].get(str(s['id'])), 'MISSING')}"))
    lines += [("t", ""), ("h2", "Measured narrowings")]
    if not rep["findings"]:
        lines.append(("t", "None measured."))
    for f in rep["findings"]:
        seg = next((s["name"] for s in SEGMENTS["segments"] if str(s["id"]) == str(f["segment"])), "segment not set")
        lines.append(("b", f"[{f['segment']}] {seg} - run {f['run']}, frame {f['frame']} (confidence {f['confidence']})"))
        lines.append(("t", f"Narrowest width (MLD): {_fmt(f['mld_mm'], f['mld_mm_err'], ' mm')}; "
                           f"normal width (RVD): {_fmt(f['rvd_mm'], f['rvd_mm_err'], ' mm')}; "
                           f"narrowing: {_fmt(f['ds_percent'], f['ds_err'], '%')}; length: {_fmt(f['length_mm'], f['length_mm_err'], ' mm')}"))
        lines.append(("s", f"Calibration: {f['calibration']}. Reference: {f['ref_side']}. Width method: {f['width_method']}."
                           + (" Notes: " + " ".join(f["notes"]) if f.get("notes") else "")))
    lines += [("t", ""), ("h2", "Danger signs (each answered by the doctor)")]
    for d in DANGER["items"]:
        lines.append(("t", f"{d['name']}: {ANSWER_LABEL.get(rep['danger'].get(d['id']), 'MISSING')}"))
    if rep["warnings"]:
        lines += [("t", ""), ("h2", "Automatic warnings and the doctor's response")]
        for w in rep["warnings"]:
            lines.append(("t", f"{w['message']} -> {w['response'] or 'acknowledged'} ({w['acknowledged_by']})"))
    lines += [("t", ""), ("s", "± ranges are approximate 95% ranges from edge-fit and calibration uncertainty. Millimetres are shown only "
                               "when the pixel-to-mm conversion is reliable; otherwise percentages only.")]
    with PdfPages(path) as pdf:
        _text_page(pdf, lines)
        for title, img, f in images:
            fig, ax = plt.subplots(figsize=(8.27, 8.27))
            ax.imshow(img, cmap="gray")
            if f.get("profile"):
                c = np.array(f["profile"]["centres"])
                ax.plot(c[:, 1], c[:, 0], color="#ff9f1c", lw=0.8)
                for pts, col in ((f.get("mld_points"), "#00e5ff"), (f.get("ref_points"), "#7CFC00")):
                    if pts:
                        p = np.array(pts)
                        ax.plot(p[:, 1], p[:, 0], color=col, lw=1.6, marker="o", ms=3)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def write_report(out_dir: Path, study: dict, runs: list[dict], findings: list[dict], warnings: list[dict], signer: dict,
                 frame_loader) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = build_json(study, runs, findings, warnings, signer)
    jpath = out_dir / f"report_{rep['study_id']}.json"
    jpath.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    images = []
    for f in findings:
        if f["status"] == "accepted" and f["data"].get("ok"):
            run = next(r for r in runs if r["id"] == f["run_id"])
            images.append((f"Segment [{f['segment']}], run {run['idx']}, frame {f['data']['frame']}: cyan = narrowest point, "
                           f"green = reference", frame_loader(run, f["data"]["frame"]), f["data"]))
    ppath = out_dir / f"report_{rep['study_id']}.pdf"
    build_pdf(ppath, rep, images)
    return jpath, ppath
