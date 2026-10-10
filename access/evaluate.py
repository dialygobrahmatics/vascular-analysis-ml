"""Score the tool against the answer key (PRD Step 9 and section 9 targets), reporting
honest numbers including the failures.

Inputs:
- the signed (or draft) study JSON files the review app exports (one per study)
- an answer-key CSV: one row per study, with at least study_id and side; optionally
  per-segment expected status and expert MLD / %DS, in columns
  seg<k>_status (open|narrow|blocked|not_visible), seg<k>_mld_mm, seg<k>_ds_percent,
  and danger_<id> (present|absent) for danger signs.

    python -m access.evaluate <reports_dir> <answer_key.csv> [--out summary.json]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .checklist import DANGER_IDS, SEGMENT_IDS

TARGETS = {
    "laterality_accuracy": ("Left or right correct", 1.0, ">="),
    "segments_answered": ("Every segment answered", 1.0, ">="),
    "danger_missed": ("Danger signs missed", 0, "<="),
    "sensitivity_50": ("Narrowings over 50% caught", 0.9, ">="),
    "false_alarm_rate": ("Healthy segments called narrow", 0.2, "<="),
    "mld_within_0_5mm": ("MLD within 0.5 mm of the doctor", 0.8, ">="),
    "ds_within_10": ("%DS within 10 points of the doctor", 0.8, ">="),
    "median_review_minutes": ("Review and sign time (median, minutes)", 5.0, "<="),
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score(reports: dict[str, dict], key_rows: list[dict]) -> dict:
    c = {"n": 0, "lat_ok": 0, "lat_n": 0, "answered": 0, "danger_missed": 0, "danger_n": 0,
         "sig_n": 0, "sig_hit": 0, "healthy_n": 0, "false_alarm": 0, "mld": [], "ds": []}
    failures, review_minutes = [], []
    for row in key_rows:
        sid = row["study_id"]
        rep = reports.get(sid)
        if rep is None:
            failures.append({"study_id": sid, "problem": "no report for this study"})
            continue
        c["n"] += 1
        if row.get("side"):
            c["lat_n"] += 1
            if (rep.get("laterality", {}).get("side") or "").upper() == row["side"].upper():
                c["lat_ok"] += 1
            else:
                failures.append({"study_id": sid, "problem": f"side {rep.get('laterality', {}).get('side')} vs expected {row['side']}"})
        status = rep.get("segment_status", {})
        if all(status.get(str(s)) for s in SEGMENT_IDS):
            c["answered"] += 1
        else:
            failures.append({"study_id": sid, "problem": "not every segment answered"})
        for d in DANGER_IDS:
            exp = row.get(f"danger_{d}")
            if exp == "present":
                c["danger_n"] += 1
                if rep.get("danger", {}).get(d) != "present":
                    c["danger_missed"] += 1
                    failures.append({"study_id": sid, "problem": f"danger sign missed: {d}"})
        by_seg = {}
        for f in rep.get("findings", []):
            by_seg.setdefault(str(f.get("segment")), []).append(f)
        for s in SEGMENT_IDS:
            exp = row.get(f"seg{s}_status")
            if not exp:
                continue
            got = status.get(str(s))
            exp_ds = _f(row.get(f"seg{s}_ds_percent"))
            if exp == "narrow" and (exp_ds is None or exp_ds > 50):
                c["sig_n"] += 1
                if got in ("narrow", "blocked"):
                    c["sig_hit"] += 1
                else:
                    failures.append({"study_id": sid, "problem": f"segment {s}: significant narrowing missed (called {got})"})
            if exp == "open":
                c["healthy_n"] += 1
                if got == "narrow":
                    c["false_alarm"] += 1
                    failures.append({"study_id": sid, "problem": f"segment {s}: healthy segment called narrow"})
            fs = by_seg.get(str(s), [])
            if fs and _f(row.get(f"seg{s}_mld_mm")) is not None and fs[0].get("mld_mm") is not None:
                c["mld"].append(abs(fs[0]["mld_mm"] - _f(row[f"seg{s}_mld_mm"])))
            if fs and exp_ds is not None and fs[0].get("ds_percent") is not None:
                c["ds"].append(abs(fs[0]["ds_percent"] - exp_ds))
        if rep.get("review_seconds"):
            review_minutes.append(rep["review_seconds"] / 60)

    def ratio(a, b):
        return None if b == 0 else round(a / b, 3)

    import statistics

    metrics = {
        "laterality_accuracy": ratio(c["lat_ok"], c["lat_n"]),
        "segments_answered": ratio(c["answered"], c["n"]),
        "danger_missed": c["danger_missed"] if c["danger_n"] else None,
        "sensitivity_50": ratio(c["sig_hit"], c["sig_n"]),
        "false_alarm_rate": ratio(c["false_alarm"], c["healthy_n"]),
        "mld_within_0_5mm": ratio(sum(e <= 0.5 for e in c["mld"]), len(c["mld"])),
        "ds_within_10": ratio(sum(e <= 10 for e in c["ds"]), len(c["ds"])),
        "median_review_minutes": round(statistics.median(review_minutes), 1) if review_minutes else None,
    }
    table = []
    for k, (label, target, op) in TARGETS.items():
        v = metrics[k]
        met = None if v is None else (v >= target if op == ">=" else v <= target)
        table.append({"metric": label, "value": v, "target": f"{op} {target}", "met": met})
    return {"studies_scored": c["n"], "metrics": metrics, "targets": table,
            "counts": {k: v for k, v in c.items() if not isinstance(v, list)},
            "mld_errors_mm": [round(e, 2) for e in c["mld"]], "ds_errors": [round(e, 1) for e in c["ds"]],
            "failures": failures}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score study reports against the answer key.")
    ap.add_argument("reports_dir")
    ap.add_argument("answer_key")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reports = {}
    for p in Path(a.reports_dir).glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        reports[str(d.get("study_id", p.stem))] = d
    with open(a.answer_key, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    res = score(reports, rows)
    for t in res["targets"]:
        mark = "n/a" if t["met"] is None else ("MET" if t["met"] else "NOT MET")
        print(f"{t['metric']:<42} {str(t['value']):>8}  target {t['target']:<8} {mark}")
    print(f"{len(res['failures'])} failure(s) listed in the output.")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
