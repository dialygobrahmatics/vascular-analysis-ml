"""Turn the doctor's procedure notes into the answer-key table (PRD Step 2, D2).

Simple, inspectable text rules (rules/notes_rules.json) instead of a language model, so
the notes never leave the machine and every extracted value can be traced to a word in
the note. Rows the rules cannot settle are flagged `needs_review` rather than guessed;
the doctor checks the table (target: at least 45 of 50 sampled rows correct).

Input: a CSV with columns study_id,note (any extra columns are carried over), or a folder
of .txt files named <study_id>.txt.
    python -m access.notes <notes.csv | folder> <answer_key.csv>
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from . import rules as rule_tables
from .laterality import side_from_text

FIELDS = ["study_id", "side", "access_type", "segments_narrowed", "segments_blocked", "max_stenosis_percent",
          "percents_mentioned", "procedure", "result", "residual_percent", "needs_review", "review_reasons"]


def _has(text: str, words) -> bool:
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text) for w in words)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.;\n]+", text) if s.strip()]


def parse_note(study_id: str, note: str, rules: dict | None = None) -> dict:
    rules = rules or rule_tables.load("notes_rules")
    low = note.lower()
    reasons = []
    side = side_from_text(note)
    if side is None:
        reasons.append("side not stated" if not re.search(r"\b(left|right|lt|rt)\b", low) else "both sides mentioned")
    access = [k for k, words in rules["access_type"].items() if _has(low, words)]
    if len(access) != 1:
        reasons.append("access type unclear" if not access else "both AVF and AVG mentioned")
    narrowed, blocked = set(), set()
    for sent in _sentences(low):
        segs = [k for k, words in rules["segments"].items() if _has(sent, words)]
        if not segs:
            continue
        if _has(sent, rules["occlusion_words"]):
            blocked.update(segs)
        if _has(sent, rules["stenosis_words"]) or re.search(r"\d{1,3}\s*%", sent):
            narrowed.update(segs)
    # "cephalic arch" also contains "arch"; the more specific segment wins over a vaguer one
    percents = [int(p) for p in re.findall(r"(\d{1,3})\s*%", low) if 0 < int(p) <= 100]
    residual = None
    m = re.search(r"residual[^.;\n]*?(\d{1,3})\s*%", low)
    if m:
        residual = int(m.group(1))
    stenosis_percents = [p for p in percents if p != residual]
    procedure = [k for k, words in rules["procedure"].items() if _has(low, words)]
    result = next((k for k, words in rules["result_words"].items() if _has(low, words)), "")
    if (narrowed or blocked) and not stenosis_percents and not blocked:
        reasons.append("narrowing mentioned without a percentage")
    if not (narrowed or blocked) and _has(low, rules["stenosis_words"]):
        reasons.append("narrowing mentioned but segment not recognised")
    return {
        "study_id": study_id, "side": side or "", "access_type": access[0] if len(access) == 1 else "",
        "segments_narrowed": ";".join(sorted(narrowed, key=int)), "segments_blocked": ";".join(sorted(blocked, key=int)),
        "max_stenosis_percent": max(stenosis_percents) if stenosis_percents else "",
        "percents_mentioned": ";".join(map(str, percents)), "procedure": ";".join(procedure), "result": result,
        "residual_percent": "" if residual is None else residual, "needs_review": "yes" if reasons else "no",
        "review_reasons": "; ".join(reasons),
    }


def read_notes(src: str | Path) -> list[tuple[str, str]]:
    src = Path(src)
    if src.is_dir():
        return [(p.stem, p.read_text(encoding="utf-8", errors="replace")) for p in sorted(src.glob("*.txt"))]
    with open(src, newline="", encoding="utf-8-sig") as fh:
        return [(row["study_id"], row["note"]) for row in csv.DictReader(fh)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Turn procedure notes into the answer-key table.")
    ap.add_argument("notes", help="CSV with study_id,note columns, or a folder of <study_id>.txt files")
    ap.add_argument("out", help="answer-key CSV to write")
    a = ap.parse_args(argv)
    rules = rule_tables.load("notes_rules")
    rows = [parse_note(sid, note, rules) for sid, note in read_notes(a.notes)]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    review = sum(r["needs_review"] == "yes" for r in rows)
    print(f"{len(rows)} notes -> {a.out}; {review} flagged for review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
