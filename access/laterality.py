"""Left or right arm (requirement R2, golden rule 1: "left or right comes first").

Side is collected from independent sources -- the DICOM header (laterality tags and
description text together, as they are typed at the same console), the doctor's
procedure note, and the doctor's own entry --
and settled only when at least two agree and none disagree. Any disagreement stops the
pipeline: nothing is measured until the doctor resolves it, and the resolution is logged.
The side is never guessed from the image.
"""

from __future__ import annotations

import re

LEFT, RIGHT = "LEFT", "RIGHT"
_TAG_VALUES = {"L": LEFT, "R": RIGHT, "LEFT": LEFT, "RIGHT": RIGHT}
_WORDS = [(re.compile(r"\b(LEFT|LT|LHS)\b"), LEFT), (re.compile(r"\b(RIGHT|RT|RHS)\b"), RIGHT)]


def side_from_text(text: str) -> str | None:
    text = (text or "").upper()
    found = {side for rx, side in _WORDS if rx.search(text)}
    return found.pop() if len(found) == 1 else None


def header_sources(runs) -> list[dict]:
    """One vote from the DICOM headers of all runs. Laterality tags and description text
    are both typed at the scanner console, so together they count as a single source; if
    they disagree with each other, the vote is a conflict."""
    found = []
    for r in runs:
        for v in r.laterality_tags.values():
            if _TAG_VALUES.get(v):
                found.append((_TAG_VALUES[v], f"series {r.series_number} tag"))
        t = side_from_text(r.description_text)
        if t:
            found.append((t, f"series {r.series_number} description"))
    if not found:
        return []
    sides = {s for s, _ in found}
    detail = ", ".join(sorted(f"{w}: {s}" for s, w in found))
    return [{"source": "DICOM header", "side": sides.pop() if len(sides) == 1 else "CONFLICT", "detail": detail}]


def decide(votes: list[dict]) -> dict:
    """status: confirmed | conflict | needs_confirmation | unknown.
    Measurement is allowed only when status == confirmed."""
    sides = [v["side"] for v in votes if v.get("side")]
    if "CONFLICT" in sides or len(set(sides)) > 1:
        return {"status": "conflict", "side": None, "votes": votes,
                "message": "Sources disagree about the side. Stopped: the doctor must resolve this before anything is measured."}
    if not sides:
        return {"status": "unknown", "side": None, "votes": votes,
                "message": "No source states the side. The doctor must enter it."}
    if len(votes) < 2:
        return {"status": "needs_confirmation", "side": sides[0], "votes": votes,
                "message": f"Only one source ({votes[0]['source']}) says {sides[0]}. A second source or the doctor must confirm."}
    return {"status": "confirmed", "side": sides[0], "votes": votes,
            "message": f"{sides[0]} arm ({' + '.join(v['source'] for v in votes)} agree)."}


def resolve(header_votes: list[dict], note_side: str | None = None, doctor_side: str | None = None,
            doctor_override_reason: str | None = None) -> dict:
    votes = list(header_votes)
    if note_side:
        votes.append({"source": "doctor's procedure note", "side": note_side.upper(), "detail": ""})
    if doctor_side:
        votes.append({"source": "doctor's entry", "side": doctor_side.upper(), "detail": ""})
    result = decide(votes)
    if result["status"] == "conflict" and doctor_side and doctor_override_reason:
        # the doctor has looked at the disagreement and decided; recorded, never silent
        result = {"status": "confirmed", "side": doctor_side.upper(), "votes": votes, "override": doctor_override_reason,
                  "message": f"{doctor_side.upper()} arm, set by the doctor despite disagreeing sources: {doctor_override_reason}"}
    return result
