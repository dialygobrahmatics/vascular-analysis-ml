"""Editable rule tables (JSON) owned by the doctor."""

from __future__ import annotations

import copy
import json
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((RULES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def reference_rule(rules: dict, segment: str | int | None) -> dict:
    """Default reference rule merged with the segment-specific overrides, if any."""
    ref = copy.deepcopy(rules["reference"]["default"])
    if segment is not None:
        ref.update({k: v for k, v in rules["reference"]["segments"].get(str(segment), {}).items() if not k.startswith("_")})
    return ref
