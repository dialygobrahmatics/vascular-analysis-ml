"""Web front end for the vascular analysis pipeline (research use only).

Run:
    python app.py
Then open http://127.0.0.1:5000 and upload an image or a video.
"""

from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path

from flask import Flask, Response, render_template, request, url_for
from werkzeug.utils import secure_filename

from run_analysis import EXTS, VESSEL_CHOICES, analyze_file
from vascular.video import DICOM_EXTS, MAX_PARTS, MIN_PARTS, VIDEO_EXTS, VideoError, analyze_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_MB = 25
MAX_VIDEO_MB = 200
VIDEO_UPLOAD_EXTS = VIDEO_EXTS | DICOM_EXTS
DEFAULT_PARTS = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_MB * 1024 * 1024

# Optional HTTP basic auth: enabled only when both env vars are set.
AUTH_USER = os.environ.get("BASIC_AUTH_USER")
AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD")


@app.before_request
def require_basic_auth():
    if not (AUTH_USER and AUTH_PASSWORD):
        return None
    auth = request.authorization
    if (
        auth
        and secrets.compare_digest(auth.username or "", AUTH_USER)
        and secrets.compare_digest(auth.password or "", AUTH_PASSWORD)
    ):
        return None
    return Response("Authentication required.", 401, {"WWW-Authenticate": 'Basic realm="Vascular Analysis"'})


def page(mode: str = "image", **ctx):
    return render_template(
        "index.html",
        mode=mode,
        image_exts=sorted(EXTS),
        video_exts=sorted(VIDEO_UPLOAD_EXTS),
        min_parts=MIN_PARTS,
        max_parts=MAX_PARTS,
        default_parts=DEFAULT_PARTS,
        max_image_mb=MAX_IMAGE_MB,
        max_video_mb=MAX_VIDEO_MB,
        **ctx,
    )


@app.errorhandler(413)
def too_large(_exc):
    return page(request.form.get("mode", "image"), error=f"File too large. Limits: images {MAX_IMAGE_MB} MB, videos {MAX_VIDEO_MB} MB."), 413


def _common_form(mode: str):
    """Shared form fields; returns (spacing_mm, vessels) or an error page."""
    spacing_raw = (request.form.get("spacing_mm") or "").strip()
    spacing_mm = None
    if spacing_raw:
        try:
            spacing_mm = float(spacing_raw)
        except ValueError:
            return page(mode, error="Pixel spacing must be a number (mm/px).")
    vessels = request.form.get("vessels", "auto")
    if vessels not in VESSEL_CHOICES:
        vessels = "auto"
    return spacing_mm, vessels


def _save_upload(file, keep_name: bool = True) -> Path:
    token = uuid.uuid4().hex[:12]
    if keep_name:
        name = f"{token}_{secure_filename(file.filename) or 'upload'}"
    else:
        # result files are named after the upload, so dropping the original name keeps
        # it out of the overlay URLs too
        name = f"{token}{Path(file.filename).suffix.lower()}"
    saved_path = UPLOAD_DIR / name
    file.save(saved_path)
    return saved_path


def _rec_body(recommendation: str) -> str:
    return recommendation.removeprefix("Recommendations: ") if recommendation else ""


def _overlay_url(path: Path) -> str:
    return url_for("static", filename=f"uploads/{path.name}")


@app.route("/", methods=["GET"])
def index():
    return page()


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return page(error="Please choose an image file to upload.")
    if (request.content_length or 0) > MAX_IMAGE_MB * 1024 * 1024:
        return page(error=f"Image too large (limit {MAX_IMAGE_MB} MB).")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in EXTS:
        hint = " For videos, use the Video tab." if suffix in VIDEO_EXTS else ""
        return page(error=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(EXTS))}.{hint}")

    form = _common_form("image")
    if not isinstance(form, tuple):
        return form
    spacing_mm, vessels = form

    saved_path = _save_upload(file)
    try:
        res = analyze_file(saved_path, UPLOAD_DIR, spacing_mm=spacing_mm, vessels=vessels)
    except Exception as exc:
        return page(error=f"Analysis failed: {exc}")

    metrics = res["metrics"]
    return page(
        "image",
        observation=res["observation"],
        recommendation=_rec_body(res["recommendation"]),
        quality=res["quality"].get("label", "unknown"),
        polarity=res["polarity"],
        filename=file.filename,
        overlay_url=_overlay_url(res["overlay"]),
        n_branches=metrics.get("n_branches"),
        total_length=metrics.get("total_length"),
        n_stenoses=len(metrics.get("candidate_stenoses", [])),
    )


@app.route("/analyze-video", methods=["POST"])
def analyze_video_route():
    file = request.files.get("video")
    if file is None or file.filename == "":
        return page("video", error="Please choose a video file to upload.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_UPLOAD_EXTS:
        hint = " For single images, use the Image tab." if suffix in EXTS else ""
        return page("video", error=f"Unsupported video type '{suffix or 'unknown'}'. Supported: {', '.join(sorted(VIDEO_UPLOAD_EXTS))}.{hint}")

    form = _common_form("video")
    if not isinstance(form, tuple):
        return form
    spacing_mm, vessels = form
    try:
        parts = int(request.form.get("parts") or DEFAULT_PARTS)
    except ValueError:
        return page("video", error="Number of parts must be a whole number.")
    if not MIN_PARTS <= parts <= MAX_PARTS:
        return page("video", error=f"Number of parts must be between {MIN_PARTS} and {MAX_PARTS}.")

    saved_path = _save_upload(file, keep_name=False)
    try:
        res = analyze_video(
            saved_path, UPLOAD_DIR, saved_path.stem, parts=parts, spacing_mm=spacing_mm, vessels=vessels
        )
    except VideoError as exc:
        return page("video", video_error=str(exc), video_error_notes=exc.notes)
    except Exception as exc:
        return page("video", error=f"Video analysis failed: {exc}")
    finally:
        # only the per-part overlays are needed afterwards; videos are large
        saved_path.unlink(missing_ok=True)

    cards = []
    for p in res["parts"]:
        card = {
            "index": p["index"],
            "range": f"{p['start_label']} - {p['end_label']}",
            "time_label": p.get("time_label"),
            "skipped": p.get("skipped"),
        }
        r = p.get("result")
        if r:
            m = r["metrics"]
            card.update(
                observation=r["observation"],
                recommendation=_rec_body(r["recommendation"]),
                quality=r["quality"].get("label", "unknown"),
                overlay_url=_overlay_url(r["overlay"]),
                n_branches=m.get("n_branches"),
                total_length=m.get("total_length"),
                n_stenoses=len(m.get("candidate_stenoses", [])),
            )
        cards.append(card)

    return page(
        "video",
        video_summary=res["summary"],
        video_notes=res["notes"],
        polarity=res["polarity"],
        parts=cards,
    )


if __name__ == "__main__":
    app.run(debug=True)
