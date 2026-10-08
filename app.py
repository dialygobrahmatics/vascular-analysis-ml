"""Web front end for the vascular analysis pipeline (research use only).

Run:
    python app.py
Then open http://127.0.0.1:5000 and upload an image or a video.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from run_analysis import EXTS, VESSEL_CHOICES, analyze_file
from vascular.video import DICOM_EXTS, MAX_DURATION_S, MAX_PARTS, MIN_PARTS, VIDEO_EXTS, VideoError, analyze_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# video job status/result files: on disk rather than in memory because gunicorn runs
# several worker processes and the progress poll may land on a different one than the
# upload; kept outside static/ so they are not publicly served
JOB_DIR = BASE_DIR / "jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_MB = 25
MAX_VIDEO_MB = 500
VIDEO_UPLOAD_EXTS = VIDEO_EXTS | DICOM_EXTS
DEFAULT_PARTS = 5
# video analyses running at once per worker process; further uploads wait their turn
MAX_CONCURRENT_VIDEO_JOBS = 2
_video_slots = threading.BoundedSemaphore(MAX_CONCURRENT_VIDEO_JOBS)
# overall progress bar: share of 0-100 each stage occupies
_STAGE_SPAN = {"reading": (0, 30), "selecting": (30, 40), "checking": (40, 45), "analysing": (45, 98)}
_STAGE_TEXT = {
    "reading": "Reading video frames",
    "selecting": "Choosing the best frame in each part",
    "checking": "Checking each part for vessel signal",
    "analysing": "Analysing",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_MB * 1024 * 1024


def _remove_leftovers(max_age_s: float = 3600) -> None:
    """A job deletes its uploaded video when it finishes; if the server restarts mid-job
    the video (up to MAX_VIDEO_MB) would stay forever, so clear old ones at startup,
    along with old job status files."""
    cutoff = time.time() - max_age_s
    stale = [p for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in VIDEO_UPLOAD_EXTS] + list(JOB_DIR.iterdir())
    for p in stale:
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


_remove_leftovers()

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
        max_video_min=f"{MAX_DURATION_S / 60:g}",
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


def _job_path(job_id: str, kind: str = "json") -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        abort(404)
    return JOB_DIR / f"{job_id}.{kind}"


def _write_job(job_id: str, **status) -> None:
    status["updated"] = time.time()
    path = _job_path(job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status), encoding="utf-8")
    # atomic, so a poll never reads a half-written file; on Windows the replace fails
    # while a poll has the file open, so retry briefly
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def _video_page(res: dict) -> str:
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
    return page("video", video_summary=res["summary"], video_notes=res["notes"], polarity=res["polarity"], parts=cards)


def _run_video_job(job_id: str, saved_path: Path, parts: int, spacing_mm: float | None, vessels: str) -> None:
    last = {"t": 0.0, "stage": None}

    def progress(stage: str, done: int, total: int, detail: str = "") -> None:
        now = time.time()
        if stage == last["stage"] and now - last["t"] < 0.5:
            return
        last.update(t=now, stage=stage)
        lo, hi = _STAGE_SPAN[stage]
        frac = min(done / total, 1.0) if total else 0.0
        if stage == "analysing":
            message = f"Analysing {detail}" if detail else "Analysing"
            step = f"{done} of {total} analysed" if total else ""
        elif stage == "reading":
            message = _STAGE_TEXT[stage]
            step = f"{done:,} of ~{total:,} frames" if total else f"{done:,} frames"
        elif stage == "checking":
            message, step = _STAGE_TEXT[stage], f"part {done + 1} of {total}"
        else:
            message, step = _STAGE_TEXT[stage], ""
        try:
            _write_job(job_id, state="running", stage=stage, percent=round(lo + (hi - lo) * frac), message=message, step=step)
        except OSError:
            pass  # a missed progress update must never fail the analysis itself

    _write_job(job_id, state="queued", percent=0, message="Waiting for another analysis to finish", step="")
    with _video_slots:
        _write_job(job_id, state="running", stage="reading", percent=0, message=_STAGE_TEXT["reading"], step="")
        # rendering needs a request context for url_for(); a test context gives the
        # same relative /static/... URLs the real request would
        with app.test_request_context("/"):
            try:
                res = analyze_video(
                    saved_path, UPLOAD_DIR, saved_path.stem, parts=parts, spacing_mm=spacing_mm, vessels=vessels, progress=progress
                )
                html = _video_page(res)
            except VideoError as exc:
                html = page("video", video_error=str(exc), video_error_notes=exc.notes)
            except Exception as exc:
                html = page("video", error=f"Video analysis failed: {exc}")
            finally:
                # only the per-part overlays are needed afterwards; videos are large
                saved_path.unlink(missing_ok=True)
        _job_path(job_id, "html").write_text(html, encoding="utf-8")
        _write_job(job_id, state="done", percent=100, message="Done", step="")


@app.route("/analyze-video", methods=["POST"])
def analyze_video_route():
    """Validates the upload, then starts the analysis in the background and returns a
    job id; the page polls /video-jobs/<id> for progress. Validation errors come back
    as a normal page so the browser can show them straight away."""
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
    job_id = uuid.uuid4().hex
    _write_job(job_id, state="queued", percent=0, message="Starting", step="")
    threading.Thread(target=_run_video_job, args=(job_id, saved_path, parts, spacing_mm, vessels), daemon=True).start()
    return jsonify(job=job_id), 202


@app.route("/video-jobs/<job_id>", methods=["GET"])
def video_job_status(job_id: str):
    path = _job_path(job_id)
    if not path.exists():
        abort(404)
    status = json.loads(path.read_text(encoding="utf-8"))
    status["age"] = round(time.time() - status.pop("updated"), 1)
    return jsonify(status)


@app.route("/video-jobs/<job_id>/result", methods=["GET"])
def video_job_result(job_id: str):
    path = _job_path(job_id, "html")
    if not path.exists():
        abort(404)
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html")


if __name__ == "__main__":
    app.run(debug=True)
