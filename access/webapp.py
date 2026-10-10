"""The doctor's review screen (PRD Step 8) as a local web app.

Local only: the app refuses to listen on anything but this computer unless explicitly
told to serve a hospital LAN, and loads nothing from the internet (no CDNs). There are no
accounts (the PRD does not ask for them): the reviewer enters their name once, which is
what the change log records as "by whom" (R10), and signs by typing it again (R11).
Every change goes through DB.update_field / DB.audit.

Golden rules enforced here, server side (the browser cannot skip them):
 1. no measurement until the side is confirmed           -> /measure returns 409
 2./3. every segment needs an explicit status            -> checklist.signing_blockers
 5. every finding carries its run, frame and confidence  -> pipeline.measure
 6. warnings and danger signs answered one at a time     -> one endpoint per item, no bulk
 7. only a signed study produces a report               -> /report requires state 'signed'
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import zipfile
import time
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from flask import (Flask, Response, abort, g, jsonify, redirect, render_template, request, send_file, session,
                   url_for)
from werkzeug.utils import secure_filename

from . import calibration as cal
from . import pipeline
from . import rules as rule_tables
from .checklist import DANGER, DANGER_ANSWER_IDS, SEGMENT_STATUS_IDS, SEGMENTS, signing_blockers
from .db import DB
from .dicom_io import is_dicom
from .measure import profile_from_json
from .report import write_report


def _extract_zip(stream, dest: Path) -> int:
    """Extract only DICOM members, never outside `dest` (no path traversal); returns count."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(stream) as z:
        for k, info in enumerate(z.infolist()):
            if info.is_dir():
                continue
            target = dest / f"{k:05d}_{secure_filename(Path(info.filename).name) or 'file'}"
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if is_dicom(target):
                n += 1
            else:
                target.unlink()
    return n

PKG = Path(__file__).resolve().parent
MAX_UPLOAD_GB = 4  # one study, uploaded from this computer to itself


def create_app(data_dir: str | Path) -> Flask:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, template_folder=str(PKG / "templates"), static_folder=str(PKG / "static"))
    key_file = data_dir / "session.key"
    if not key_file.exists():
        key_file.write_bytes(secrets.token_bytes(32))
    app.config.update(SECRET_KEY=key_file.read_bytes(), SESSION_COOKIE_SAMESITE="Strict", SESSION_COOKIE_HTTPONLY=True,
                      MAX_CONTENT_LENGTH=MAX_UPLOAD_GB * 1024 ** 3)
    db = DB(data_dir / "access.db")
    jobs: dict[str, dict] = {}

    # -------------------------------------------------------------- reviewer --
    def reviewer_required(fn):
        """The reviewer's name (asked once, kept in the session) is the "by whom" of the log."""
        @wraps(fn)
        def wrapper(*a, **kw):
            name = session.get("reviewer")
            if not name:
                if request.path.startswith("/api/"):
                    return jsonify(error="Enter your name first (reload the page)."), 401
                return redirect(url_for("start", next=request.path))
            g.user = {"username": name, "full_name": name}
            return fn(*a, **kw)
        return wrapper

    @app.before_request
    def same_origin_posts():
        # SameSite=Strict cookies already stop cross-site posts; this also refuses posts
        # whose Origin header names another site
        if request.method in ("POST", "PUT", "DELETE"):
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                abort(403)

    @app.after_request
    def no_external(resp):
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                                                   "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/start", methods=["GET", "POST"])
    def start():
        error = None
        if request.method == "POST":
            name = " ".join(request.form.get("name", "").split())
            if len(name) >= 2:
                session.clear()
                session["reviewer"] = name
                db.audit(name, None, "session", None, "reviewer", None, name)
                nxt = request.args.get("next") or "/"
                return redirect(nxt if nxt.startswith("/") else "/")
            error = "Please enter your full name."
        return render_template("start.html", error=error)

    @app.route("/favicon.ico")
    def favicon():
        return Response(status=204)

    @app.route("/login", methods=["GET", "POST"])
    def old_login():
        # pages left open from the earlier version (with accounts) still point here
        return redirect(url_for("start"))

    @app.errorhandler(404)
    def not_found(_exc):
        if request.path.startswith("/api/"):
            return jsonify(error="Not found."), 404
        return render_template("not_found.html", user=g.get("user")), 404

    @app.route("/change-reviewer")
    def change_reviewer():
        session.clear()
        return redirect(url_for("start"))

    # --------------------------------------------------------------- studies --
    @app.route("/")
    @reviewer_required
    def studies():
        rows = db.query("SELECT * FROM studies ORDER BY imported_at DESC")
        for r in rows:
            r["n_runs"] = db.one("SELECT COUNT(*) AS n FROM runs WHERE study_id = ?", (r["id"],))["n"]
        return render_template("studies.html", studies=rows, user=g.user)

    def start_import(folder: Path, source: str) -> str:
        job = secrets.token_hex(8)
        jobs[job] = {"state": "running", "message": "Starting", "percent": 0}
        user = g.user["username"]

        def work():
            try:
                ids = pipeline.import_study(db, folder, data_dir, lambda m, f: jobs[job].update(message=m, percent=round(f * 100)))
                db.audit(user, ids[0] if ids else None, "studies", None, "import", None, source)
                jobs[job].update(state="done", study_ids=ids, percent=100, message="Done")
            except Exception as exc:  # shown to the user, who can fix the input and retry
                jobs[job].update(state="error", message=str(exc))

        threading.Thread(target=work, daemon=True).start()
        return job

    @app.route("/api/import", methods=["POST"])
    @reviewer_required
    def api_import():
        """Import from a folder path on this computer (handy for large batches)."""
        folder = (request.json or {}).get("folder", "").strip().strip('"')
        if not folder or not Path(folder).is_dir():
            return jsonify(error="That folder does not exist on this computer."), 400
        return jsonify(job=start_import(Path(folder), folder))

    @app.route("/api/import-upload", methods=["POST"])
    @reviewer_required
    def api_import_upload():
        """Import files chosen in the browser (a folder, several .dcm files, or a .zip).
        The browser sends them to this same computer; they are kept under the data folder."""
        files = request.files.getlist("files")
        if not files:
            return jsonify(error="No files were chosen."), 400
        dest = data_dir / "uploads" / (time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3))
        dest.mkdir(parents=True, exist_ok=True)
        kept = 0
        for i, f in enumerate(files):
            name = Path(f.filename or f"file{i}").name
            if name.lower().endswith(".zip"):
                kept += _extract_zip(f.stream, dest / f"zip{i}")
                continue
            target = dest / f"{i:05d}_{secure_filename(name) or 'file'}"
            f.save(target)
            if is_dicom(target):
                kept += 1
            else:
                target.unlink()  # not DICOM (e.g. DICOMDIR index, thumbnails, notes)
        if kept == 0:
            shutil.rmtree(dest, ignore_errors=True)
            return jsonify(error="None of the chosen files are DICOM files."), 400
        return jsonify(job=start_import(dest, f"upload of {kept} DICOM file(s)"), files=kept)

    @app.route("/api/jobs/<job>")
    @reviewer_required
    def api_job(job):
        return jsonify(jobs.get(job) or {"state": "error", "message": "Unknown job"})

    # ---------------------------------------------------------------- review --
    def get_study(sid: int) -> dict:
        st = db.one("SELECT * FROM studies WHERE id = ?", (sid,))
        if st is None:
            abort(404)
        return st

    def get_run(rid: int) -> dict:
        r = db.one("SELECT * FROM runs WHERE id = ?", (rid,))
        if r is None:
            abort(404)
        return r

    def locked(st):
        return st["state"] == "signed"

    def editable(sid):
        st = get_study(sid)
        if locked(st):
            abort(Response(json.dumps({"error": "This study is signed and can no longer be changed."}), 409, mimetype="application/json"))
        return st

    @app.route("/study/<int:sid>")
    @reviewer_required
    def review(sid):
        st = get_study(sid)
        if not st["review_started_at"] and not locked(st):
            db.update_field("studies", sid, "review_started_at", time.time(), g.user["username"], sid)
        return render_template("review.html", study=st, user=g.user)

    def state_json(sid: int) -> dict:
        st = get_study(sid)
        lat = pipeline.current_laterality(db, st)
        if lat != st["laterality"] and not locked(st):
            db.execute("UPDATE studies SET laterality = ? WHERE id = ?", (lat, sid))
        runs = db.query("SELECT * FROM runs WHERE study_id = ? ORDER BY idx", (sid,))
        findings = db.query("SELECT * FROM findings WHERE study_id = ? ORDER BY run_id, id", (sid,))
        warnings = db.query("SELECT * FROM warnings WHERE study_id = ? ORDER BY id", (sid,))
        blockers = signing_blockers(lat["status"], st["segment_status"], st["danger"],
                                    [{"acknowledged": bool(w["acknowledged_by"])} for w in warnings],
                                    [{"segment": f["segment"], "status": f["status"]} for f in findings if f["data"].get("ok")])
        return {
            "study": {k: st[k] for k in ("id", "uid", "pseudonym", "state", "segment_status", "danger", "signed_by", "signed_at",
                                         "measured", "doctor_side", "override_reason", "note_side")},
            "laterality": lat, "runs": [{k: r[k] for k in ("id", "idx", "best", "calibration", "draft_notes")}
                                        | {"info": {k: r["info"].get(k) for k in ("n_frames", "fps", "rows", "cols", "is_dsa",
                                                                                     "description_text", "series_number", "display_scale")}}
                                        for r in runs],
            "findings": findings, "warnings": warnings, "blockers": blockers,
            "segments": SEGMENTS, "danger_items": DANGER, "devices": cal.load_devices(),
            "user": {"username": g.user["username"]},
        }

    @app.route("/api/study/<int:sid>")
    @reviewer_required
    def api_state(sid):
        return jsonify(state_json(sid))

    @app.route("/api/study/<int:sid>/laterality", methods=["POST"])
    @reviewer_required
    def api_laterality(sid):
        editable(sid)
        body = request.json or {}
        side = (body.get("side") or "").upper() or None
        if side not in (None, "LEFT", "RIGHT"):
            return jsonify(error="Side must be LEFT or RIGHT."), 400
        if "note_side" in body:
            ns = (body.get("note_side") or "").upper() or None
            db.update_field("studies", sid, "note_side", ns, g.user["username"], sid)
        db.update_field("studies", sid, "doctor_side", side, g.user["username"], sid)
        db.update_field("studies", sid, "override_reason", (body.get("override_reason") or "").strip() or None, g.user["username"], sid)
        st = get_study(sid)
        lat = pipeline.current_laterality(db, st)
        db.update_field("studies", sid, "laterality", lat, g.user["username"], sid)
        if lat["status"] == "confirmed" and not st["measured"]:
            pipeline.draft_measurements(db, sid)  # golden rule 1: only now are drafts measured
        return jsonify(state_json(sid))

    def require_side(sid):
        lat = pipeline.current_laterality(db, get_study(sid))
        if lat["status"] != "confirmed":
            abort(Response(json.dumps({"error": "Confirm the side (left / right) first. " + lat["message"]}), 409,
                           mimetype="application/json"))

    # frames / masks
    @app.route("/run/<int:rid>/frame/<int:n>.jpg")
    @reviewer_required
    def frame_jpg(rid, n):
        arrays = pipeline.run_cache(Path(get_run(rid)["cache_dir"]))
        fr = arrays["frames"]
        ok, buf = cv2.imencode(".jpg", fr[int(np.clip(n, 0, len(fr) - 1))], [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/run/<int:rid>/mask/<kind>.png")
    @reviewer_required
    def mask_png(rid, kind):
        if kind not in ("vessel", "device"):
            abort(404)
        p = Path(get_run(rid)["cache_dir"]) / f"{kind}.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rgba = np.zeros((*m.shape, 4), np.uint8)
        color = (255, 159, 28) if kind == "vessel" else (230, 60, 230)
        rgba[m > 0] = (*color, 95)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        return Response(buf.tobytes(), mimetype="image/png")

    @app.route("/api/run/<int:rid>/mask/<kind>", methods=["POST"])
    @reviewer_required
    def api_mask(rid, kind):
        """Doctor's corrected outline (R9): a PNG the size of the frame, white = class."""
        run = get_run(rid)
        editable(run["study_id"])
        if kind not in ("vessel", "device"):
            abort(404)
        data = request.get_data()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify(error="Could not read the outline image."), 400
        alpha = img[..., 3] if img.ndim == 3 and img.shape[2] == 4 else (img if img.ndim == 2 else img.max(axis=2))
        cache = Path(run["cache_dir"])
        arrays = pipeline.run_cache(cache)
        if alpha.shape != arrays[kind].shape:
            return jsonify(error="Outline size does not match the frame."), 400
        new = alpha > 0
        old_px = int(arrays[kind].sum())
        arrays[kind] = new
        np.savez_compressed(cache / "analysis.npz", **arrays)
        cv2.imwrite(str(cache / f"{kind}.png"), new.astype(np.uint8) * 255)
        db.audit(g.user["username"], run["study_id"], "runs", rid, f"{kind}_outline", {"pixels": old_px}, {"pixels": int(new.sum())})
        # findings on this run are re-measured on the corrected outline, keeping the doctor's points
        out = []
        for f in db.query("SELECT * FROM findings WHERE run_id = ?", (rid,)):
            i = f["inputs"]
            out.append(pipeline.measure(db, run, g.user["username"], i.get("start"), i.get("end"), f["segment"],
                                        i.get("mld_index"), i.get("ref_indices"), finding_id=f["id"]))
        return jsonify(ok=True, remeasured=len(out))

    @app.route("/api/run/<int:rid>/bestframe", methods=["POST"])
    @reviewer_required
    def api_bestframe(rid):
        run = get_run(rid)
        editable(run["study_id"])
        n = int((request.json or {}).get("index", -1))
        arrays = pipeline.run_cache(Path(run["cache_dir"]))
        frames = arrays["frames"].astype(np.float32) / 255.0
        if not 0 <= n < len(frames):
            return jsonify(error="No such frame."), 400
        from .draft_seg import draft_masks

        seg = draft_masks(frames, n)
        arrays.update(vessel=seg["vessel"], device=seg["device"], dye=seg["dye_raw"].astype(np.float32))
        cache = Path(run["cache_dir"])
        np.savez_compressed(cache / "analysis.npz", **arrays)
        cv2.imwrite(str(cache / "vessel.png"), seg["vessel"].astype(np.uint8) * 255)
        cv2.imwrite(str(cache / "device.png"), seg["device"].astype(np.uint8) * 255)
        best = run["best"] | {"index": n, "chosen_by": g.user["username"]}
        db.update_field("runs", rid, "best", best, g.user["username"], run["study_id"])
        for f in db.query("SELECT * FROM findings WHERE run_id = ?", (rid,)):
            i = f["inputs"]
            pipeline.measure(db, get_run(rid), g.user["username"], i.get("start"), i.get("end"), f["segment"],
                             None, None, finding_id=f["id"])
        return jsonify(ok=True)

    @app.route("/api/run/<int:rid>/measure", methods=["POST"])
    @reviewer_required
    def api_measure(rid):
        run = get_run(rid)
        editable(run["study_id"])
        require_side(run["study_id"])
        b = request.json or {}
        res = pipeline.measure(db, run, g.user["username"], b.get("start"), b.get("end"), b.get("segment"),
                               finding_id=b.get("finding_id"), status="accepted" if b.get("accept") else "draft")
        return jsonify(res), (200 if res.get("ok") else 422)

    @app.route("/api/finding/<int:fid>/points", methods=["POST"])
    @reviewer_required
    def api_points(fid):
        f = db.one("SELECT * FROM findings WHERE id = ?", (fid,))
        if f is None:
            abort(404)
        editable(f["study_id"])
        require_side(f["study_id"])
        b = request.json or {}
        t0 = time.time()
        res = pipeline.remeasure_points(db, f, get_run(f["run_id"]), g.user["username"], b.get("mld_index"), b.get("ref_indices"))
        res["seconds"] = round(time.time() - t0, 3)
        return jsonify(res)

    @app.route("/api/finding/<int:fid>", methods=["POST"])
    @reviewer_required
    def api_finding(fid):
        f = db.one("SELECT * FROM findings WHERE id = ?", (fid,))
        if f is None:
            abort(404)
        editable(f["study_id"])
        b = request.json or {}
        if "segment" in b:
            seg = None if b["segment"] in (None, "") else str(b["segment"])
            if seg is not None and seg not in {str(s["id"]) for s in SEGMENTS["segments"]}:
                return jsonify(error="Unknown segment."), 400
            db.update_field("findings", fid, "segment", seg, g.user["username"], f["study_id"])
            if f["data"].get("ok"):  # the reference rule depends on the segment
                pipeline.remeasure_points(db, db.one("SELECT * FROM findings WHERE id = ?", (fid,)), get_run(f["run_id"]),
                                          g.user["username"], f["inputs"].get("mld_index"), f["inputs"].get("ref_indices"))
        if "status" in b:
            if b["status"] not in ("accepted", "rejected", "draft"):
                return jsonify(error="Unknown status."), 400
            db.update_field("findings", fid, "status", b["status"], g.user["username"], f["study_id"])
        return jsonify(state_json(f["study_id"]))

    @app.route("/api/run/<int:rid>/calibration", methods=["POST"])
    @reviewer_required
    def api_calibration(rid):
        run = get_run(rid)
        editable(run["study_id"])
        rules = rule_tables.load("measurement_rules")
        b = request.json or {}
        method = b.get("method")
        _, infos = pipeline.study_runs_info(db, run["study_id"])
        info = infos[run["idx"] - 1]
        auto = cal.from_dicom(info, rules)
        scale = run["info"].get("display_scale") or 1.0
        if auto.mm_per_px and scale < 1.0:
            auto.mm_per_px /= scale
        if method == "marker":
            (r0, c0), (r1, c1) = b["p1"], b["p2"]
            manual = cal.from_marker(float(np.hypot(r1 - r0, c1 - c0)), float(b["known_mm"]), rules)
        elif method == "device":
            (r0, c0), (r1, c1) = b["p1"], b["p2"]
            manual = cal.from_device(b.get("device_id", ""), float(np.hypot(r1 - r0, c1 - c0)), 1.0, rules)
        elif method == "auto":
            manual = None
        else:
            return jsonify(error="Unknown calibration method."), 400
        chosen = cal.choose([auto] + ([manual] if manual else []))
        if manual and not manual.reliable:
            return jsonify(error=manual.detail), 422
        db.update_field("runs", rid, "calibration", chosen.to_dict(), g.user["username"], run["study_id"])
        if manual and chosen.method != manual.method:
            return jsonify(state_json(run["study_id"]) | {"message": "The DICOM geometry is available and is preferred over "
                                                                    f"a {manual.method} (PRD order of preference); kept the geometry."})
        for f in db.query("SELECT * FROM findings WHERE run_id = ?", (rid,)):
            if f["data"].get("ok"):
                pipeline.remeasure_points(db, f, get_run(rid), g.user["username"], f["inputs"].get("mld_index"),
                                          f["inputs"].get("ref_indices"))
        return jsonify(state_json(run["study_id"]))

    @app.route("/api/study/<int:sid>/segment", methods=["POST"])
    @reviewer_required
    def api_segment(sid):
        st = editable(sid)
        b = request.json or {}
        seg, status = str(b.get("segment")), b.get("status")
        if seg not in {str(s["id"]) for s in SEGMENTS["segments"]} or (status not in SEGMENT_STATUS_IDS and status is not None):
            return jsonify(error="Unknown segment or status."), 400
        new = dict(st["segment_status"])
        if status is None:
            new.pop(seg, None)
        else:
            new[seg] = status
        db.update_field("studies", sid, "segment_status", new, g.user["username"], sid)
        return jsonify(state_json(sid))

    @app.route("/api/study/<int:sid>/danger", methods=["POST"])
    @reviewer_required
    def api_danger(sid):
        st = editable(sid)
        b = request.json or {}
        item, answer = b.get("item"), b.get("answer")
        if item not in {d["id"] for d in DANGER["items"]} or answer not in DANGER_ANSWER_IDS:
            return jsonify(error="Unknown item or answer."), 400
        db.update_field("studies", sid, "danger", dict(st["danger"]) | {item: answer}, g.user["username"], sid)
        return jsonify(state_json(sid))

    @app.route("/api/warning/<int:wid>/ack", methods=["POST"])
    @reviewer_required
    def api_ack(wid):
        w = db.one("SELECT * FROM warnings WHERE id = ?", (wid,))
        if w is None:
            abort(404)
        editable(w["study_id"])
        response = ((request.json or {}).get("response") or "").strip()
        if not response:
            return jsonify(error="Write what you found when you checked this warning."), 400
        db.update_field("warnings", wid, "response", response, g.user["username"], w["study_id"])
        db.update_field("warnings", wid, "acknowledged_at", time.time(), g.user["username"], w["study_id"])
        db.update_field("warnings", wid, "acknowledged_by", g.user["username"], g.user["username"], w["study_id"])
        return jsonify(state_json(w["study_id"]))

    @app.route("/api/study/<int:sid>/sign", methods=["POST"])
    @reviewer_required
    def api_sign(sid):
        st = editable(sid)
        typed = " ".join(((request.json or {}).get("name") or "").split())
        if typed.casefold() != g.user["username"].casefold():
            return jsonify(error=f"To sign, type your name exactly as entered: {g.user['username']}."), 403
        state = state_json(sid)
        if state["blockers"]:
            return jsonify(error="Not ready to sign.", blockers=state["blockers"]), 409
        now = time.time()
        db.update_field("studies", sid, "signed_by", g.user["username"], g.user["username"], sid)
        db.update_field("studies", sid, "signed_at", now, g.user["username"], sid)
        st = get_study(sid)
        st["laterality"] = state["laterality"]
        runs = db.query("SELECT * FROM runs WHERE study_id = ? ORDER BY idx", (sid,))

        def loader(run, n):
            return pipeline.run_cache(Path(run["cache_dir"]))["frames"][n]

        jpath, ppath = write_report(
            data_dir / "reports", st, runs, db.query("SELECT * FROM findings WHERE study_id = ?", (sid,)),
            db.query("SELECT * FROM warnings WHERE study_id = ?", (sid,)), g.user, loader)
        db.update_field("studies", sid, "report_json", str(jpath), g.user["username"], sid)
        db.update_field("studies", sid, "report_pdf", str(ppath), g.user["username"], sid)
        db.update_field("studies", sid, "state", "signed", g.user["username"], sid)
        return jsonify(state_json(sid))

    @app.route("/study/<int:sid>/report.<fmt>")
    @reviewer_required
    def report_file(sid, fmt):
        st = get_study(sid)
        if st["state"] != "signed":
            abort(Response("No report: the study has not been signed.", 409))
        path = st["report_pdf"] if fmt == "pdf" else st["report_json"] if fmt == "json" else None
        if not path:
            abort(404)
        return send_file(path, as_attachment=True)

    @app.route("/study/<int:sid>/audit")
    @reviewer_required
    def audit_page(sid):
        get_study(sid)
        rows = db.query("SELECT * FROM audit WHERE study_id = ? ORDER BY id", (sid,))
        for r in rows:
            r["when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        return render_template("audit.html", rows=rows, sid=sid, user=g.user)

    @app.route("/api/finding/<int:fid>/profile")
    @reviewer_required
    def api_profile(fid):
        f = db.one("SELECT * FROM findings WHERE id = ?", (fid,))
        if f is None or not f["data"].get("profile"):
            abort(404)
        p = profile_from_json(f["data"]["profile"])
        return jsonify(arc=p["arc_px"].tolist(), width=[None if not np.isfinite(v) else float(v) for v in p["width_px"]])

    app.db = db
    return app
