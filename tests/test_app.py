"""End-to-end review workflow through the web app, golden rules enforced server side."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from access.anonymise import anonymise_folder  # noqa: E402
from access.checklist import DANGER_IDS, SEGMENT_IDS  # noqa: E402
from access.webapp import create_app  # noqa: E402

DOCTOR = "Dr. Test Reviewer"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from make_synthetic_study import make_study

    root = tmp_path_factory.mktemp("app")
    make_study(root / "raw", seed=5)
    anonymise_folder(root / "raw", root / "anon", b"s" * 32, progress=lambda *_: None)
    app = create_app(root / "data")
    study_dir = next(p for p in (root / "anon").iterdir() if p.is_dir())
    return app, study_dir, root


def login(app, name=DOCTOR):
    c = app.test_client()
    r = c.post("/start", data={"name": name})
    assert r.status_code == 302
    return c


def wait(c, job):
    for _ in range(600):
        s = c.get(f"/api/jobs/{job}").get_json()
        if s["state"] != "running":
            return s
        time.sleep(0.2)
    raise AssertionError("import did not finish")


def test_full_review_workflow(env):
    app, study_dir, root = env
    c = login(app)
    assert c.get("/api/study/1").status_code in (200, 404)

    # import
    s = wait(c, c.post("/api/import", json={"folder": str(study_dir)}).get_json()["job"])
    assert s["state"] == "done", s
    sid = s["study_ids"][0]
    assert c.get(f"/study/{sid}").status_code == 200  # opening the review page starts the review clock
    st = c.get(f"/api/study/{sid}").get_json()
    assert len(st["runs"]) == 3
    assert st["laterality"]["status"] == "needs_confirmation"  # header alone is one source

    # golden rule 1: no measurement until the side is confirmed
    rid = st["runs"][0]["id"]
    r = c.post(f"/api/run/{rid}/measure", json={})
    assert r.status_code == 409

    # disagreement stops; agreement (header + doctor) confirms and drafts are measured
    st = c.post(f"/api/study/{sid}/laterality", json={"side": "RIGHT"}).get_json()
    assert st["laterality"]["status"] == "conflict" and not st["findings"]
    st = c.post(f"/api/study/{sid}/laterality", json={"side": "LEFT", "note_side": "LEFT"}).get_json()
    assert st["laterality"]["status"] == "confirmed"
    drafts = [f for f in st["findings"] if f["data"].get("ok")]
    assert len(drafts) == 3
    f1 = next(f for f in drafts if f["data"]["run_idx"] == 1)
    assert abs(f1["data"]["ds_percent"] - 65) < 5 and f1["data"]["mm_available"]

    # cannot sign yet: every segment, danger sign, warning and draft must be answered
    r = c.post(f"/api/study/{sid}/sign", json={"name": DOCTOR})
    assert r.status_code == 409 and r.get_json()["blockers"]

    # doctor moves the MLD point: recalculated quickly (R9) and logged (R10)
    t0 = time.time()
    res = c.post(f"/api/finding/{f1['id']}/points", json={"mld_index": f1["data"]["mld_index"] + 10}).get_json()
    assert res["ok"] and res["finding"]["mld_source"] == "doctor" and time.time() - t0 < 2.5
    c.post(f"/api/finding/{f1['id']}/points", json={"mld_index": None})

    # assign segments, accept / reject drafts
    for f in st["findings"]:
        if f["data"].get("ok") and f["data"]["run_idx"] in (1, 2):
            c.post(f"/api/finding/{f['id']}", json={"segment": "3" if f["data"]["run_idx"] == 1 else "6"})
            c.post(f"/api/finding/{f['id']}", json={"status": "accepted"})
        else:
            c.post(f"/api/finding/{f['id']}", json={"status": "rejected"})
    # a narrowing in a segment marked open is a contradiction
    for s_id in SEGMENT_IDS:
        c.post(f"/api/study/{sid}/segment", json={"segment": s_id, "status": "open"})
    st = c.get(f"/api/study/{sid}").get_json()
    assert any("marked open" in b for b in st["blockers"])
    c.post(f"/api/study/{sid}/segment", json={"segment": 3, "status": "narrow"})
    c.post(f"/api/study/{sid}/segment", json={"segment": 6, "status": "narrow"})
    c.post(f"/api/study/{sid}/segment", json={"segment": 7, "status": "not_visible"})
    for d in DANGER_IDS:
        c.post(f"/api/study/{sid}/danger", json={"item": d, "answer": "absent"})
    st = c.get(f"/api/study/{sid}").get_json()
    for w in st["warnings"]:
        assert c.post(f"/api/warning/{w['id']}/ack", json={"response": ""}).status_code == 400  # needs an answer
        c.post(f"/api/warning/{w['id']}/ack", json={"response": "checked, no leak"})
    st = c.get(f"/api/study/{sid}").get_json()
    assert st["blockers"] == [], st["blockers"]

    # the reviewer signs by typing their name again
    assert c.post(f"/api/study/{sid}/sign", json={"name": "someone else"}).status_code == 403
    assert c.get(f"/study/{sid}/report.pdf").status_code == 409  # rule 7: no report before signing
    st = c.post(f"/api/study/{sid}/sign", json={"name": DOCTOR.lower()}).get_json()
    assert st["study"]["state"] == "signed" and st["study"]["signed_by"] == DOCTOR

    pdf = c.get(f"/study/{sid}/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    rep = json.loads(c.get(f"/study/{sid}/report.json").data)
    assert rep["laterality"]["side"] == "LEFT" and len(rep["findings"]) == 2
    assert all(str(s) in rep["segment_status"] for s in SEGMENT_IDS) and rep["review_seconds"] is not None
    assert "recommend" not in json.dumps(rep["findings"]).lower()

    # signed study is locked
    assert c.post(f"/api/study/{sid}/segment", json={"segment": 1, "status": "narrow"}).status_code == 409

    # audit log: rich and append-only
    rows = app.db.query("SELECT * FROM audit WHERE study_id = ?", (sid,))
    assert any(r["field"] == "status" and r["new"] == "accepted" for r in rows)
    assert any(r["field"] == "points" for r in rows)
    assert {r["username"] for r in rows} <= {DOCTOR, "system"}  # "by whom" is the reviewer's name
    with pytest.raises(sqlite3.DatabaseError):
        with app.db.connect() as con:
            con.execute("UPDATE audit SET new = 'x' WHERE id = 1")
    assert c.get(f"/study/{sid}/audit").status_code == 200


def test_name_required_and_cross_site_post_refused(env):
    app, _, _ = env
    anon = app.test_client()
    assert anon.get("/api/study/1").status_code == 401
    assert anon.get("/").status_code == 302
    assert anon.post("/start", data={"name": " "}).status_code == 200  # a name is needed
    c = login(app)
    r = c.post("/api/study/1/segment", json={"segment": 1, "status": "open"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_pages_render_without_external_resources(env):
    app, _, _ = env
    c = login(app)
    for url in ("/", "/study/1"):
        html = c.get(url).get_data(as_text=True)
        assert "http://" not in html and "https://" not in html
    assert "default-src 'self'" in c.get("/").headers["Content-Security-Policy"]


def test_import_by_upload_and_zip_and_refuse_non_anonymised(env):
    import io
    import zipfile

    app, study_dir, root = env
    c = login(app)
    files = sorted(study_dir.glob("*.dcm"))
    # uploaded files (as the folder picker sends them), plus a non-DICOM file that is ignored
    data = {"files": [(open(f, "rb"), f"study/{f.name}") for f in files] + [(io.BytesIO(b"not dicom"), "study/notes.txt")]}
    r = c.post("/api/import-upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["files"] == len(files)
    assert wait(c, r.get_json()["job"])["state"] == "done"

    # the same study as a .zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in files:
            z.write(f, f"inner/{f.name}")
    buf.seek(0)
    r = c.post("/api/import-upload", data={"files": [(buf, "study.zip")]}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["files"] == len(files)
    assert wait(c, r.get_json()["job"])["state"] == "done"

    # raw (not anonymised) files are refused
    raw = sorted((root / "raw").glob("*.dcm"))[:1]
    r = c.post("/api/import-upload", data={"files": [(open(raw[0], "rb"), raw[0].name)]}, content_type="multipart/form-data")
    s = wait(c, r.get_json()["job"])
    assert s["state"] == "error" and "not anonymised" in s["message"]

    # nothing usable chosen
    r = c.post("/api/import-upload", data={"files": [(io.BytesIO(b"x"), "a.txt")]}, content_type="multipart/form-data")
    assert r.status_code == 400
