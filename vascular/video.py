"""Video / cine input: decoding, video-level checks, equal-duration splitting and
best-frame selection per part.

Rule-based like the rest of the pipeline: every video-level flag comes from a measured
frame statistic, and each selected frame goes through the same single-image analysis as
an uploaded image (run_analysis.analyze_array). Frames are analysed at a reduced size
(ANALYSIS_SIDE) to keep a 10-part video within one web request; pixel spacing is
rescaled to match, so mm measurements stay correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

from .io_utils import _normalize
from .preprocess import POLARITIES, polarity_from_scores, polarity_scores

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v", ".mpg", ".mpeg", ".wmv"}
DICOM_EXTS = {".dcm", ".dicom"}

MAX_DURATION_S = 420.0  # 7 minutes
MAX_FRAMES_NO_FPS = 12600  # frame cap when the file carries no frame rate (7 min at 30 fps)
MIN_PARTS, MAX_PARTS = 1, 50
ANALYSIS_SIDE = 768  # longest side of a frame sent to the full analysis
SCORE_SIDE = 256  # longest side used to score candidate frames
THUMB_SIDE = 96  # longest side used for per-frame brightness / motion statistics
# candidate frames held in memory (uniformly spaced over the video): at least
# MIN_CANDIDATES, and enough that every part keeps several to choose from. The kept
# count settles between half and all of the cap, so 10 per part leaves 5-10.
MIN_CANDIDATES = 150
CANDIDATES_PER_PART = 10

# per-frame statistics are measured on the frame as decoded, scaled to [0, 1]
BLANK_STD = 0.01
DARK_MEAN, BRIGHT_MEAN = 0.03, 0.97
# a video is static when no frame differs from the first by more than this (mean abs).
# Measured against the first frame, not the previous one: at 30 fps a slow but real
# change (contrast filling over minutes) is tiny between consecutive frames.
FROZEN_DRIFT = 0.004
CUT_DIFF = 0.25
# frame-to-frame displacement (phase correlation, as a fraction of frame width); unlike
# the raw intensity change this is not raised by contrast flowing in or washing out
MOTION_SHIFT = 0.01
# preprocess.polarity_scores of the chosen polarity. Measured: real coronary X-ray frames
# 0.20-0.34, synthetic opacified frames 0.4-0.7, frames with no vessels 0.15-0.18. The
# absolute floor alone has little margin, so a part also counts as "no signal" when it is
# far below the video's best part (frames before contrast arrival / after washout).
VESSEL_SIGNAL_MIN = 0.19
VESSEL_SIGNAL_REL = 0.5
BLUR_SHARPNESS = 1.5e-4  # preprocess.quality's bar for "adequate" sharpness
LOW_RES_SIDE = 256
SCORE_WEIGHTS = {"vessel": 0.6, "sharpness": 0.25, "contrast": 0.15}

MODALITY_PHRASE = {
    "bright": "an MR/CT angiographic projection (bright vessels)",
    "dark": "an X-ray/DSA angiogram (dark vessels)",
}


def _no_progress(stage: str, done: int, total: int, detail: str = "") -> None:
    """Progress callback signature: `stage` is "reading", "selecting", "checking" or "analysing";
    `total` is 0 when unknown (e.g. a video without a frame count)."""


class VideoError(ValueError):
    """The video cannot be analysed at all; `notes` are user-facing recommendations."""

    def __init__(self, message: str, notes: list[str] | tuple[str, ...] = ()):
        super().__init__(message)
        self.notes = list(notes)


@dataclass
class Scan:
    candidates: list[tuple[int, np.ndarray]]  # (frame index, uint8 frame at analysis size)
    n_frames: int
    fps: float | None
    scale: float  # analysis size / original size
    original_shape: tuple[int, int]
    mean: np.ndarray  # per decoded frame
    std: np.ndarray
    diff: np.ndarray  # mean abs change vs the previous frame (0 for the first)
    shift: np.ndarray  # displacement vs the previous frame, fraction of width (0 for the first)
    drift: np.ndarray  # mean abs change vs the first frame
    spacing: tuple[float, float] | None
    kind: str


def _fit(shape: tuple[int, int], side: int) -> tuple[int, int]:
    """OpenCV (width, height) that fits `shape` inside `side` without upscaling."""
    h, w = shape
    s = min(1.0, side / max(h, w))
    return max(1, round(w * s)), max(1, round(h * s))


def _open_dicom(path: Path):
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise VideoError(
            "The DICOM file could not be read.",
            ["Check that the file is a complete DICOM export (not a DICOMDIR or a truncated transfer) and upload it again."],
        ) from exc
    arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] in (3, 4) and int(getattr(ds, "SamplesPerPixel", 1)) > 1):
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim == 2:
        arr = arr[None]
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    # one normalisation for the whole run, so a frame before contrast arrival stays
    # genuinely darker/emptier than the opacified frames instead of being stretched
    arr = (_normalize(arr) * 255).astype(np.uint8)

    fps = None
    if getattr(ds, "CineRate", None):
        fps = float(ds.CineRate)
    elif getattr(ds, "FrameTime", None):
        fps = 1000.0 / float(ds.FrameTime)
    ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    spacing = (float(ps[0]), float(ps[1])) if ps is not None and len(ps) >= 2 else None
    return fps, spacing, "DICOM cine", iter(arr), len(arr)


def _open_video(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(
            "The video could not be opened.",
            ["Re-export the video as MP4 (H.264) or AVI and upload it again."],
        )
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = fps if 0.0 < fps <= 1000.0 else None
    # container frame count: only used for progress, so an estimate is fine
    n_est = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    def frames():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        finally:
            cap.release()

    return fps, None, "video", frames(), n_est


def scan(path: Path, max_candidates: int = MIN_CANDIDATES, progress=_no_progress) -> Scan:
    """Decode once, keeping per-frame statistics for every frame and a uniformly spaced
    subset of frames (at analysis size) as best-frame candidates."""
    import cv2

    path = Path(path)
    opener = _open_dicom if path.suffix.lower() in DICOM_EXTS else _open_video
    fps, spacing, kind, frames, n_est = opener(path)

    candidates: list[tuple[int, np.ndarray]] = []
    means, stds, diffs, shifts, drifts = [], [], [], [], []
    first = None
    stride, prev = 1, None
    analysis_size = thumb_size = window = None
    original_shape = (0, 0)
    for i, gray in enumerate(frames):
        if i % 25 == 0:
            progress("reading", i, n_est)
        if fps and i / fps > MAX_DURATION_S or (not fps and i >= MAX_FRAMES_NO_FPS):
            limit = f"{MAX_DURATION_S / 60:.0f} minutes" if fps else f"{MAX_FRAMES_NO_FPS} frames"
            raise VideoError(
                f"The video is longer than the {limit} limit.",
                [f"Trim the video to the relevant run (at most {limit}) and upload it again."],
            )
        if analysis_size is None:
            original_shape = gray.shape[:2]
            analysis_size = _fit(original_shape, ANALYSIS_SIDE)
            thumb_size = _fit(original_shape, THUMB_SIDE)
            window = cv2.createHanningWindow(thumb_size, cv2.CV_32F)
        thumb = cv2.resize(gray, thumb_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        means.append(float(thumb.mean()))
        stds.append(float(thumb.std()))
        diffs.append(float(np.abs(thumb - prev).mean()) if prev is not None else 0.0)
        if first is None:
            first = thumb
        drifts.append(float(np.abs(thumb - first).mean()))
        if prev is not None and min(thumb.shape) >= 8:
            # phaseCorrelate windows its inputs in place, so hand it copies
            (dx, dy), _ = cv2.phaseCorrelate(prev.copy(), thumb.copy(), window)
            shifts.append(float(np.hypot(dx, dy)) / thumb.shape[1])
        else:
            shifts.append(0.0)
        prev = thumb
        if i % stride == 0:
            frame = gray if gray.shape[1::-1] == analysis_size else cv2.resize(gray, analysis_size, interpolation=cv2.INTER_AREA)
            candidates.append((i, np.ascontiguousarray(frame)))
            if len(candidates) > max_candidates:
                # candidate indices are multiples of `stride`; dropping every other one
                # leaves exact multiples of 2*stride, so spacing stays uniform
                candidates = candidates[::2]
                stride *= 2

    n = len(means)
    if n == 0:
        raise VideoError(
            "No frames could be decoded from the video.",
            ["The file may be corrupt or use an unsupported codec; re-export it as MP4 (H.264) and upload it again."],
        )
    return Scan(
        candidates=candidates,
        n_frames=n,
        fps=fps,
        scale=analysis_size[0] / original_shape[1],
        original_shape=original_shape,
        mean=np.array(means),
        std=np.array(stds),
        diff=np.array(diffs),
        drift=np.array(drifts),
        shift=np.array(shifts),
        spacing=spacing,
        kind=kind,
    )


def _blank(sc: Scan) -> np.ndarray:
    return (sc.std < BLANK_STD) | (sc.mean < DARK_MEAN) | (sc.mean > BRIGHT_MEAN)


def _as_float(frame: np.ndarray) -> np.ndarray:
    return _normalize(frame.astype(np.float32))


def _video_polarity(frames: list[np.ndarray], vessels: str) -> dict:
    """One polarity for the whole video (averaged over up to 5 frames spread across it),
    so every part is analysed and worded consistently."""
    if vessels in POLARITIES:
        return {"polarity": vessels, "source": "manual", "confident": True}
    picks = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(5, len(frames))).round().astype(int))]
    per_frame = [polarity_scores(_as_float(f)) for f in picks]
    pol = polarity_from_scores({p: float(np.mean([s[p] for s in per_frame])) for p in POLARITIES})
    pol["frames_sampled"] = len(picks)
    return pol


def _frame_scores(frames: list[np.ndarray], contrast: np.ndarray, polarity: str, progress=_no_progress) -> np.ndarray:
    """Best-frame score: mostly how much vessel-like ridge structure the frame has (this
    is what separates an opacified frame from one before contrast arrival), then
    sharpness and contrast. Each term is rank-normalised across the video so no single
    statistic's scale dominates."""
    import cv2
    from skimage.filters import frangi, laplace

    vessel, sharp = [], []
    for j, f in enumerate(frames):
        if j % 10 == 0:
            progress("selecting", j, len(frames))
        small = _as_float(cv2.resize(f, _fit(f.shape, SCORE_SIDE), interpolation=cv2.INTER_AREA))
        vessel.append(float(frangi(small, sigmas=[1.0, 2.0], black_ridges=(polarity == "dark")).mean()))
        sharp.append(float(laplace(ndi.gaussian_filter(small, 1.0)).var()))
    n = len(frames)
    r = {k: rankdata(v) / n for k, v in (("vessel", vessel), ("sharpness", sharp), ("contrast", contrast))}
    return sum(SCORE_WEIGHTS[k] * r[k] for k in SCORE_WEIGHTS)


def _time_label(index: int, fps: float | None) -> str:
    if not fps:
        return f"frame {index}"
    t = index / fps
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _list(nums: list[int]) -> str:
    return ", ".join(str(n) for n in nums)


def analyze_video(
    path: Path,
    outdir: Path,
    stem: str,
    parts: int = 5,
    spacing_mm: float | None = None,
    vessels: str = "auto",
    progress=_no_progress,
    **params,
) -> dict:
    from run_analysis import analyze_array

    path = Path(path)
    parts = int(np.clip(parts, MIN_PARTS, MAX_PARTS))
    sc = scan(path, max(MIN_CANDIDATES, CANDIDATES_PER_PART * parts), progress)
    notes: list[str] = []

    if sc.n_frames < 2:
        raise VideoError(
            "The file contains a single frame, so there is nothing to split.",
            ["Upload it in Image mode instead, or upload the full cine run."],
        )

    blank = _blank(sc)
    cands = [(i, f) for i, f in sc.candidates if not blank[i]]
    if not cands:
        raise VideoError(
            "Every frame of the video is blank, black or washed out.",
            [
                "Check the export settings (window/level, brightness) and that the run actually contains image data, then upload it again.",
            ],
        )

    if parts > sc.n_frames:
        notes.append(f"The video has only {sc.n_frames} frames, so it was split into {sc.n_frames} parts instead of {parts}.")
        parts = sc.n_frames

    pol = _video_polarity([f for _, f in cands], vessels)
    scores = _frame_scores([f for _, f in cands], sc.std[[i for i, _ in cands]], pol["polarity"], progress)

    if spacing_mm:
        spacing = (float(spacing_mm) / sc.scale,) * 2
    elif sc.spacing:
        spacing = (sc.spacing[0] / sc.scale, sc.spacing[1] / sc.scale)
    else:
        spacing = None

    # pass 1: pick each part's best frame and measure its vessel signal
    bounds = np.linspace(0, sc.n_frames, parts + 1)
    results, picks = [], {}
    for k in range(parts):
        progress("checking", k, parts)
        lo, hi = int(round(bounds[k])), int(round(bounds[k + 1]))
        part = {
            "index": k + 1,
            "start_frame": lo,
            "end_frame": hi - 1,
            "start_label": _time_label(lo, sc.fps),
            "end_label": _time_label(hi, sc.fps) if sc.fps else _time_label(hi - 1, sc.fps),
        }
        results.append(part)
        in_part = [j for j, (i, _) in enumerate(cands) if lo <= i < hi]
        if not in_part:
            part["skipped"] = "No usable frame in this part: every frame was blank, black or washed out."
            continue
        best = max(in_part, key=lambda j: scores[j])
        idx, frame = cands[best]
        picks[k] = frame
        part.update(frame_index=idx, time_label=_time_label(idx, sc.fps), score=round(float(scores[best]), 3))
        part["vessel_signal"] = round(polarity_scores(_as_float(frame))[pol["polarity"]], 3)

    # pass 2: analyse the parts that show vessels
    peak = max((p["vessel_signal"] for p in results if "vessel_signal" in p), default=0.0)
    floor = max(VESSEL_SIGNAL_MIN, VESSEL_SIGNAL_REL * peak)
    n_todo = sum(1 for k in picks if results[k]["vessel_signal"] >= floor)
    n_done = 0
    for k, frame in picks.items():
        part = results[k]
        if part["vessel_signal"] < floor:
            # segmenting a frame without opacified vessels only traces background
            # texture, so report "no vessel data" instead of noise measurements
            part["no_signal"] = True
            part["skipped"] = (
                "No clear vessel signal in this part, even in its best frame (typically before contrast "
                "arrival, after washout, or a section without angiographic content)."
            )
            continue
        meta = {
            "source": str(path),
            # report text refers to "this video" rather than the uploaded file name
            "label": f"the part {k + 1} frame of this video at {part['time_label']}",
            "title": f"Part {k + 1} - frame at {part['time_label']}",
            "format": f"{sc.kind} frame",
        }
        progress("analysing", n_done, n_todo, f"part {k + 1} of {parts}")
        n_done += 1
        try:
            part["result"] = analyze_array(_as_float(frame), spacing, meta, outdir, f"{stem}_part{k + 1}", vessels=pol, **params)
        except Exception as exc:
            part["skipped"] = f"Analysis of the selected frame failed: {exc}"

    done = [p for p in results if p.get("result")]
    with_vessels = [p["index"] for p in done if p["result"]["metrics"]["n_branches"] > 0]
    with_narrowing = [p["index"] for p in done if p["result"]["metrics"]["candidate_stenoses"]]
    no_vessels = [p["index"] for p in done if p["result"]["metrics"]["n_branches"] == 0]
    no_signal = [p["index"] for p in results if p.get("no_signal")]
    unusable = [p["index"] for p in results if not p.get("result") and not p.get("no_signal")]

    # ---- video-level (generic) recommendations ----
    if not with_vessels and (done or no_signal):
        notes.append(
            "No analysable vessel structure was found in any part of the video. Contrast may not have been "
            "injected or may not have reached the field of view, the vessel appearance setting may be wrong, or "
            "the video may not be an angiogram; review the run manually."
        )
    elif no_vessels or no_signal:
        notes.append(
            f"Part(s) {_list(sorted(no_vessels + no_signal))} contained no analysable vessel structure (often frames "
            "before contrast arrival or after washout); the overall picture is based on the remaining parts."
        )
    if unusable:
        notes.append(f"Part(s) {_list(unusable)} produced no result (no usable frame or a failed analysis); see the part details.")
    if pol["source"] == "auto" and not pol["confident"]:
        notes.append(
            f"Vessel appearance could not be determined confidently; the video was analysed as "
            f"{MODALITY_PHRASE[pol['polarity']]}. If that is wrong, re-run with the vessel appearance option set manually."
        )
    blurred = [p["index"] for p in done if p["result"]["quality"]["sharpness"] < BLUR_SHARPNESS]
    if done and len(blurred) * 2 >= len(done):
        notes.append(
            f"The video appears blurred: even the sharpest frame was below the sharpness threshold in part(s) "
            f"{_list(blurred)}. Motion, heavy compression or a low-resolution export reduce measurement reliability; "
            "use the original-quality export if available."
        )
    blank_frac = float(blank.mean())
    if blank_frac > 0.5:
        notes.append(
            f"{blank_frac * 100:.0f}% of frames are blank, black or washed out; check the export window/level settings "
            "and trim empty sections before uploading."
        )
    moving = sc.diff[1:]
    if len(moving) and float(sc.drift.max()) < FROZEN_DRIFT:
        notes.append(
            "The video is essentially static (frames barely change), so splitting adds no information; "
            "a single image upload is more appropriate."
        )
    elif with_vessels and len(moving) and float(np.median(sc.shift[1:])) > MOTION_SHIFT:
        notes.append(
            "Significant frame-to-frame motion was detected; patient or table motion makes results less "
            "consistent between parts."
        )
    n_cuts = int((moving > CUT_DIFF).sum()) if len(moving) else 0
    if n_cuts:
        notes.append(
            f"{n_cuts} abrupt change(s) between consecutive frames were detected (cuts, view switches or screen "
            "transitions); if the video combines several runs or views, upload each one separately."
        )
    if max(sc.original_shape) < LOW_RES_SIDE:
        h, w = sc.original_shape
        notes.append(f"The video resolution is low ({w}x{h}); small vessels and narrowings may not be measurable.")
    if sc.fps and sc.n_frames / sc.fps < 1.0:
        notes.append("The video is shorter than one second; parts may be only a few frames apart.")

    # ---- summary ----
    duration = f"{sc.n_frames / sc.fps:.1f} s, " if sc.fps else ""
    fps_txt = f" at {sc.fps:.0f} fps" if sc.fps else ""
    how = "auto-detected" if pol["source"] == "auto" else "set manually"
    summary = (
        f"This video ({duration}{sc.n_frames} frames{fps_txt}) was split into {parts} equal part(s), and the "
        f"best-quality frame of each part was analysed as {MODALITY_PHRASE[pol['polarity']]}, vessel appearance {how}."
    )
    if not with_vessels and no_signal:
        summary += " No clear vessel signal was found in any part, so no vessel measurements are reported."
    elif not done:
        summary += " None of the parts produced a result."
    elif not with_vessels:
        summary += " No analysable vessel structure was segmented in any part."
    else:
        summary += f" Vessel structure was segmented in {len(with_vessels)} of {parts} part(s)"
        if with_narrowing:
            summary += f", and candidate focal narrowings were flagged in part(s) {_list(with_narrowing)}."
        else:
            summary += ", with no candidate focal narrowing flagged in any part."
        grades = [p["result"]["quality"]["label"] for p in done]
        summary += " Frame quality: " + ", ".join(f"{grades.count(g)} {g}" for g in ("adequate", "limited", "poor") if g in grades) + "."
    summary += " Automated research output; not a clinical diagnosis."

    report = {
        "file": str(path),
        "frames": sc.n_frames,
        "fps": sc.fps,
        "analysis_scale": round(sc.scale, 4),
        "polarity": pol,
        "summary": summary,
        "notes": notes,
        "parts": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in p.items() if k != "result"}
            | (
                {
                    "observation": p["result"]["observation"],
                    "recommendation": p["result"]["recommendation"],
                    "quality": p["result"]["quality"],
                    "n_branches": p["result"]["metrics"]["n_branches"],
                    "n_candidate_stenoses": len(p["result"]["metrics"]["candidate_stenoses"]),
                }
                if p.get("result")
                else {}
            )
            for p in results
        ],
    }
    with open(Path(outdir) / f"{stem}_video.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return {"summary": summary, "notes": notes, "parts": results, "polarity": pol, "n_frames": sc.n_frames, "fps": sc.fps}
