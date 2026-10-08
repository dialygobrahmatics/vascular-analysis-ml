"""CLI: automatic vascular analysis of angiography images and videos (research use only).

Handles both vessel appearances: bright vessels (MR/CT angiography MIP) and dark vessels
(X-ray / DSA angiography), detected automatically unless --vessels is given.

Usage:
    python run_analysis.py <file-or-folder> [--outdir outputs] [--spacing-mm 0.3]
                           [--vessels auto|bright|dark] [--parts 5]

Outputs per image: <name>_observation.txt (single paragraph), <name>_metrics.json,
<name>_overlay.png (segmentation + centerline + candidate narrowing markers).
Videos are split into equal parts and the best frame of each part is analysed the same
way (<name>_partN_*), plus <name>_video.json with the video-level summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

from vascular.io_utils import load_image
from vascular.preprocess import POLARITIES, detect_polarity, enhance, field_of_view, quality
from vascular.quantify import analyze
from vascular.report import build_observation, build_recommendations
from vascular.sanity import label_like_blobs, panel_layout, seam_lines
from vascular.segment import segment

EXTS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VESSEL_CHOICES = ("auto",) + POLARITIES


def _overlay(title: str, proc: np.ndarray, seg: dict, metrics: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    axes[0].imshow(proc, cmap="gray")
    axes[0].set_title("input (enhanced)", color="white")
    axes[1].imshow(proc, cmap="gray")
    for c in find_contours(seg["mask"], 0.5):
        axes[1].plot(c[:, 1], c[:, 0], color="orange", lw=0.8)
    axes[1].set_title("vessel segmentation", color="white")
    axes[2].imshow(proc, cmap="gray")
    sk = np.argwhere(seg["skeleton"])
    if len(sk):
        axes[2].scatter(sk[:, 1], sk[:, 0], s=0.3, c="red")
    for s in metrics["candidate_stenoses"][:5]:
        r, c = s["min_rc"]
        axes[2].scatter([c], [r], s=90, facecolors="none", edgecolors="cyan", lw=1.2)
    axes[2].set_title("centerline + candidate narrowing", color="white")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}  |  research output - not for diagnostic use", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="black")
    plt.close(fig)


def resolve_polarity(base: np.ndarray, vessels: str | dict) -> dict:
    """`vessels` is "auto", "bright", "dark", or an already-resolved polarity dict (the
    video path decides once per video so every part is analysed consistently)."""
    if isinstance(vessels, dict):
        return vessels
    if vessels in POLARITIES:
        return {"polarity": vessels, "source": "manual", "confident": True}
    return detect_polarity(base)


def analyze_array(
    img: np.ndarray,
    spacing: tuple[float, float] | None,
    meta: dict,
    outdir: Path,
    stem: str,
    vessels: str | dict = "auto",
    min_branch_px: int = 12,
    stenosis_ratio: float = 0.7,
    stenosis_min_run: int = 6,
    min_ref_diam_px: float = 5.0,
) -> dict:
    """Run the full pipeline on one 2D image scaled to [0, 1] and write its outputs."""
    fov = field_of_view(img)
    # flatten the collimator/FOV step edge with the interior median before
    # CLAHE, otherwise large Frangi scales read that edge as a vessel ridge
    base = np.where(fov, img, np.median(img[fov]))
    pol = resolve_polarity(base, vessels)
    shown = enhance(base)
    # segmentation and quantification expect bright vessels, so X-ray/DSA frames are
    # inverted after enhancement; the FOV is still found on the original image, whose
    # dark collimator border would turn bright (and be read as content) once inverted
    proc = 1.0 - shown if pol["polarity"] == "dark" else shown
    qual = quality(proc, fov)
    seg = segment(proc, fov, raw=img)

    density = float(np.logical_and(seg["mask"], fov).sum()) / max(int(fov.sum()), 1)
    metrics = analyze(
        seg["mask"],
        seg["skeleton"],
        spacing_mm=spacing,
        min_branch_px=min_branch_px,
        stenosis_ratio=stenosis_ratio,
        stenosis_min_run=stenosis_min_run,
        min_ref_diam_px=min_ref_diam_px,
    )
    unit = "mm" if spacing else "px"
    paragraph = build_observation(
        metrics, qual, meta, unit=unit, density=density, calibrated=bool(spacing), polarity=pol["polarity"]
    )

    rows_idx, cols_idx = np.where(seg["mask"])
    vessel_aspect = None
    if len(rows_idx):
        vh = int(rows_idx.max() - rows_idx.min() + 1)
        vw = int(cols_idx.max() - cols_idx.min() + 1)
        vessel_aspect = vh / max(vw, 1)

    checks = {
        # raw img, not proc: CLAHE brightens a true empty gap enough to erase the
        # "no content here" signal a gap between separate panels depends on
        "panels": panel_layout(img),
        "artifacts": seg["artifacts"],
        "seams": seam_lines(proc, fov),
        # burned-in annotations are bright on the image as displayed, whatever the
        # vessel polarity, so look for them on the un-inverted image
        "labels": label_like_blobs(shown, fov, seg["mask"]),
        "image_height": img.shape[0],
        "image_width": img.shape[1],
        "vessel_aspect": round(vessel_aspect, 3) if vessel_aspect else None,
        "polarity": pol,
    }
    recommendation = build_recommendations(checks, metrics, qual)
    full_text = paragraph + (f"\n\n{recommendation}" if recommendation else "")

    (outdir / f"{stem}_observation.txt").write_text(full_text, encoding="utf-8")
    with open(outdir / f"{stem}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "file": meta.get("source"),
                "quality": qual,
                "units": unit,
                "metrics": metrics,
                "checks": checks,
                "observation": paragraph,
                "recommendation": recommendation,
            },
            fh,
            indent=2,
        )
    overlay_png = outdir / f"{stem}_overlay.png"
    _overlay(meta.get("title") or meta.get("label") or Path(meta.get("source", stem)).name, shown, seg, metrics, overlay_png)
    return {
        "observation": paragraph,
        "recommendation": recommendation,
        "metrics": metrics,
        "quality": qual,
        "polarity": pol,
        "density": density,
        "overlay": overlay_png,
    }


def analyze_file(path: Path, outdir: Path, spacing_mm: float | None = None, *, vessels: str = "auto", **params) -> dict:
    img, spacing, meta = load_image(path)
    if spacing_mm:
        spacing = (float(spacing_mm), float(spacing_mm))
    return analyze_array(img, spacing, meta, outdir, path.stem, vessels=vessels, **params)


def main(argv=None) -> int:
    from vascular.video import VIDEO_EXTS, VideoError, analyze_video

    ap = argparse.ArgumentParser(description="Automatic vessel analysis of angiography images/videos (research use only).")
    ap.add_argument("input", help="image/video file or folder (DICOM / PNG / TIFF / MP4 / AVI / MOV ...)")
    ap.add_argument("--outdir", default="outputs", help="output folder (default: ./outputs)")
    ap.add_argument("--spacing-mm", type=float, default=None, help="override pixel spacing (mm/px)")
    ap.add_argument("--vessels", choices=VESSEL_CHOICES, default="auto", help="vessel appearance: bright (MR/CT), dark (X-ray/DSA) or auto")
    ap.add_argument("--parts", type=int, default=5, help="videos: number of equal parts to split into (1-10, default 5)")
    ap.add_argument("--min-branch-px", type=int, default=12, help="ignore centerline branches shorter than this")
    ap.add_argument("--stenosis-ratio", type=float, default=0.7, help="candidate narrowing threshold (default 0.7)")
    ap.add_argument("--min-lesion-px", type=int, default=6, help="minimum run of narrowed calibre (px) for a candidate (default 6)")
    ap.add_argument(
        "--min-ref-diam-px",
        type=float,
        default=5.0,
        help="ignore candidate narrowings whose reference calibre is below this many pixels (default 5.0)",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    supported = EXTS | VIDEO_EXTS
    files = (
        [in_path]
        if in_path.is_file()
        else sorted(p for p in in_path.iterdir() if p.suffix.lower() in supported)
    )
    if not files:
        print(f"No supported images or videos found at {in_path}")
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = dict(
        min_branch_px=args.min_branch_px,
        stenosis_ratio=args.stenosis_ratio,
        stenosis_min_run=args.min_lesion_px,
        min_ref_diam_px=args.min_ref_diam_px,
    )

    for f in files:
        if f.suffix.lower() in VIDEO_EXTS:
            try:
                res = analyze_video(f, outdir, f.stem, parts=args.parts, spacing_mm=args.spacing_mm, vessels=args.vessels, **params)
            except VideoError as exc:
                print(f"\n=== {f.name} (video: not analysed) ===\n{exc}")
                for note in exc.notes:
                    print(f"- {note}")
                continue
            print(f"\n=== {f.name} (video, {len(res['parts'])} parts) ===")
            print(res["summary"])
            for note in res["notes"]:
                print(f"- {note}")
            for part in res["parts"]:
                print(f"\n--- part {part['index']} ({part['start_label']}-{part['end_label']}) ---")
                if part.get("result"):
                    print(part["result"]["observation"])
                    if part["result"]["recommendation"]:
                        print(part["result"]["recommendation"])
                else:
                    print(part["skipped"])
            continue

        try:
            res = analyze_file(f, outdir, args.spacing_mm, vessels=args.vessels, **params)
        except Exception as exc:
            print(f"[skip] {f.name}: {exc}")
            continue
        print(f"\n=== {f.name} (quality: {res['quality']['label']}, vessels: {res['polarity']['polarity']}) ===")
        print(res["observation"])
        if res["recommendation"]:
            print()
            print(res["recommendation"])
        print(f"[saved] {outdir / (f.stem + '_observation.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
