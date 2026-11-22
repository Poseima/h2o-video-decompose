#!/usr/bin/env python3
"""General-purpose local 3D Gaussian Splatting (3DGS) reconstruction pipeline for h2o-video-decompose.

Run the following stages with the tools available on the current machine:
  Stage 1 (frames) : Extract clean rigid-segment frames, optionally crop static watermark bands,
                     and filter motion-blurred outliers via Laplacian sharpness.
  Stage 2 (colmap) : Run COLMAP sequential SfM + image_undistorter + TXT export, and compute
                     camera translation baseline vs. scene depth to warn about limited novel-view angles.
  Stage 3 (brush)  : Train 3DGS with ArthurBrussee/brush and compute held-out evaluation PSNR.
                     Output a PLY that can be imported separately into Blender.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np


def run_cmd(cmd, cwd=None, log_path=None):
    print(f"[RUN] {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    if log_path is not None:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    dt = time.time() - t0
    print(f"[DONE] rc={proc.returncode} ({dt:.1f}s)", flush=True)
    if proc.returncode != 0:
        if log_path is not None and Path(log_path).exists():
            tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd[0]}")
    return dt


def extract_and_filter_frames(video_path: Path, out_images: Path, start_s: float, end_s: float,
                              sample_fps: float, max_side: int, crop_bottom_ratio: float,
                              crop_top_ratio: float = 0.0, min_keep_ratio: float = 0.80):
    """Extract rigid-segment frames with FFmpeg, strip static watermark margins, and drop blur outliers."""
    raw_dir = out_images.parent / "_raw_frames"
    for d in (raw_dir, out_images):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video_path)
    ])
    stream = json.loads(probe)["streams"][0]
    src_w, src_h = int(stream["width"]), int(stream["height"])
    crop_y = int(round(src_h * crop_top_ratio)) // 2 * 2
    crop_h = int(round(src_h * (1.0 - crop_top_ratio - crop_bottom_ratio))) // 2 * 2
    if crop_h < 64:
        raise ValueError(f"Invalid crop ratios: top={crop_top_ratio}, bottom={crop_bottom_ratio}")

    scale_factor = min(1.0, float(max_side) / float(max(src_w, crop_h)))
    dst_w = max(64, int(round(src_w * scale_factor)) // 2 * 2)
    dst_h = max(64, int(round(crop_h * scale_factor)) // 2 * 2)

    vf = f"fps={sample_fps},crop={src_w}:{crop_h}:0:{crop_y},scale={dst_w}:{dst_h}:flags=lanczos"
    run_cmd([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "2",
        str(raw_dir / "frame_%04d.jpg")
    ])

    raw_files = sorted(raw_dir.glob("frame_*.jpg"))
    if not raw_files:
        raise RuntimeError("No frames extracted; verify start/end timestamps.")

    scores = []
    for p in raw_files:
        g = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        scores.append((p, float(cv.Laplacian(g, cv.CV_64F).var())))

    vars_only = np.array([s[1] for s in scores], dtype=np.float64)
    cutoff = float(np.percentile(vars_only, (1.0 - min_keep_ratio) * 100.0))
    kept_count = 0
    for idx, (p, score) in enumerate(scores):
        if idx == 0 or idx == len(scores) - 1 or score >= cutoff:
            shutil.copy2(p, out_images / f"img_{kept_count:04d}.jpg")
            kept_count += 1

    shutil.rmtree(raw_dir)
    return {
        "video": str(video_path),
        "segment_s": [start_s, end_s],
        "sample_fps": sample_fps,
        "output_resolution": [dst_w, dst_h],
        "crop_top_ratio": crop_top_ratio,
        "crop_bottom_ratio": crop_bottom_ratio,
        "extracted_count": len(raw_files),
        "kept_count": kept_count,
        "sharpness_median": round(float(np.median(vars_only)), 2),
        "sharpness_cutoff": round(cutoff, 2),
    }


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def analyze_colmap_baseline(sparse_txt_dir: Path):
    """Quantify real camera translation baseline vs. median scene depth from COLMAP sparse TXT.
    Crucial for predicting safe novel-view orbit angles before training/rendering 3DGS.
    """
    images_file = sparse_txt_dir / "images.txt"
    pts_file = sparse_txt_dir / "points3D.txt"
    if not images_file.exists():
        return {"status": "missing_sparse_txt"}

    lines = [l.strip() for l in images_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
    cam_centers = []
    cam_fwds = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        R = qvec2rotmat(qvec)
        cam_centers.append(-R.T @ tvec)
        cam_fwds.append(R[2])
    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    cam_fwds = np.asarray(cam_fwds, dtype=np.float64)
    if len(cam_centers) < 2:
        return {"status": "too_few_cameras"}

    # Decompose camera translation into along-look (dolly) vs lateral (orbit/parallax) baseline
    mean_fwd = np.mean(cam_fwds, axis=0)
    mean_fwd /= np.linalg.norm(mean_fwd) + 1e-9
    disp = cam_centers - cam_centers[0]
    along_vals = disp @ mean_fwd
    lateral_vecs = disp - np.outer(along_vals, mean_fwd)
    dolly_span = float(np.max(along_vals) - np.min(along_vals))
    lateral_span = float(np.max(np.linalg.norm(lateral_vecs, axis=1)))
    total_path_len = float(np.sum(np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)))

    # Median scene depth relative to average camera center
    pts = []
    if pts_file.exists():
        for line in pts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    if len(pts) >= 10:
        pts_arr = np.asarray(pts, dtype=np.float64)
        cam_mid = np.median(cam_centers, axis=0)
        med_depth = float(np.median(np.linalg.norm(pts_arr - cam_mid, axis=1)))
    else:
        med_depth = max(1.0, total_path_len * 2.0)

    lateral_ratio = lateral_span / max(1e-4, med_depth)
    dolly_ratio = dolly_span / max(1e-4, med_depth)
    est_parallax_deg = math.degrees(math.atan(lateral_ratio))

    if lateral_ratio >= 0.45:
        motion_class = "wide_orbit_or_lateral_walk"
        safe_novel_deg = "±25° ~ ±45° (wide lateral baseline captured)"
    elif lateral_ratio >= 0.16:
        motion_class = "moderate_parallax_walk"
        safe_novel_deg = "±10° ~ ±20° around captured trajectory"
    elif dolly_ratio >= 0.35:
        motion_class = "mostly_forward_dolly_narrow_lateral"
        safe_novel_deg = "±6° ~ ±12° lateral arc + smooth dolly/crane along path (avoid wide side-orbit on close foreground objects)"
    else:
        motion_class = "mostly_pure_rotation_low_baseline"
        safe_novel_deg = "±3° ~ ±6° micro-parallax only (for wide single-object orbit, use Image-to-3D mesh instead)"

    return {
        "registered_cameras": len(cam_centers),
        "median_scene_depth_units": round(med_depth, 3),
        "dolly_span_units": round(dolly_span, 3),
        "lateral_span_units": round(lateral_span, 3),
        "total_path_length_units": round(total_path_len, 3),
        "dolly_to_depth_ratio": round(dolly_ratio, 3),
        "lateral_to_depth_ratio": round(lateral_ratio, 3),
        "captured_lateral_parallax_deg": round(est_parallax_deg, 1),
        "camera_motion_class": motion_class,
        "recommended_safe_novel_view_budget": safe_novel_deg,
    }


def run_colmap_sfm(images_dir: Path, colmap_dir: Path, overlap: int = 18):
    """Run COLMAP 4.x sequential SfM + image_undistorter + TXT export."""
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = colmap_dir / "undistorted"

    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.max_image_size", "1600",
        "--SiftExtraction.max_num_features", "8192",
    ], log_path=colmap_dir / "01_feature_extractor.log")

    run_cmd([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", "0",
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.quadratic_overlap", "1",
    ], log_path=colmap_dir / "02_sequential_matcher.log")

    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ], log_path=colmap_dir / "03_mapper.log")

    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not models:
        raise RuntimeError("COLMAP mapper produced no sparse model (insufficient rigid structure or parallax).")

    best_model = None
    best_images = -1
    model_stats = {}
    for m in models:
        out_txt = subprocess.check_output(["colmap", "model_analyzer", "--path", str(m)], stderr=subprocess.STDOUT, text=True)
        n_reg, n_pts, reproj = 0, 0, None
        for line in out_txt.splitlines():
            if "Registered images:" in line:
                n_reg = int(line.split("Registered images:")[1].strip())
            elif "Points:" in line:
                n_pts = int(line.split("Points:")[1].strip())
            elif "Mean reprojection error:" in line:
                reproj = float(line.split("Mean reprojection error:")[1].strip().replace("px", ""))
        if n_reg > best_images:
            best_images = n_reg
            best_model = m
            model_stats = {
                "best_model_dir": str(m),
                "registered_images": n_reg,
                "sparse_points_3d": n_pts,
                "mean_reprojection_error_px": reproj,
            }

    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(best_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], log_path=colmap_dir / "04_undistorter.log")

    txt_dir = dense_dir / "sparse_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "model_converter",
        "--input_path", str(dense_dir / "sparse"),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ], log_path=colmap_dir / "05_converter_txt.log")

    total_imgs = len(list(images_dir.glob("*.jpg")))
    model_stats["total_input_images"] = total_imgs
    model_stats["registration_rate"] = round(best_images / max(1, total_imgs), 4)
    model_stats["baseline_analysis"] = analyze_colmap_baseline(txt_dir)
    return dense_dir, model_stats


def evaluate_brush_psnr(eval_dir: Path, gt_dir: Path):
    if not eval_dir.exists():
        return {}
    psnrs = []
    for p in sorted(eval_dir.glob("*.png")):
        gt_p = gt_dir / (p.stem + ".jpg")
        if not gt_p.exists():
            cands = [c for c in gt_dir.glob("*.jpg") if c.stem in p.stem]
            gt_p = cands[0] if cands else None
        if gt_p and gt_p.exists():
            a = cv.imread(str(p)).astype(np.float64)
            b = cv.imread(str(gt_p)).astype(np.float64)
            if a.shape != b.shape:
                b = cv.resize(b, (a.shape[1], a.shape[0]), interpolation=cv.INTER_AREA)
            mse = float(np.mean((a - b) ** 2))
            psnrs.append((p.name, round(20.0 * math.log10(255.0 / math.sqrt(max(1e-9, mse))), 2)))
    if not psnrs:
        return {}
    vals = [v for _, v in psnrs]
    return {
        "eval_views_count": len(psnrs),
        "mean_psnr_db": round(float(np.mean(vals)), 2),
        "max_psnr_db": round(float(np.max(vals)), 2),
        "per_view_psnr_db": psnrs,
    }


def run_brush_3dgs(colmap_undistorted: Path, splat_dir: Path, brush_bin: Path,
                   total_steps: int = 20000, max_res: int = 1600, max_splats: int = 500000,
                   eval_split_every: int = 12):
    """Train 3DGS with Brush and compute held-out PSNR."""
    if splat_dir.exists():
        shutil.rmtree(splat_dir)
    splat_dir.mkdir(parents=True, exist_ok=True)

    export_every = max(2500, total_steps // 4)
    dt = run_cmd([
        str(brush_bin),
        str(colmap_undistorted),
        "--total-steps", str(total_steps),
        "--max-resolution", str(max_res),
        "--max-splats", str(max_splats),
        "--growth-stop-iter", str(int(total_steps * 0.65)),
        "--eval-split-every", str(eval_split_every),
        "--eval-every", str(export_every),
        "--eval-save-to-disk",
        "--export-every", str(export_every),
        "--export-path", str(splat_dir),
        "--export-name", "scene_{iter}.ply",
    ], log_path=splat_dir / "brush_train.log")

    plys = sorted(splat_dir.glob("scene_*.ply"))
    final_ply = plys[-1] if plys else None
    if final_ply is not None:
        shutil.copy2(final_ply, splat_dir / "scene_final.ply")

    eval_dirs = sorted([d for d in splat_dir.glob("eval_*") if d.is_dir()], key=lambda d: int(d.name.split("_")[1]))
    psnr_stats = evaluate_brush_psnr(eval_dirs[-1], colmap_undistorted / "images") if eval_dirs else {}
    return {
        "train_time_s": round(dt, 1),
        "total_steps": total_steps,
        "exported_plys": [p.name for p in plys],
        "final_ply": str(splat_dir / "scene_final.ply") if final_ply else None,
        "final_ply_size_mb": round(final_ply.stat().st_size / (1024 * 1024), 2) if final_ply else 0.0,
        "heldout_eval": psnr_stats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=None, help="Source video path (required for frames/all)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output project directory")
    ap.add_argument("--start", type=float, default=0.0, help="Rigid segment start time (s)")
    ap.add_argument("--end", type=float, default=5.0, help="Rigid segment end time (s)")
    ap.add_argument("--fps", type=float, default=18.0, help="Sampling FPS for COLMAP")
    ap.add_argument("--max-side", type=int, default=1600, help="Max image side for training")
    ap.add_argument("--crop-top", type=float, default=0.0, help="Top margin crop ratio (e.g. static HUD)")
    ap.add_argument("--crop-bottom", type=float, default=0.0, help="Bottom margin crop ratio (e.g. 0.068 for watermark)")
    ap.add_argument("--overlap", type=int, default=18, help="COLMAP sequential matcher frame overlap")
    ap.add_argument("--steps", type=int, default=20000, help="Brush 3DGS training steps")
    ap.add_argument("--max-splats", type=int, default=500000, help="Upper bound on 3D Gaussians")
    ap.add_argument("--brush", help="Brush executable (default: H2O_BRUSH_BIN, then brush_app or brush on PATH)")
    ap.add_argument("--stage", choices=["all", "frames", "colmap", "baseline", "brush"], default="all")
    args = ap.parse_args()

    if args.stage in ("all", "frames"):
        if args.video is None:
            ap.error("--video is required for --stage frames or all")
        for name in ("ffmpeg", "ffprobe"):
            if not shutil.which(name):
                ap.error(f"{name} was not found on PATH")
    if args.stage in ("all", "colmap") and not shutil.which("colmap"):
        ap.error("COLMAP was not found on PATH")
    brush_bin = None
    if args.stage in ("all", "brush"):
        configured = args.brush or os.environ.get("H2O_BRUSH_BIN")
        if configured:
            candidate = Path(configured).expanduser()
            executable = shutil.which(str(candidate.absolute() if candidate.is_file() else candidate))
        else:
            executable = shutil.which("brush_app") or shutil.which("brush")
        if not executable:
            ap.error("Brush executable not found. Set --brush, H2O_BRUSH_BIN, or add brush_app to PATH.")
        brush_bin = Path(executable).absolute()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out_dir / "frames_clean"
    colmap_dir = args.out_dir / "colmap"
    splat_dir = args.out_dir / "splat"
    report_path = args.out_dir / "gs3d_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    if args.stage in ("all", "frames"):
        report["frames"] = extract_and_filter_frames(
            args.video, images_dir, args.start, args.end, args.fps, args.max_side,
            crop_bottom_ratio=args.crop_bottom, crop_top_ratio=args.crop_top
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.stage in ("all", "colmap"):
        dense_dir, colmap_stats = run_colmap_sfm(images_dir, colmap_dir, overlap=args.overlap)
        report["colmap"] = colmap_stats
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        dense_dir = colmap_dir / "undistorted"

    if args.stage == "baseline":
        report["baseline_check"] = analyze_colmap_baseline(dense_dir / "sparse_txt")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.stage in ("all", "brush"):
        report["brush"] = run_brush_3dgs(
            dense_dir, splat_dir, brush_bin,
            total_steps=args.steps, max_res=args.max_side, max_splats=args.max_splats
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
