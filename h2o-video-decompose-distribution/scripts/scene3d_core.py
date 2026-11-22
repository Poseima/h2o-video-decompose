"""General-purpose 3D & Blender scene inverse decomposition and quantitative estimators."""

import json
import math
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter

from measure_core import flow, pick, sample, solve_pair


def classify_scene_paradigm(frames_bgr, cam_info, mat_info, light_info, optics_info):
    """Automatically discriminate whether a shot is:
    - '2D_PLANAR_UI_MOTION': 2D/2.5D flat UI, white-page typing, slides, screen recording, planar card motion
    - '3D_BLENDER_SCENE': 3D modeled/rendered environment, product, hard-surface, voxel/toon 3D, volumetric scene
    - 'HYBRID_2D_OVER_3D': 3D scene with screen-locked or planar 2D typography/HUD overlays
    """
    bgr = frames_bgr[len(frames_bgr) // 2]
    h, w = bgr.shape[:2]
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # 1. Check flat-color dominance (typical of 2D UI / white-page typography / slides)
    # Quantize to 16 gray bins and check how much area the top 2 bins cover
    hist = cv.calcHist([gray], [0], None, [16], [0, 256]).ravel() / float(h * w)
    top2_bin_coverage = float(np.sum(np.sort(hist)[-2:]))
    flat_grad_ratio = mat_info["flat_gradient_ratio"]
    cont_shading_ratio = mat_info["continuous_shading_ratio"]

    # 2. Check single-plane 2D homography fit residual across frames
    single_plane_p95 = None
    inlier_ratio = 1.0
    if len(frames_bgr) >= 2:
        g0 = cv.cvtColor(frames_bgr[0], cv.COLOR_BGR2GRAY)
        g1 = cv.cvtColor(frames_bgr[-1], cv.COLOR_BGR2GRAY)
        mask_all = np.full((h, w), 255, dtype=np.uint8)
        a_pts, b_pts = flow(g0, g1, pick(g0, mask_all, spacing=16))
        if len(a_pts) >= 10:
            fit = solve_pair(a_pts, b_pts, strict=False)
            if fit is not None:
                single_plane_p95 = fit.get("holdout_p95_px")
                inlier_ratio = float(fit["inlier_observations"]) / float(len(a_pts))

    # 3. Score 3D evidence vs 2D planar evidence
    has_vp = any(v is not None for v in cam_info["vanishing_points_px"].values())
    depth_layers = optics_info["motion_and_parallax_3d"].get("estimated_depth_parallax_layers", 1)
    radial_prof = light_info["radial_halo_profile"]
    has_volumetric_or_bloom = bool(
        radial_prof.get("bloom_detected", False) or radial_prof.get("volume_fog_detected", False)
    )

    score_2d = 0.0
    score_3d = 0.0
    reasons = []

    if top2_bin_coverage > 0.68 and flat_grad_ratio > 0.58:
        score_2d += 2.5
        reasons.append(f"High flat background coverage ({top2_bin_coverage:.2f}) typical of 2D page/UI")
    if single_plane_p95 is not None and single_plane_p95 <= 2.75 and inlier_ratio >= 0.80:
        score_2d += 2.0
        reasons.append(f"Motion well-explained by single 2D homography (P95={single_plane_p95:.2f}px)")

    if cont_shading_ratio >= 0.24 or top2_bin_coverage < 0.55:
        score_3d += 2.0
        reasons.append(f"Continuous 3D surface shading gradients ({cont_shading_ratio:.2f})")
    if has_vp and (cam_info["line_clusters"]["cluster_a_count"] + cam_info["line_clusters"]["cluster_b_count"] >= 8):
        score_3d += 1.8
        reasons.append("Converging 3D perspective line clusters & vanishing points detected")
    if depth_layers >= 2 or (single_plane_p95 is not None and single_plane_p95 > 3.5):
        score_3d += 1.8
        reasons.append(f"Multi-depth motion parallax detected ({depth_layers} depth layers)")
    if has_volumetric_or_bloom:
        score_3d += 1.2
        reasons.append("3D emissive/volumetric light scattering halo profile detected")

    if score_2d >= score_3d + 1.2:
        paradigm = "2D_PLANAR_UI_MOTION"
        recommended_pipeline = "Run `decompose.py measure` (2D/2.5D planar homography H_f + ESF 10-90% edge width + font IoU)"
    elif score_3d >= 2.5 and score_2d >= 2.0:
        paradigm = "HYBRID_2D_OVER_3D"
        recommended_pipeline = "Hybrid: Run `decompose.py scene3d` for 3D Blender scene/camera/lighting + `decompose.py measure` for 2D typography/HUD layers"
    else:
        paradigm = "3D_BLENDER_SCENE"
        recommended_pipeline = "Run `decompose.py scene3d` (6-pillar 3D camera, topology, BSDF, lighting/volumetrics, 6-DoF motion & compositor optics)"

    return {
        "detected_paradigm": paradigm,
        "score_2d_planar": round(score_2d, 2),
        "score_3d_scene": round(score_3d, 2),
        "top2_histogram_bin_coverage": round(top2_bin_coverage, 4),
        "single_plane_homography_p95_px": round(single_plane_p95, 3) if single_plane_p95 is not None else None,
        "single_plane_inlier_ratio": round(inlier_ratio, 3),
        "evidence": reasons,
        "recommended_pipeline": recommended_pipeline,
    }


def srgb_to_linear(srgb_0_255):
    """Convert 0..255 sRGB array to 0..1 linear light."""
    c = np.clip(np.asarray(srgb_0_255, dtype=np.float32) / 255.0, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_hex(rgb_0_255):
    r, g, b = [int(np.clip(round(v), 0, 255)) for v in rgb_0_255[:3]]
    return f"#{r:02X}{g:02X}{b:02X}"


def solve_homography_4pt(pa, pb):
    """Solve 8-parameter perspective transform mapping pa -> pb."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A = np.asarray(matrix, dtype=np.float64)
    B = np.asarray(pb, dtype=np.float64).reshape(8)
    coeffs = np.linalg.solve(A, B)
    H = np.append(coeffs, 1.0).reshape(3, 3)
    return coeffs, H


def _line_intersection(l1, l2):
    """Intersect two segment tuples (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = [float(v) for v in l1]
    x3, y3, x4, y4 = [float(v) for v in l2]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float64)


def analyze_camera_and_projection(bgr, corners_xy=None, focal_mm_override=None, ref_depth_m=9.0):
    """Pillar 1: Estimate projection type (PERSP vs ORTHO), vanishing points, focal length (36mm eq),
    camera tilt/roll, and (if 4 planar corners are supplied) exact 3D camera-space coordinates.
    """
    h, w = bgr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    diag = math.hypot(w, h)
    max_side = float(max(w, h))

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 1.0), 45, 140)
    raw_lines = cv.HoughLinesP(
        edges, 1.0, np.pi / 180.0, threshold=45,
        minLineLength=max(24, int(min(w, h) * 0.045)),
        maxLineGap=max(8, int(min(w, h) * 0.015))
    )

    vertical_lines = []
    horiz_cluster_a = []
    horiz_cluster_b = []
    all_segments = []

    if raw_lines is not None:
        for item in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in item]
            length = math.hypot(x2 - x1, y2 - y1)
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if ang > 90.0:
                ang -= 180.0
            elif ang <= -90.0:
                ang += 180.0
            all_segments.append((x1, y1, x2, y2, ang, length))
            if abs(abs(ang) - 90.0) <= 18.0:
                vertical_lines.append((x1, y1, x2, y2, ang, length))
            elif 4.0 <= ang < 72.0:
                horiz_cluster_a.append((x1, y1, x2, y2, ang, length))
            elif -72.0 < ang <= -4.0:
                horiz_cluster_b.append((x1, y1, x2, y2, ang, length))

    def estimate_vp(cluster):
        if len(cluster) < 3:
            return None, 0.0, 0
        top = sorted(cluster, key=lambda s: s[5], reverse=True)[:32]
        pts = []
        weights = []
        angles = [s[4] for s in top]
        angle_spread = float(np.percentile(angles, 85) - np.percentile(angles, 15)) if len(angles) >= 3 else 0.0
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                if abs(top[i][4] - top[j][4]) < 1.2:
                    continue
                pt = _line_intersection(top[i][:4], top[j][:4])
                if pt is not None and np.all(np.isfinite(pt)) and np.linalg.norm(pt - np.array([cx, cy])) < 25.0 * diag:
                    pts.append(pt)
                    weights.append(top[i][5] * top[j][5])
        if not pts:
            return None, angle_spread, len(top)
        pts_arr = np.asarray(pts)
        med = np.median(pts_arr, axis=0)
        dists = np.linalg.norm(pts_arr - med, axis=1)
        inliers = dists <= max(45.0, np.percentile(dists, 65))
        vp = np.average(pts_arr[inliers], axis=0, weights=np.asarray(weights)[inliers])
        return vp.tolist(), angle_spread, int(np.sum(inliers))

    vp_vert, spread_v, n_v = estimate_vp(vertical_lines)
    vp_a, spread_a, n_a = estimate_vp(horiz_cluster_a)
    vp_b, spread_b, n_b = estimate_vp(horiz_cluster_b)

    # Projection mode classification (PERSP vs ORTHO)
    max_spread = max(spread_a, spread_b, spread_v)
    vp_dists = [
        float(np.linalg.norm(np.asarray(vp) - np.array([cx, cy]))) / diag
        for vp in (vp_vert, vp_a, vp_b) if vp is not None
    ]
    min_vp_dist_diag = min(vp_dists) if vp_dists else 99.0
    is_ortho = (max_spread < 1.1 and min_vp_dist_diag > 12.0)
    projection_type = "ORTHO" if is_ortho else "PERSP"

    # Estimate focal length from orthogonal vanishing points when available
    est_focal_px = None
    vp_pairs = [(vp_a, vp_b), (vp_a, vp_vert), (vp_b, vp_vert)]
    for v1, v2 in vp_pairs:
        if v1 is not None and v2 is not None:
            u1 = np.asarray(v1) - np.array([cx, cy])
            u2 = np.asarray(v2) - np.array([cx, cy])
            dot = float(np.dot(u1, u2))
            if dot < -0.04 * (max_side ** 2):
                f_cand = math.sqrt(-dot)
                if 0.35 * max_side <= f_cand <= 4.5 * max_side:
                    est_focal_px = f_cand
                    break

    if focal_mm_override is not None:
        focal_mm = float(focal_mm_override)
        focal_px = focal_mm / 36.0 * max_side
        focal_source = "user_override"
    elif est_focal_px is not None:
        focal_px = est_focal_px
        focal_mm = round(36.0 * focal_px / max_side, 2)
        focal_source = "orthogonal_vanishing_points"
    else:
        focal_mm = 35.0
        focal_px = (35.0 / 36.0) * max_side
        focal_source = "default_35mm_prior"

    # Estimate camera roll & pitch from vertical vanishing point or dominant vertical lines
    if vertical_lines:
        v_angles = np.asarray([s[4] for s in vertical_lines])
        v_weights = np.asarray([s[5] for s in vertical_lines])
        # Deviations from +/- 90 deg
        devs = np.where(v_angles >= 0, v_angles - 90.0, v_angles + 90.0)
        camera_roll_deg = round(float(np.average(devs, weights=v_weights)), 2)
    else:
        camera_roll_deg = 0.0

    if vp_vert is not None:
        dy_vp = vp_vert[1] - cy
        pitch_rad = math.atan2(focal_px, abs(dy_vp)) * (1.0 if dy_vp > 0 else -1.0)
        camera_pitch_deg = round(math.degrees(pitch_rad), 2)
    else:
        camera_pitch_deg = 0.0

    # Framing Mask (cinematic letterbox / pillarbox bars) & Lens Geometric Curvature Hint
    row_max = np.max(gray, axis=1)
    row_std = np.std(gray.astype(np.float32), axis=1)
    col_max = np.max(gray, axis=0)
    col_std = np.std(gray.astype(np.float32), axis=0)
    top_bar = 0
    while top_bar < h // 3 and row_max[top_bar] <= 10 and row_std[top_bar] <= 2.0:
        top_bar += 1
    bot_bar = 0
    while bot_bar < h // 3 and row_max[h - 1 - bot_bar] <= 10 and row_std[h - 1 - bot_bar] <= 2.0:
        bot_bar += 1
    left_bar = 0
    while left_bar < w // 3 and col_max[left_bar] <= 10 and col_std[left_bar] <= 2.0:
        left_bar += 1
    right_bar = 0
    while right_bar < w // 3 and col_max[w - 1 - right_bar] <= 10 and col_std[w - 1 - right_bar] <= 2.0:
        right_bar += 1

    # Require symmetric or clearly bounded matte bars (>= 4% frame side) and non-black inner content
    inner_w = max(1, w - left_bar - right_bar)
    inner_h = max(1, h - top_bar - bot_bar)
    inner_mean = float(np.mean(gray[top_bar:h - bot_bar if bot_bar > 0 else h, left_bar:w - right_bar if right_bar > 0 else w]))
    top_f, bot_f = top_bar / float(h), bot_bar / float(h)
    left_f, right_f = left_bar / float(w), right_bar / float(w)
    letterbox_detected = bool(
        inner_mean >= 14.0
        and (
            (top_f >= 0.04 and bot_f >= 0.04 and abs(top_f - bot_f) <= 0.05)
            or (left_f >= 0.04 and right_f >= 0.04 and abs(left_f - right_f) <= 0.05)
        )
    )
    active_aspect = round(inner_w / float(inner_h), 3) if letterbox_detected else round(w / float(h), 3)

    # Lens geometric curvature hint from long boundary edge contours (barrel vs pincushion vs rectilinear)
    border_edges = cv.Canny(gray, 60, 160)
    contours_lens, _ = cv.findContours(border_edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    signed_bows = []
    min_arc_len = 0.22 * min(w, h)
    for cnt in contours_lens:
        if len(cnt) < 40:
            continue
        pts_c = cnt[:, 0, :].astype(np.float64)
        p0, p1 = pts_c[0], pts_c[-1]
        chord = p1 - p0
        chord_len = float(np.linalg.norm(chord))
        if chord_len < min_arc_len:
            continue
        # Check straightness ratio (exclude closed loops or sharp right-angle corners)
        arc_len = float(cv.arcLength(cnt, False))
        if arc_len > chord_len * 1.12:
            continue
        mid = 0.5 * (p0 + p1)
        # Only use contours near outer frame margins (radial distance > 0.35 of min(w,h)/2)
        rad_mid = math.hypot(mid[0] - cx, mid[1] - cy)
        if rad_mid < 0.32 * min(w, h):
            continue
        # Compute signed radial bow relative to frame center (+ = bows outward = barrel; - = pincushion)
        u_c = chord / max(1e-6, chord_len)
        n_c = np.array([-u_c[1], u_c[0]])
        outward = np.array([mid[0] - cx, mid[1] - cy])
        if float(np.dot(n_c, outward)) < 0:
            n_c = -n_c
        rel = pts_c - p0
        proj = rel @ u_c
        perp = rel @ n_c
        mid_band = (proj >= 0.35 * chord_len) & (proj <= 0.65 * chord_len)
        if np.any(mid_band) and float(np.max(np.abs(perp))) <= 0.08 * chord_len:
            signed_bows.append(float(np.median(perp[mid_band])))
    if len(signed_bows) >= 4:
        med_bow = round(float(np.median(signed_bows)), 2)
        if med_bow >= 3.2:
            lens_dist_hint = "BARREL_WIDE"
        elif med_bow <= -3.2:
            lens_dist_hint = "PINCUSHION_TELE"
        else:
            lens_dist_hint = "RECTILINEAR_NONE"
    else:
        med_bow = 0.0
        lens_dist_hint = "RECTILINEAR_NONE"

    result = {
        "projection_type": projection_type,
        "sensor_width_mm": 36.0,
        "estimated_focal_mm": focal_mm,
        "focal_px": round(float(focal_px), 2),
        "focal_estimation_method": focal_source,
        "camera_pitch_deg": camera_pitch_deg,
        "camera_roll_deg": camera_roll_deg,
        "framing_and_lens_geometry": {
            "letterbox_detected": letterbox_detected,
            "letterbox_bars_frac": {
                "top": round(top_f, 4),
                "bottom": round(bot_f, 4),
                "left": round(left_f, 4),
                "right": round(right_f, 4),
            },
            "active_content_aspect_ratio": active_aspect,
            "lens_distortion_hint": lens_dist_hint,
            "edge_line_curvature_px": med_bow,
        },
        "vanishing_points_px": {
            "vertical_vp": [round(v, 2) for v in vp_vert] if vp_vert else None,
            "horizon_vp_a": [round(v, 2) for v in vp_a] if vp_a else None,
            "horizon_vp_b": [round(v, 2) for v in vp_b] if vp_b else None,
        },
        "line_clusters": {
            "vertical_count": len(vertical_lines),
            "cluster_a_count": len(horiz_cluster_a),
            "cluster_b_count": len(horiz_cluster_b),
            "sample_segments": [
                {"x1": round(s[0], 1), "y1": round(s[1], 1), "x2": round(s[2], 1), "y2": round(s[3], 1), "angle_deg": round(s[4], 1)}
                for s in sorted(all_segments, key=lambda x: x[5], reverse=True)[:18]
            ],
        },
    }

    # If a 4-corner planar 3D object/surface is specified: solve exact 3D corners and foreshortening depth ratio
    if corners_xy is not None and len(corners_xy) == 4:
        pts = np.asarray(corners_xy, dtype=np.float64)  # TL, TR, BR, BL
        p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]
        h_left = float(np.linalg.norm(p_bl - p_tl))
        h_right = float(np.linalg.norm(p_br - p_tr))
        w_top = float(np.linalg.norm(p_tr - p_tl))
        w_bot = float(np.linalg.norm(p_br - p_bl))

        # Foreshortening depth ratio: D_L / D_R = h_right / h_left
        depth_ratio_lr = h_right / max(1e-5, h_left)
        d_left = float(ref_depth_m) * (2.0 * depth_ratio_lr / (1.0 + depth_ratio_lr))
        d_right = float(ref_depth_m) * (2.0 / (1.0 + depth_ratio_lr))

        fov_scale = 36.0 / max(1e-3, focal_mm)

        def px_to_cam3d(px_pt, depth_m):
            x3 = ((px_pt[0] / float(w)) - 0.5) * depth_m * fov_scale
            y3 = (0.5 - (px_pt[1] / float(h))) * depth_m * fov_scale * (float(h) / float(w))
            z3 = -depth_m
            return np.array([x3, y3, z3], dtype=np.float64)

        P_TL = px_to_cam3d(p_tl, d_left)
        P_BL = px_to_cam3d(p_bl, d_left)
        P_TR = px_to_cam3d(p_tr, d_right)
        P_BR = px_to_cam3d(p_br, d_right)

        center_3d = 0.25 * (P_TL + P_TR + P_BR + P_BL)
        vec_right = 0.5 * ((P_TR - P_TL) + (P_BR - P_BL))
        vec_up = 0.5 * ((P_TL - P_BL) + (P_TR - P_BR))
        normal_3d = np.cross(vec_right, vec_up)
        n_len = float(np.linalg.norm(normal_3d))
        if n_len > 1e-7:
            normal_3d = normal_3d / n_len
        if normal_3d[2] < 0:
            normal_3d = -normal_3d

        width_3d = float(np.linalg.norm(vec_right))
        height_3d = float(np.linalg.norm(vec_up))
        focus_distance_m = float(np.linalg.norm(center_3d))

        result["planar_object_3d"] = {
            "corners_2d_px": {"TL": p_tl.tolist(), "TR": p_tr.tolist(), "BR": p_br.tolist(), "BL": p_bl.tolist()},
            "projected_heights_px": {"left_px": round(h_left, 2), "right_px": round(h_right, 2)},
            "projected_widths_px": {"top_px": round(w_top, 2), "bottom_px": round(w_bot, 2)},
            "foreshortening_depth_ratio_L_over_R": round(depth_ratio_lr, 5),
            "depth_left_m": round(d_left, 4),
            "depth_right_m": round(d_right, 4),
            "explicit_focus_distance_m": round(focus_distance_m, 4),
            "dimensions_3d_m": {
                "width_m": round(width_3d, 4),
                "height_m": round(height_3d, 4),
                "aspect_ratio": round(width_3d / max(1e-5, height_3d), 4),
            },
            "corners_camera_space_3d_m": {
                "P_TL": [round(v, 4) for v in P_TL.tolist()],
                "P_TR": [round(v, 4) for v in P_TR.tolist()],
                "P_BR": [round(v, 4) for v in P_BR.tolist()],
                "P_BL": [round(v, 4) for v in P_BL.tolist()],
                "center_3d": [round(v, 4) for v in center_3d.tolist()],
                "surface_normal_3d": [round(v, 4) for v in normal_3d.tolist()],
            },
        }

    return result


def rectify_and_analyze_surface(bgr, corners_xy, out_size=(860, 280), symmetric_complete=False):
    """Pillar 2A: Perspective-correct orthographic unwarp of a 3D planar surface,
    plus voxel/pixel-art quantization detection and modular array periodicity analysis.
    """
    sh, sw = bgr.shape[:2]
    W, H = int(out_size[0]), int(out_size[1])
    pts = np.asarray(corners_xy, dtype=np.float64)
    p_tl, p_tr, p_br, p_bl = pts[0], pts[1], pts[2], pts[3]

    # Check if right or left edge extends slightly outside the frame (clamped region)
    # Build homography mapping destination [0..W, 0..H] -> source [p_tl, p_tr, p_br, p_bl]
    dst_pts = [(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))]
    src_pts = [tuple(p_tl), tuple(p_tr), tuple(p_br), tuple(p_bl)]
    _, H_dst_to_src = solve_homography_4pt(dst_pts, src_pts)

    # Generate pixel grid for cv.remap with perspective-correct coordinates
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ones = np.ones_like(xx)
    homog = np.stack([xx, yy, ones], axis=-1) @ H_dst_to_src.T
    map_x = (homog[:, :, 0] / np.maximum(1e-7, homog[:, :, 2])).astype(np.float32)
    map_y = (homog[:, :, 1] / np.maximum(1e-7, homog[:, :, 2])).astype(np.float32)

    valid_mask = (map_x >= 0) & (map_x <= sw - 1) & (map_y >= 0) & (map_y <= sh - 1)
    rectified = cv.remap(bgr, map_x, map_y, interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    # Optional symmetry completion when part of the planar asset lies outside the frame boundary
    symmetry_axis_x = W // 2
    if symmetric_complete and not np.all(valid_mask):
        arr = rectified.astype(np.float32)
        # Detect horizontal symmetry center from top border/star features or default to W//2
        top_strip = arr[int(0.05 * H):int(0.28 * H), :]
        g_exc = np.maximum(0.0, top_strip[:, :, 1] - 0.65 * top_strip[:, :, 2])
        if np.sum(g_exc) > 500.0:
            xs = np.arange(W, dtype=np.float32)
            symmetry_axis_x = int(round(float(np.sum(g_exc * xs[None, :]) / np.sum(g_exc))))
            symmetry_axis_x = int(np.clip(symmetry_axis_x, int(0.42 * W), int(0.58 * W)))

        # Replace invalid/out-of-frame columns with mirrored valid columns
        invalid_cols = np.where(~np.all(valid_mask, axis=0))[0]
        if len(invalid_cols) > 0:
            first_inv = max(12, int(invalid_cols[0]) - 10)
            for x in range(first_inv, W):
                mx = int(np.clip(2 * symmetry_axis_x - 1 - x, 0, W - 1))
                alpha = min(1.0, (x - first_inv) / 10.0)
                arr[:, x, :] = (1.0 - alpha) * arr[:, x, :] + alpha * arr[:, mx, :]
            rectified = np.clip(arr, 0, 255).astype(np.uint8)

    # Detect Voxel / Pixel-Art quantization step period & Modular Array repetition period via 1D/2D autocorrelation
    gray_rect = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(np.diff(gray_rect, axis=1))
    gy = np.abs(np.diff(gray_rect, axis=0))
    prof_x = np.mean(gx, axis=0)
    prof_y = np.mean(gy, axis=1)

    def detect_period(profile, min_lag=2, max_lag=120):
        sig = profile - np.mean(profile)
        norm = float(np.sum(sig ** 2))
        if norm < 1e-5:
            return None, 0.0
        ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:] / norm
        hi = min(max_lag, len(ac) - 1)
        if hi <= min_lag:
            return None, 0.0
        window = ac[min_lag:hi]
        best_idx = int(np.argmax(window)) + min_lag
        best_val = float(ac[best_idx])
        return best_idx, best_val

    pixel_step_x, pixel_score_x = detect_period(prof_x, min_lag=2, max_lag=12)
    array_step_x, array_score_x = detect_period(prof_x, min_lag=24, max_lag=max(26, W // 3))
    array_step_y, array_score_y = detect_period(prof_y, min_lag=20, max_lag=max(22, H // 2))

    # Measure fraction of near-zero adjacent diffs (pixel plateau ratio)
    plateau_ratio = float(np.mean(gx < 1.2))
    is_pixel_art = (pixel_score_x > 0.18 or plateau_ratio > 0.32)

    topology_info = {
        "rectified_size_px": [W, H],
        "in_frame_coverage_ratio": round(float(np.mean(valid_mask)), 4),
        "symmetry_axis_x_px": symmetry_axis_x,
        "texture_sampling_mode": "Closest" if is_pixel_art else "Linear",
        "voxel_pixel_step_px": pixel_step_x if is_pixel_art else 1,
        "voxel_autocorr_score": round(pixel_score_x, 4),
        "modular_array_periodicity": {
            "detected_x_repeat": bool(array_score_x > 0.14),
            "step_x_px": array_step_x if array_score_x > 0.14 else None,
            "estimated_columns": round(W / float(array_step_x), 1) if (array_step_x and array_score_x > 0.14) else None,
            "autocorr_x": round(array_score_x, 4),
            "detected_y_repeat": bool(array_score_y > 0.14),
            "step_y_px": array_step_y if array_score_y > 0.14 else None,
            "autocorr_y": round(array_score_y, 4),
        },
    }
    return rectified, topology_info


def decompose_surface_layers_and_clean_plate(rectified_bgr, full_bgr, corners_xy, halo_pad_px=16):
    """Pillar 2B: Multi-scale Difference-of-Gaussians (DoG) ridge extraction + chromatic dominance
    to separate Foreground 3D relief elements, Midground structure/frame elements, and generate a
    halo-expanded clean background plate with zero ghost border.
    """
    H, W = rectified_bgr.shape[:2]
    # Convert BGR -> RGB float
    rgb = cv.cvtColor(rectified_bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    blur_R = cv.GaussianBlur(R, (0, 0), 6.5)
    blur_G = cv.GaussianBlur(G, (0, 0), 6.5)
    blur_B = cv.GaussianBlur(B, (0, 0), 6.5)

    ridge_R = R - 0.72 * blur_R
    ridge_G = G - 0.72 * blur_G
    ridge_B = B - 0.72 * blur_B

    # Chromatic dominance & multi-scale ridge masks (preserves hollow tube interiors!)
    primary_dom = np.maximum(R - 1.15 * G, B - 1.15 * G)
    fg_mask = np.zeros((H, W), dtype=np.float32)
    inner_y0, inner_y1 = int(0.18 * H), int(0.90 * H)
    inner_x0, inner_x1 = int(0.03 * W), int(0.97 * W)
    fg_raw = np.clip((primary_dom - 34.0) / 36.0, 0.0, 1.0) * np.clip(( np.maximum(ridge_R, ridge_B) - 28.0) / 38.0, 0.0, 1.0)
    fg_mask[inner_y0:inner_y1, inner_x0:inner_x1] = fg_raw[inner_y0:inner_y1, inner_x0:inner_x1]

    lum_ridge = 0.45 * ridge_R + 0.45 * ridge_G + 0.10 * ridge_B
    all_ridges = np.clip((lum_ridge - 22.0) / 35.0, 0.0, 1.0)
    sec_dom = np.clip((G - 0.72 * R - 16.0) / 28.0, 0.0, 1.0) * np.clip((ridge_G - 14.0) / 25.0, 0.0, 1.0)
    all_ridges = np.maximum(all_ridges, sec_dom)
    mg_mask = np.clip(all_ridges * (1.0 - 0.85 * fg_mask), 0.0, 1.0)
    combined_height = np.clip(np.maximum(fg_mask, 0.65 * mg_mask), 0.0, 1.0)

    # Halo-expanded background clean plate (prevents leftover outer border ghosting when shifting 3D object/camera)
    sh, sw = full_bgr.shape[:2]
    pts = np.asarray(corners_xy, dtype=np.float32)
    center_2d = np.mean(pts, axis=0)
    expanded_pts = []
    for p in pts:
        v = p - center_2d
        dist = max(1e-4, float(np.linalg.norm(v)))
        expanded_pts.append(p + (v / dist) * float(halo_pad_px * 1.45))
    expanded_poly = np.rint(np.asarray(expanded_pts)).astype(np.int32)

    mask_erase = np.zeros((sh, sw), dtype=np.uint8)
    cv.fillConvexPoly(mask_erase, expanded_poly, 255)

    # Directional perspective-aligned clean-plate fill + Telea inpaint refinement
    bg_clean = full_bgr.copy().astype(np.float32)
    ys_mask, xs_mask = np.where(mask_erase > 0)
    if len(xs_mask) > 0:
        x_min, x_max = int(np.min(xs_mask)), int(np.max(xs_mask))
        for x in range(x_min, x_max + 1):
            col_ys = ys_mask[xs_mask == x]
            if not len(col_ys):
                continue
            y_top, y_bot = int(np.min(col_ys)), int(np.max(col_ys))
            span = max(1, y_bot - y_top)
            for y in range(y_top, y_bot + 1):
                u = (y - y_top) / float(span)
                xb = int(np.clip(x - (1.0 - u) * 68, 0, sw - 1))
                yb = int(np.clip(y_bot + 14 + (y - y_top) * 0.58, 0, sh - 1))
                bg_clean[y, x] = full_bgr[yb, xb]

        blurred = cv.GaussianBlur(bg_clean, (9, 9), 3.5)
        alpha_mask = cv.GaussianBlur(mask_erase.astype(np.float32) / 255.0, (15, 15), 5.0)[:, :, None]
        bg_clean = (1.0 - alpha_mask) * full_bgr.astype(np.float32) + alpha_mask * blurred

    return (
        (fg_mask * 255).astype(np.uint8),
        (mg_mask * 255).astype(np.uint8),
        (combined_height * 255).astype(np.uint8),
        np.clip(bg_clean, 0, 255).astype(np.uint8),
    )


def analyze_material_and_shading(bgr):
    """Pillar 3: Classify shading paradigm (PBR vs NPR/Toon vs Emissive), estimate Principled BSDF
    parameters, measure multi-scale surface texture/grunge spectrum (fine/mid/macro), and detect
    stylized dark ink outlines (Line Art / Freestyle / Inverted Hull).
    """
    h, w = bgr.shape[:2]
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    lin = srgb_to_linear(rgb)
    lum_lin = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0

    # Gradient distribution inside non-edge smooth regions to distinguish PBR continuous vs NPR Toon bands
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    gray_n = gray / 255.0
    sm = cv.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv.Sobel(sm, cv.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv.Sobel(sm, cv.CV_32F, 0, 1, ksize=3) / 8.0
    gmag = np.hypot(gx, gy)

    flat_ratio = float(np.mean(gmag < 1.1))
    mid_grad_ratio = float(np.mean((gmag >= 1.1) & (gmag <= 8.5)))
    sharp_edge_ratio = float(np.mean(gmag > 14.0))
    emissive_pixel_ratio = float(np.mean((val > 0.78) & (sat > 0.45)))

    if flat_ratio > 0.62 and mid_grad_ratio < 0.22 and sharp_edge_ratio > 0.04:
        shading_paradigm = "NPR_TOON_SHADER_TO_RGB"
    elif emissive_pixel_ratio > 0.035 and float(np.percentile(lum_lin, 25)) < 0.05:
        shading_paradigm = "PBR_WITH_STRONG_EMISSIVE_NEON"
    else:
        shading_paradigm = "PBR_PRINCIPLED_BSDF"

    # Estimate surface Roughness from specular highlight peak sharpness and spatial variance in dark/midtone surfaces
    dark_mid_mask = (val >= 0.08) & (val <= 0.65)
    if np.any(dark_mid_mask):
        local_contrast = cv.GaussianBlur(lum_lin ** 2, (15, 15), 4.0) - cv.GaussianBlur(lum_lin, (15, 15), 4.0) ** 2
        local_std = np.sqrt(np.maximum(0.0, local_contrast))[dark_mid_mask]
        cv_spec = float(np.percentile(local_std, 90) / max(1e-4, np.median(lum_lin[dark_mid_mask])))
        roughness_median = float(np.clip(0.58 - 0.18 * math.log1p(cv_spec), 0.08, 0.88))
        roughness_min = round(max(0.05, roughness_median - 0.22), 2)
        roughness_max = round(min(0.95, roughness_median + 0.20), 2)
        has_roughness_noise_map = bool(cv_spec > 0.65)
    else:
        roughness_median = 0.45
        roughness_min, roughness_max = 0.25, 0.65
        has_roughness_noise_map = False

    # Estimate Metallic tendency: compare saturation of top 5% non-clipped highlights vs midtones
    hi_mask = (val >= 0.72) & (val <= 0.96)
    mid_mask = (val >= 0.25) & (val < 0.65)
    sat_hi = float(np.median(sat[hi_mask])) if np.any(hi_mask) else 0.2
    sat_mid = float(np.median(sat[mid_mask])) if np.any(mid_mask) else 0.3
    metallic_ratio = sat_hi / max(0.08, sat_mid)

    # 1. Universal Multi-Scale Surface Texture & Weathering/Grunge Spectrum (on interior non-edge surfaces)
    strong_edges = (cv.Canny(gray.astype(np.uint8), 65, 165) > 0).astype(np.uint8)
    edge_excl = cv.dilate(strong_edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))) > 0
    interior_mask = (~edge_excl) & (gray > 18.0) & (gray < 240.0)

    g_s08 = cv.GaussianBlur(gray_n, (0, 0), 0.8)
    g_s20 = cv.GaussianBlur(gray_n, (0, 0), 2.0)
    g_s60 = cv.GaussianBlur(gray_n, (0, 0), 6.0)
    g_s18 = cv.GaussianBlur(gray_n, (0, 0), 18.0)

    band_fine = g_s08 - g_s20    # ~2-5 px micro grain / halftone / sand
    band_mid = g_s20 - g_s60     # ~5-16 px paper fiber / brush / fabric / pores
    band_macro = g_s60 - g_s18   # ~16-48 px paint chips / blotchy grunge / weathering

    if int(np.sum(interior_mask)) >= 200:
        fine_rms = float(np.std(band_fine[interior_mask]))
        mid_rms = float(np.std(band_mid[interior_mask]))
        macro_rms = float(np.std(band_macro[interior_mask]))
        # Detect localized bright/dark paint-chip speckles on interior surfaces
        speckle_dev = np.abs(gray_n - g_s60)[interior_mask]
        wear_speckle_ratio = float(np.mean(speckle_dev > 0.085))
        # Zone-split micro-texture: prevents smooth white/bright faces hiding behind over-speckled colored walls
        bright_int = interior_mask & (lum_lin >= 0.32)
        mid_int = interior_mask & (lum_lin >= 0.025) & (lum_lin < 0.32)
        bright_fine_rms = float(np.std(band_fine[bright_int])) if int(np.sum(bright_int)) >= 120 else None
        mid_surf_fine_rms = float(np.std(band_fine[mid_int])) if int(np.sum(mid_int)) >= 120 else None
        # Interior texture stretch anisotropy (catches Generated-coord stretching along extruded side walls)
        tx = cv.Sobel(band_mid, cv.CV_32F, 1, 0, ksize=3)
        ty = cv.Sobel(band_mid, cv.CV_32F, 0, 1, ksize=3)
        jxx = float(np.mean((tx * tx)[interior_mask]))
        jyy = float(np.mean((ty * ty)[interior_mask]))
        jxy = float(np.mean((tx * ty)[interior_mask]))
        tr = jxx + jyy
        det_term = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
        lam1 = max(1e-10, 0.5 * (tr + det_term))
        lam2 = max(1e-10, 0.5 * (tr - det_term))
        tex_stretch_aniso = round(math.sqrt(lam1 / lam2), 3)
    else:
        fine_rms = mid_rms = macro_rms = wear_speckle_ratio = 0.0
        bright_fine_rms = mid_surf_fine_rms = None
        tex_stretch_aniso = 1.0

    band_vals = [("fine_2_5px", fine_rms, 3.5), ("mid_5_16px", mid_rms, 10.0), ("macro_16_48px", macro_rms, 28.0)]
    dom_band_name, dom_band_rms, dom_scale_px = max(band_vals, key=lambda x: x[1])

    # Require actual fine+mid surface micro/meso texture OR localized paint-wear speckles
    # (prevents smooth 3D shading gradients across clean studio objects from triggering grunge!)
    surface_grunge_detected = bool((fine_rms >= 0.0075 and mid_rms >= 0.022) or wear_speckle_ratio >= 0.045)
    if surface_grunge_detected:
        finish_class = "COARSE_TEXTURED_OR_GRUNGE"
        rec_bump = round(float(np.clip(0.18 + 6.5 * mid_rms + 3.0 * fine_rms, 0.18, 0.65)), 2)
    elif fine_rms >= 0.0068:
        finish_class = "FINE_GRAIN_OR_MATTE"
        rec_bump = round(float(np.clip(0.10 + 4.5 * fine_rms, 0.10, 0.35)), 2)
    else:
        finish_class = "CLEAN_SMOOTH"
        rec_bump = 0.0

    # 2. Universal Bilateral Dark Ink Outline / Contour Detector (using HSV Value = max(R,G,B))
    # Detects elongated 1D dark contour strokes sandwiched between brighter lit regions on BOTH sides,
    # filtering out broad dark shadows, night neon gaps, and tiny particle dust gaps.
    v_255 = hsv[:, :, 2]
    k_bh = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    blackhat_v = cv.morphologyEx(v_255, cv.MORPH_BLACKHAT, k_bh)
    dark_valley = (blackhat_v > 42.0) & (v_255 < 86.0)

    step_px = 3
    v_pad = cv.copyMakeBorder(v_255, step_px, step_px, step_px, step_px, cv.BORDER_REPLICATE)
    bilateral_hit = np.zeros((h, w), dtype=bool)
    for dy_s, dx_s in ((0, step_px), (step_px, 0), (step_px, step_px), (step_px, -step_px)):
        p_pos = v_pad[step_px + dy_s:step_px + dy_s + h, step_px + dx_s:step_px + dx_s + w]
        p_neg = v_pad[step_px - dy_s:step_px - dy_s + h, step_px - dx_s:step_px - dx_s + w]
        bilateral_hit |= (p_pos > 110.0) & (p_neg > 110.0) & ((p_pos - v_255) > 42.0) & ((p_neg - v_255) > 42.0)

    raw_outline = (dark_valley & bilateral_hit).astype(np.uint8)
    # Keep only continuous elongated contour strokes (eliminates tiny scattered noise/particle gaps!)
    raw_outline = cv.morphologyEx(raw_outline * 255, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_cc, cc_labels, cc_stats, _ = cv.connectedComponentsWithStats(raw_outline)
    outline_mask = np.zeros((h, w), dtype=bool)
    long_stroke_count = 0
    for lbl in range(1, n_cc):
        bw_c = int(cc_stats[lbl, cv.CC_STAT_WIDTH])
        bh_c = int(cc_stats[lbl, cv.CC_STAT_HEIGHT])
        area_c = int(cc_stats[lbl, cv.CC_STAT_AREA])
        diag_c = math.hypot(bw_c, bh_c)
        # Elongated contour stroke: diagonal >= 22px and not a solid square blob (fill ratio < 0.45)
        fill_c = area_c / float(max(1, bw_c * bh_c))
        if diag_c >= 22.0 and area_c >= 24 and fill_c < 0.48:
            outline_mask |= (cc_labels == lbl)
            long_stroke_count += 1

    fg_bright_edges = int(np.sum((gmag > 12.0) & (cv.dilate((v_255 > 110.0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)))
    outline_px_count = int(np.sum(outline_mask))
    outline_edge_ratio = outline_px_count / float(max(200, fg_bright_edges))

    if outline_px_count >= 120:
        dist_out = cv.distanceTransform(outline_mask.astype(np.uint8), cv.DIST_L2, 3)
        est_outline_width_px = round(float(np.clip(2.0 * np.percentile(dist_out[outline_mask], 75), 1.0, 8.0)), 2)
    else:
        est_outline_width_px = 0.0

    # Suppress false positive dark gaps between bright neon fixtures in dark night scenes
    is_dark_emissive_night = bool(emissive_pixel_ratio > 0.035 and float(np.percentile(lum_lin, 25)) < 0.025)
    ink_outline_detected = bool(
        (not is_dark_emissive_night) and long_stroke_count >= 4 and outline_edge_ratio >= 0.018 and outline_px_count >= 350
    )

    # 4. Special Physical & Stylized Material Sub-Classes
    # 4a) Cel / Toon Ramp Quantization + Shadow Hue Shift
    non_void = lum_lin > 0.015
    hist_l, _ = np.histogram(lum_lin[non_void], bins=32, range=(0.015, 0.98)) if int(np.sum(non_void)) >= 200 else (np.zeros(32), None)
    hist_n = hist_l.astype(np.float64) / max(1.0, float(np.sum(hist_l)))
    # Count sharp discrete peaks separated by valleys (>= 1.5x higher than both neighbors and >= 4% mass)
    cel_peaks = 0
    for bi in range(1, len(hist_n) - 1):
        if hist_n[bi] >= 0.045 and hist_n[bi] > hist_n[bi - 1] * 1.35 and hist_n[bi] > hist_n[bi + 1] * 1.35:
            cel_peaks += 1
    cel_bands_detected = bool(shading_paradigm == "NPR_TOON_SHADER_TO_RGB" or (cel_peaks >= 2 and flat_ratio >= 0.65))
    quantized_bands_est = int(max(2, min(4, cel_peaks))) if cel_bands_detected else 0

    hue_deg_all = hsv[:, :, 0] * 2.0
    lit_chroma = (lum_lin >= 0.35) & (lum_lin <= 0.85) & (sat >= 0.18)
    shd_chroma = (lum_lin >= 0.03) & (lum_lin <= 0.22) & (sat >= 0.18)
    if int(np.sum(lit_chroma)) >= 100 and int(np.sum(shd_chroma)) >= 100:
        hl_rad = np.deg2rad(hue_deg_all[lit_chroma])
        hs_rad = np.deg2rad(hue_deg_all[shd_chroma])
        lit_hue = math.degrees(math.atan2(float(np.mean(np.sin(hl_rad))), float(np.mean(np.cos(hl_rad))))) % 360.0
        shd_hue = math.degrees(math.atan2(float(np.mean(np.sin(hs_rad))), float(np.mean(np.cos(hs_rad))))) % 360.0
        raw_dh = abs(shd_hue - lit_hue)
        shadow_hue_shift_deg = round(min(raw_dh, 360.0 - raw_dh), 1)
    else:
        shadow_hue_shift_deg = 0.0

    # 4b) Specular / Clearcoat / Glossy Lacquer Spike Separation
    p995_l, p96_l, p60_l = [float(np.percentile(lum_lin, q)) for q in (99.5, 96.0, 60.0)]
    spec_spike = max(0.0, p995_l - p96_l)
    hi_low_sat_frac = float(np.mean((lum_lin >= max(0.45, p96_l)) & (sat <= 0.22)))
    clearcoat_or_glossy_detected = bool(
        (spec_spike >= 0.14) and (0.001 <= hi_low_sat_frac <= 0.08) and (roughness_median <= 0.42) and (not is_dark_emissive_night)
    )

    # 4c) Subsurface Scattering (SSS) / Translucency Terminator Chroma Boost Hint
    # Real SSS materials (wax/jade/skin/jelly/translucent plastic) boost saturation in the 0.08..0.35 terminator band
    term_mask = (lum_lin >= 0.08) & (lum_lin <= 0.35)
    bright_body = (lum_lin >= 0.45) & (lum_lin <= 0.85)
    if int(np.sum(term_mask)) >= 200 and int(np.sum(bright_body)) >= 200:
        sat_term = float(np.median(sat[term_mask]))
        sat_body = float(np.median(sat[bright_body]))
        sss_chroma_boost = round(sat_term - sat_body, 3)
        sss_hint_detected = bool(
            sss_chroma_boost >= 0.16 and sat_term >= 0.42 and (not is_dark_emissive_night) and shading_paradigm != "NPR_TOON_SHADER_TO_RGB"
        )
    else:
        sss_chroma_boost = 0.0
        sss_hint_detected = False

    # 4d) Glass / Transmission / Refraction Heuristic Hint (strictly advisory; requires visual check)
    # Looks for tight neutral specular highlights co-existing with sharp high-gradient internal refraction edges
    g_edge_sharp = np.hypot(cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3), cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3))
    refract_score = round(float(np.mean((g_edge_sharp > 45.0) & (sat < 0.28) & (lum_lin > 0.20) & (lum_lin < 0.92))), 4)
    glass_hint_detected = bool(clearcoat_or_glossy_detected and refract_score >= 0.045 and emissive_pixel_ratio < 0.02)

    return {
        "shading_paradigm": shading_paradigm,
        "flat_gradient_ratio": round(flat_ratio, 4),
        "continuous_shading_ratio": round(mid_grad_ratio, 4),
        "emissive_area_ratio": round(emissive_pixel_ratio, 4),
        "estimated_principled_bsdf": {
            "roughness_median": round(roughness_median, 3),
            "roughness_range": [roughness_min, roughness_max],
            "procedural_roughness_noise_detected": has_roughness_noise_map,
            "highlight_to_midtone_saturation_ratio": round(metallic_ratio, 3),
            "surface_type_hint": "colored_specular_or_emissive" if metallic_ratio > 0.95 else "dielectric_pbr",
            "allow_emission_for_non_emissive_surfaces": False,
        },
        "surface_texture_profile": {
            "surface_finish_class": finish_class,
            "surface_grunge_detected": surface_grunge_detected,
            "fine_grain_rms_2_5px": round(fine_rms, 4),
            "mid_texture_rms_5_16px": round(mid_rms, 4),
            "macro_grunge_rms_16_48px": round(macro_rms, 4),
            "wear_speckle_area_ratio": round(wear_speckle_ratio, 4),
            "bright_surface_fine_rms": round(bright_fine_rms, 4) if bright_fine_rms is not None else None,
            "mid_surface_fine_rms": round(mid_surf_fine_rms, 4) if mid_surf_fine_rms is not None else None,
            "texture_stretch_anisotropy": tex_stretch_aniso,
            "dominant_texture_band": dom_band_name,
            "dominant_texture_scale_px": dom_scale_px,
            "recommended_bump_strength": rec_bump,
            "recommended_blender_noise_scales_note": (
                "Use multi-scale Object-coord procedural layers (macro Scale 3~8 + mid Scale 12~28 + fine Scale 40~60) "
                "or image texture driving both Base Color and Roughness/Bump; avoid Generated-coord stretching or >80 noise"
                if surface_grunge_detected else "Clean low-bump surface; keep bump minimal or off"
            ),
        },
        "stylized_ink_outline": {
            "ink_outline_detected": ink_outline_detected,
            "bilateral_contour_edge_ratio": round(outline_edge_ratio, 4),
            "contour_pixel_count": outline_px_count,
            "estimated_outline_width_px": est_outline_width_px,
            "recommended_blender_method": (
                f"Grease Pencil Line Art / Freestyle / Solidify Inverted Hull (~{est_outline_width_px}px dark stroke)"
                if ink_outline_detected else "NONE"
            ),
        },
        "special_material_signals": {
            "toon_shading_profile": {
                "cel_bands_detected": cel_bands_detected,
                "quantized_band_count_estimate": quantized_bands_est,
                "shadow_hue_shift_deg": shadow_hue_shift_deg,
            },
            "specular_and_clearcoat_profile": {
                "clearcoat_or_glossy_specular_detected": clearcoat_or_glossy_detected,
                "specular_peak_excess": round(spec_spike, 4),
                "neutral_highlight_area_ratio": round(hi_low_sat_frac, 4),
            },
            "sss_or_translucency_hint": {
                "sss_hint_detected": sss_hint_detected,
                "terminator_chroma_boost": sss_chroma_boost,
            },
            "transmission_glass_hint": {
                "glass_or_refraction_hint_detected": glass_hint_detected,
                "refraction_edge_score": refract_score,
                "note": "Advisory optical hint; verify visually before enabling Principled Transmission/IOR",
            },
        },
    }


def analyze_patch_material_profile(patch_bgr, global_light_info=None):
    """Lightweight per-object material profiler so mixed-material scenes (e.g. glossy glass
    + matte metal + rough paper in one frame) never collapse into a single global average."""
    ph, pw = patch_bgr.shape[:2]
    if ph < 24 or pw < 24:
        return None
    rgb = cv.cvtColor(patch_bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    hsv = cv.cvtColor(patch_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    val_u8 = np.max(rgb, axis=2).astype(np.uint8)
    gray = cv.cvtColor(patch_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)

    # Local roughness from highlight-to-midtone edge sharpness
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    p92 = float(np.percentile(lum, 92))
    p50 = float(np.percentile(lum, 50))
    hi_grad = gmag[lum >= max(0.18, p92)]
    spec_sharp = float(np.percentile(hi_grad, 80)) if len(hi_grad) >= 20 else float(np.percentile(gmag, 80))
    rough_med = round(float(np.clip(0.88 - 0.0085 * spec_sharp, 0.08, 0.92)), 3)

    # Local metallic / colored specular vs dielectric
    mid_m = (lum >= 0.08) & (lum <= 0.55)
    hi_m = lum >= max(0.45, float(np.percentile(lum, 95)))
    sat_mid = float(np.median(sat[mid_m])) if int(np.sum(mid_m)) >= 30 else 0.2
    sat_hi = float(np.median(sat[hi_m])) if int(np.sum(hi_m)) >= 15 else sat_mid
    met_ratio = sat_hi / max(0.05, sat_mid)
    metallic_hint = "METALLIC_OR_COLORED_SPEC" if (met_ratio > 0.95 and sat_hi > 0.22) else "DIELECTRIC"

    # Local surface texture / grunge on non-edge interior
    edge_m = cv.dilate(cv.Canny(gray.astype(np.uint8), 45, 130), np.ones((7, 7), np.uint8)) > 0
    interior = (~edge_m) & (lum > 0.015) & (lum < 0.92)
    if int(np.sum(interior)) >= 100:
        b1 = cv.GaussianBlur(lum, (0, 0), 1.0)
        b2 = cv.GaussianBlur(lum, (0, 0), 2.8)
        b3 = cv.GaussianBlur(lum, (0, 0), 7.5)
        f_rms = float(np.std((b1 - b2)[interior]))
        m_rms = float(np.std((b2 - b3)[interior]))
        spk = float(np.mean(np.abs((b1 - b2)[interior]) > 0.035))
    else:
        f_rms, m_rms, spk = 0.0, 0.0, 0.0

    grunge_on = bool((f_rms >= 0.0080 and m_rms >= 0.024) or (spk >= 0.050))
    if grunge_on:
        finish = "COARSE_TEXTURED_OR_GRUNGE"
        bump_str = round(float(np.clip(0.18 + m_rms * 7.5, 0.22, 0.78)), 2)
    elif f_rms >= 0.0058 or m_rms >= 0.017:
        finish = "FINE_GRAIN_OR_MATTE"
        bump_str = round(float(np.clip(0.06 + m_rms * 4.5, 0.08, 0.24)), 2)
    else:
        finish = "CLEAN_SMOOTH"
        bump_str = 0.0

    # Per-object dark ink outline check using HSV Value
    v_f = val_u8.astype(np.float32)
    dark_px = v_f < 68.0
    bright_px = v_f > 118.0
    kern3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bright_dil = cv.dilate(bright_px.astype(np.uint8), kern3, iterations=2) > 0
    dark_adj = dark_px & bright_dil
    n_cc, _, st_cc, _ = cv.connectedComponentsWithStats(dark_adj.astype(np.uint8))
    long_strokes = 0
    for lbl in range(1, n_cc):
        bx, by, cw, ch, ca = [int(v) for v in st_cc[lbl]]
        if math.hypot(cw, ch) >= max(14.0, 0.12 * min(pw, ph)) and ca >= 18 and (ca / max(1.0, float(cw * ch))) < 0.48:
            long_strokes += 1
    obj_ink_outline = bool(long_strokes >= 2 and float(np.mean(dark_adj)) >= 0.012)

    # Local clearcoat / bright silhouette rim hint
    spec_spike = max(0.0, float(np.percentile(lum, 99.5)) - float(np.percentile(lum, 95.0)))
    clearcoat_hint = bool(spec_spike >= 0.14 and rough_med <= 0.42)

    return {
        "recommended_roughness": rough_med,
        "surface_finish_class": finish,
        "surface_grunge_detected": grunge_on,
        "recommended_bump_strength": bump_str,
        "metallic_hint": metallic_hint,
        "ink_outline_detected": obj_ink_outline,
        "clearcoat_or_glossy_hint": clearcoat_hint,
        "fine_mid_texture_rms": [round(f_rms, 4), round(m_rms, 4)],
    }


def analyze_lighting_and_volumetrics(bgr):
    """Pillar 4: Evidence-gated lighting, bloom/soft-glow, and volumetric scattering analysis.
    Works across night emissive scenes, daylight exteriors, clean studio product renders,
    and flat/NPR shading without forcing volume fog or bloom when absent in the source.
    """
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    lin = srgb_to_linear(rgb)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0

    p05, p20, p50, p85, p99 = [float(np.percentile(lum, q)) for q in (5, 20, 50, 85, 99)]
    key_to_fill_ratio = round(p85 / max(1e-4, p20), 2)

    # Localized highlight / emissive core detection:
    # A uniform bright white studio/UI background must NOT be misclassified as a glowing light core,
    # while AgX/Filmic soft-compressed emissive cores (p99 in 0.22..0.32 on dark scenes) are still captured.
    core_thresh = max(0.32, p85 + 0.25 * max(0.0, p99 - p85), float(np.percentile(lum, 97.0)))
    core_binary = (lum >= core_thresh).astype(np.uint8)
    core_area_frac = float(np.mean(core_binary))
    has_localized_light_cores = bool(
        (0.0004 <= core_area_frac <= 0.22) and ((p99 - p50) >= 0.16) and (p99 >= 0.32)
    )
    if not has_localized_light_cores and p99 >= 0.22 and (p99 / max(0.012, p50)) >= 5.0 and (p99 - p50) >= 0.14:
        soft_thresh = max(0.20, p85 + 0.30 * max(0.0, p99 - p85), float(np.percentile(lum, 97.5)))
        soft_binary = ((lum >= soft_thresh) & ((sat >= 0.22) | (lum >= 0.28))).astype(np.uint8)
        soft_frac = float(np.mean(soft_binary))
        if 0.0004 <= soft_frac <= 0.18:
            core_binary = soft_binary
            core_area_frac = soft_frac
            has_localized_light_cores = True

    radii = np.arange(0, 42, dtype=np.int32)
    if has_localized_light_cores:
        dist_map = cv.distanceTransform(1 - core_binary, cv.DIST_L2, 5)
        radial_lum = []
        radial_sat = []
        d_int = np.rint(dist_map).astype(np.int32)
        for r in radii:
            ring = (d_int == r)
            if np.sum(ring) >= 12:
                radial_lum.append(float(np.median(lum[ring])))
                radial_sat.append(float(np.median(sat[ring])))
            else:
                radial_lum.append(radial_lum[-1] if radial_lum else p50)
                radial_sat.append(radial_sat[-1] if radial_sat else 0.35)
        radial_lum = np.asarray(radial_lum, dtype=np.float64)
        radial_sat = np.asarray(radial_sat, dtype=np.float64)

        l_core = float(np.mean(radial_lum[:2]))
        l_near = float(np.mean(radial_lum[3:9]))
        l_far = float(np.mean(radial_lum[12:28]))
        l_ambient = float(min(p20, np.mean(radial_lum[32:])))

        core_excess = max(0.05, l_core - l_ambient)
        near_bloom_excess = max(0.0, l_near - l_ambient)
        far_volumetric_excess = max(0.0, l_far - l_ambient)
        near_to_core_ratio = near_bloom_excess / core_excess
        far_to_core_ratio = far_volumetric_excess / core_excess

        # Exponential decay scale lambda_fog from r=6 to r=24
        excess_6 = max(1e-4, float(radial_lum[6]) - l_ambient)
        excess_24 = max(1e-5, float(radial_lum[24]) - l_ambient)
        decay_slope = max(0.015, math.log(excess_6 / max(1e-5, excess_24)) / 18.0)
        lambda_fog_px = round(1.0 / decay_slope, 2)

        # Evidence gates: require real optical bloom / volumetric scatter tail, not crisp step edges
        bloom_detected = bool(near_bloom_excess >= 0.022 and near_to_core_ratio >= 0.055)
        volume_fog_detected = bool(
            far_volumetric_excess >= 0.012 and far_to_core_ratio >= 0.022 and lambda_fog_px >= 6.5
        )

        # Core-to-rim saturation ratio (detects AgX highlight rolloff)
        sat_core = float(np.mean(radial_sat[:2]))
        sat_rim = float(np.max(radial_sat[2:8]))
        core_sat_rolloff = round(sat_core / max(0.05, sat_rim), 3)

        if volume_fog_detected:
            halo_to_ambient_ratio = near_bloom_excess / max(0.008, l_ambient)
            est_anisotropy = float(np.clip(0.35 + 0.14 * math.log1p(max(0.0, halo_to_ambient_ratio)), 0.15, 0.85))
            est_density = float(np.clip(0.002 + 0.014 * far_to_core_ratio, 0.002, 0.035))
        else:
            est_anisotropy = 0.0
            est_density = 0.0

        emissive_core_detected = bool(
            float(np.mean((lum >= 0.28) & (sat >= 0.30))) >= 0.003 and p50 < 0.38
        )
        rec_emis_strength = (
            round(float(np.clip(0.95 + 0.25 * (l_core / max(0.2, sat_core)), 0.9, 1.45)), 2)
            if emissive_core_detected else 0.0
        )
        rec_emis_gamma = round(float(np.clip(1.0 + 0.15 * core_sat_rolloff, 1.02, 1.25)), 2) if emissive_core_detected else 1.0
        rec_soft_glow_mix = (
            round(float(np.clip(0.03 + 0.32 * near_bloom_excess, 0.03, 0.14)), 3)
            if bloom_detected else 0.0
        )
        clean_pad_px = int(round(max(12.0, min(32.0, 2.2 * lambda_fog_px)))) if (bloom_detected or volume_fog_detected) else 8
    else:
        radial_lum = np.full(len(radii), p50, dtype=np.float64)
        l_core = p99
        l_near = p50
        l_far = p50
        l_ambient = p20
        near_bloom_excess = 0.0
        far_volumetric_excess = 0.0
        near_to_core_ratio = 0.0
        far_to_core_ratio = 0.0
        lambda_fog_px = 0.0
        core_sat_rolloff = 1.0
        bloom_detected = False
        volume_fog_detected = False
        emissive_core_detected = False
        est_anisotropy = 0.0
        est_density = 0.0
        rec_emis_strength = 0.0
        rec_emis_gamma = 1.0
        rec_soft_glow_mix = 0.0
        clean_pad_px = 8

    # Dominant light direction from global luminance gradient on midtones
    gy, gx = np.gradient(cv.GaussianBlur(lum, (31, 31), 9.0))
    mean_gx, mean_gy = float(np.mean(gx)), float(np.mean(gy))
    light_azimuth_deg = round(math.degrees(math.atan2(-mean_gy, mean_gx)), 1)

    # 3. Universal Shadow Penumbra Softness & Key-Light Hardness
    h_img, w_img = bgr.shape[:2]
    gray_l = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    sm_l = cv.GaussianBlur(gray_l, (5, 5), 0.75)
    dx_l = cv.Sobel(sm_l, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
    dy_l = cv.Sobel(sm_l, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
    # Sample edges in dark-to-midtone shadow transition zones (exclude blazing emissive highlights)
    shadow_zone_mask = ((lum >= 0.008) & (lum <= 0.55)).astype(np.uint8) * 255
    shadow_edges = sample(gray_l, dx_l, dy_l, shadow_zone_mask, 10)
    if len(shadow_edges) >= 8:
        penumbra_px = round(float(np.percentile(shadow_edges[:, 2], 45)), 2)
    else:
        penumbra_px = 4.0

    if penumbra_px <= 2.8:
        key_hardness = "HARD_KEY_OR_SMALL_AREA"
        rec_area_note = "Small Area Light (size 0.3~1.5m) or narrow-angle Sun; avoid oversized flat fill lamps"
    elif penumbra_px <= 6.5:
        key_hardness = "MEDIUM_SOFT_AREA"
        rec_area_note = "Medium Area Light (size 1.5~3.5m) with directional contrast & real cast shadows"
    else:
        key_hardness = "BROAD_DIFFUSE_OR_HDRI"
        rec_area_note = "Large soft Area Light / HDRI dome with gentle shadow penumbra"

    # 4. Universal Radial Vignette / Peripheral Falloff Detector (ignores pure black void backgrounds)
    yy, xx = np.ogrid[:h_img, :w_img]
    r_norm = np.hypot((xx - w_img * 0.5) / (0.5 * w_img), (yy - h_img * 0.5) / (0.5 * h_img))
    center_mask = r_norm <= 0.55
    corner_mask = r_norm >= 0.92
    center_nonblack = lum[center_mask & (lum > 0.012)]
    corner_all = lum[corner_mask]
    corner_nonblack = lum[corner_mask & (lum > 0.012)]
    corner_void_frac = float(np.mean(corner_all <= 0.008)) if len(corner_all) else 1.0

    if len(center_nonblack) >= 100 and len(corner_nonblack) >= 60 and corner_void_frac < 0.45:
        c_med = float(np.median(center_nonblack))
        k_med = float(np.median(corner_nonblack))
        c2k_ratio = round(k_med / max(1e-4, c_med), 3)
        vignette_detected = bool(c2k_ratio <= 0.78)
        vig_strength = round(float(np.clip(1.0 - c2k_ratio, 0.0, 0.75)), 3) if vignette_detected else 0.0
    else:
        c_med = float(np.median(center_nonblack)) if len(center_nonblack) else p50
        k_med = float(np.median(corner_all)) if len(corner_all) else p05
        c2k_ratio = round(k_med / max(1e-4, c_med), 3)
        vignette_detected = False
        vig_strength = 0.0

    # 5. Split-Tone Key vs Fill Light Chromaticity, Elevation & Backlight Rim Detection
    lit_mask = (lum >= max(0.20, float(np.percentile(lum, 72)))) & (lum <= float(np.percentile(lum, 96)))
    fill_mask = (lum >= 0.015) & (lum <= max(0.08, float(np.percentile(lum, 35))))
    if int(np.sum(lit_mask)) >= 120:
        key_med_rgb = np.median(rgb[lit_mask], axis=0)
        key_hex = rgb_to_hex(key_med_rgb)
        key_warmth = float((key_med_rgb[0] - key_med_rgb[2]) / max(15.0, float(np.mean(key_med_rgb))))
    else:
        key_med_rgb = np.array([200.0, 200.0, 200.0], dtype=np.float32)
        key_hex = "#C8C8C8"
        key_warmth = 0.0
    if int(np.sum(fill_mask)) >= 120:
        fill_med_rgb = np.median(rgb[fill_mask], axis=0)
        fill_hex = rgb_to_hex(fill_med_rgb)
        fill_warmth = float((fill_med_rgb[0] - fill_med_rgb[2]) / max(15.0, float(np.mean(fill_med_rgb))))
    else:
        fill_med_rgb = np.array([40.0, 40.0, 40.0], dtype=np.float32)
        fill_hex = "#282828"
        fill_warmth = 0.0

    warmth_delta = round(key_warmth - fill_warmth, 3)
    # True split-tone lighting (e.g. warm key + cool fill/shadow, or cool key + warm fill) has OPPOSITE warmth signs,
    # preventing white-vs-red albedo differences from falsely triggering split-tone lighting!
    opposite_temp_poles = bool(
        (key_warmth >= 0.08 and fill_warmth <= -0.06)
        or (key_warmth <= -0.06 and fill_warmth >= 0.08)
    )
    key_fill_color_dist = float(np.linalg.norm(
        (key_med_rgb / max(10.0, float(np.mean(key_med_rgb)))) - (fill_med_rgb / max(10.0, float(np.mean(fill_med_rgb))))
    ))
    split_tone_detected = bool(opposite_temp_poles and abs(warmth_delta) >= 0.24 and key_fill_color_dist >= 0.22)

    # Key elevation hint from global vertical vs horizontal luminance gradient
    grad_norm = math.hypot(mean_gx, mean_gy)
    if grad_norm < 0.00018:
        key_elevation = "FLAT_FRONTAL_OR_SYMMETRIC"
    elif abs(mean_gy) >= abs(mean_gx) * 1.15:
        key_elevation = "TOP_KEY" if mean_gy < 0 else "BOTTOM_UP_LIGHT"
    else:
        key_elevation = "SIDE_KEY"

    # Rim / Backlight Silhouette Glow Detector (bright thin rim along silhouette edges vs adjacent interior)
    sil_edges = cv.Canny(gray_l.astype(np.uint8), 65, 165)
    rim_band = cv.dilate(sil_edges, np.ones((3, 3), np.uint8)) > 0
    near_ring = (cv.dilate(sil_edges, np.ones((11, 11), np.uint8)) > 0) & (~cv.dilate(sil_edges, np.ones((5, 5), np.uint8)) > 0)
    if int(np.sum(rim_band)) >= 120 and int(np.sum(near_ring)) >= 200:
        rim_lum_p80 = float(np.percentile(lum[rim_band], 80))
        near_lum_p65 = float(np.percentile(lum[near_ring], 65))
        rim_excess = round(max(0.0, rim_lum_p80 - near_lum_p65), 4)
        # Only trigger when rim edges distinctly outshine adjacent surfaces and are not just white-on-black text
        fg_mid_frac = float(np.mean((lum > 0.03) & (lum < 0.45)))
        rim_light_detected = bool(rim_excess >= 0.14 and rim_lum_p80 >= 0.35 and fg_mid_frac >= 0.25 and (not has_localized_light_cores or p50 > 0.03))
    else:
        rim_excess = 0.0
        rim_light_detected = False

    # 6. Daylight / Exterior Depth-Based Atmospheric Haze Detector (complements night point-light volume fog)
    # Compares upper-distant band (15..45% height) vs lower-foreground band (62..95% height):
    # Real atmospheric aerial perspective lifts shadow floor (p05) while compressing local contrast in the distance.
    upper_band = lum[int(0.15 * h_img):int(0.45 * h_img), :]
    lower_band = lum[int(0.62 * h_img):int(0.95 * h_img), :]
    upper_sat = sat[int(0.15 * h_img):int(0.45 * h_img), :]
    lower_sat = sat[int(0.62 * h_img):int(0.95 * h_img), :]
    u_p05, u_p90 = float(np.percentile(upper_band, 5)), float(np.percentile(upper_band, 90))
    l_p05, l_p90 = float(np.percentile(lower_band, 5)), float(np.percentile(lower_band, 90))
    u_contrast = max(1e-4, u_p90 - u_p05)
    l_contrast = max(1e-4, l_p90 - l_p05)
    black_floor_lift = round(u_p05 - l_p05, 4)
    contrast_decay = round(u_contrast / l_contrast, 3)
    # Exclude pure white studio backdrops (where upper band p05 > 0.42 or overall p50 > 0.55) and pure black voids
    daylight_haze_detected = bool(
        (black_floor_lift >= 0.055)
        and (contrast_decay <= 0.68)
        and (0.06 <= u_p05 <= 0.42)
        and (l_p05 <= 0.12)
        and (p50 <= 0.55)
        and (float(np.median(upper_sat)) <= float(np.median(lower_sat)) + 0.08)
    )

    # 7. Ground / Planar Specular Reflection & Wet-Floor Streak Detector
    # Checks lower 32% of frame for either (a) vertically mirrored planar reflection of middle band,
    # or (b) elongated vertical glossy reflection streaks (wet street / polished dark marble).
    mid_y0, mid_y1 = int(0.36 * h_img), int(0.68 * h_img)
    bot_y0, bot_y1 = int(0.68 * h_img), h_img
    mid_strip = lum[mid_y0:mid_y1, :]
    bot_strip = lum[bot_y0:bot_y1, :]
    bot_sat_strip = sat[bot_y0:bot_y1, :]
    rh = min(mid_strip.shape[0], bot_strip.shape[0])
    if rh >= 24:
        m_res = cv.resize(mid_strip[:rh, :], (160, 48), interpolation=cv.INTER_AREA)
        b_res = cv.resize(np.flipud(bot_strip[:rh, :]), (160, 48), interpolation=cv.INTER_AREA)
        m_std, b_std = float(np.std(m_res)), float(np.std(b_res))
        if m_std >= 0.035 and b_std >= 0.035:
            mirror_ncc = round(float(np.mean(((m_res - np.mean(m_res)) / m_std) * ((b_res - np.mean(b_res)) / b_std))), 3)
        else:
            mirror_ncc = 0.0
    else:
        mirror_ncc = 0.0

    # Vertical & orientation-invariant specular streak anisotropy on lower 32%
    bot_gray = gray_l[bot_y0:bot_y1, :]
    bx_s = cv.Sobel(bot_gray, cv.CV_32F, 1, 0, ksize=3)
    by_s = cv.Sobel(bot_gray, cv.CV_32F, 0, 1, ksize=3)
    bx_g = np.abs(bx_s)
    by_g = np.abs(by_s)
    bot_bright = bot_strip >= max(0.08, float(np.percentile(bot_strip, 78)))
    if int(np.sum(bot_bright)) >= 80 and float(np.std(bot_strip)) >= 0.025:
        v_streak_ratio = round(float(np.mean(bx_g[bot_bright]) / max(1.0, float(np.mean(by_g[bot_bright])))), 3)
        jxx = cv.GaussianBlur(bx_s * bx_s, (9, 9), 2.0)
        jyy = cv.GaussianBlur(by_s * by_s, (9, 9), 2.0)
        jxy = cv.GaussianBlur(bx_s * by_s, (9, 9), 2.0)
        coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2)) / np.maximum(1e-5, jxx + jyy)
        oriented_streak_coherence = round(float(np.median(coh[bot_bright])), 3)
    else:
        v_streak_ratio = 1.0
        oriented_streak_coherence = 0.0

    mid_mean_l = float(np.mean(mid_strip))
    bot_mean_l = float(np.mean(bot_strip))
    bot_med_sat = float(np.median(bot_sat_strip[bot_bright])) if int(np.sum(bot_bright)) >= 80 else 0.0

    # Physical Fresnel reflection in lower strip is darker than upper scene (excludes symmetric white studio backdrops!)
    if mirror_ncc >= 0.42 and bot_mean_l <= mid_mean_l * 0.92 and p50 < 0.62:
        ground_refl_detected = True
        refl_type = "PLANAR_MIRROR_OR_WATER"
        rec_ground_rough = round(float(np.clip(0.28 - 0.35 * (mirror_ncc - 0.42), 0.02, 0.25)), 2)
    elif (
        (v_streak_ratio >= 1.95 and float(np.percentile(bot_strip, 95)) >= 0.18 and bot_med_sat < 0.52 and bot_mean_l <= mid_mean_l * 0.95)
        or (oriented_streak_coherence >= 0.62 and float(np.percentile(bot_strip, 95)) >= 0.14 and p50 < 0.25 and bot_mean_l <= mid_mean_l * 0.95)
    ):
        ground_refl_detected = True
        refl_type = "WET_OR_GLOSSY_STREAKS"
        rec_ground_rough = 0.14
    else:
        ground_refl_detected = False
        refl_type = "NONE"
        rec_ground_rough = None

    # 8. Universal Indirect Emissive Spill & Non-Core Crushed-Black Profile
    # Measures whether localized luminous sources (lamps, screens, signs, fire, sci-fi fixtures)
    # actually cast chromatic indirect illumination onto surrounding geometry (8..35 px ring).
    em_core_thresh = max(0.24, float(np.percentile(lum, 96.0)))
    em_core_mask = ((lum >= em_core_thresh) & (sat >= 0.28)).astype(np.uint8)
    em_core_frac = float(np.mean(em_core_mask))
    if 0.0003 <= em_core_frac <= 0.18 and int(np.sum(em_core_mask)) >= 20 and p50 < 0.30:
        dist_em = cv.distanceTransform(1 - em_core_mask, cv.DIST_L2, 5)
        spill_ring = (dist_em >= 8.0) & (dist_em <= 35.0)
        non_core_zone = dist_em >= 6.0
        spill_ring_lum = float(np.mean(lum[spill_ring])) if np.any(spill_ring) else 0.0
        spill_ring_sat = float(np.mean(sat[spill_ring])) if np.any(spill_ring) else 0.0
        spill_ring_chroma = float(np.mean(lum[spill_ring] * sat[spill_ring])) if np.any(spill_ring) else 0.0
        non_core_crushed_black = float(np.mean(lum[non_core_zone] < 0.008)) if np.any(non_core_zone) else float(np.mean(lum < 0.008))
        emissive_indirect_spill_detected = bool(
            emissive_core_detected and spill_ring_chroma >= 0.018 and spill_ring_lum >= 0.038
        )
    else:
        spill_ring_lum = 0.0
        spill_ring_sat = 0.0
        spill_ring_chroma = 0.0
        non_core_crushed_black = float(np.mean(lum < 0.008))
        emissive_indirect_spill_detected = False

    # 9. Universal 4-Quadrant Regional Tonal Dynamic Range (catches flat wash-out anywhere in frame)
    hh, ww = h_img // 2, w_img // 2
    quad_slices = {
        "top_left": lum[:hh, :ww],
        "top_right": lum[:hh, ww:],
        "bottom_left": lum[hh:, :ww],
        "bottom_right": lum[hh:, ww:],
    }
    quad_contrasts = {}
    active_q_vals = []
    for qname, ql in quad_slices.items():
        q_nonvoid = float(np.mean(ql >= 0.008))
        if q_nonvoid >= 0.25:
            q_p50 = float(np.percentile(ql, 50))
            q_p97 = float(np.percentile(ql, 97))
            qc = round(q_p97 / max(0.005, q_p50), 3)
            quad_contrasts[qname] = qc
            active_q_vals.append(qc)
        else:
            quad_contrasts[qname] = None
    active_quad_med = round(float(np.median(active_q_vals)), 3) if active_q_vals else 1.0
    global_hi_to_med = round(float(np.percentile(lum, 97)) / max(0.005, p50), 3)

    # 10. Machine-Readable Physical Render Requirements (Scheme A — zero speed-compromise spec)
    req_cycles_reasons = []
    if emissive_indirect_spill_detected:
        req_cycles_reasons.append("localized emissive indirect spill onto surrounding geometry")
    if volume_fog_detected:
        req_cycles_reasons.append("volumetric scattering / atmospheric halo decay")
    if ground_refl_detected:
        req_cycles_reasons.append(f"surface specular/planar reflection ({refl_type})")
    require_cycles = bool(len(req_cycles_reasons) > 0)
    if (bloom_detected and volume_fog_detected) or (bloom_detected and far_to_core_ratio >= 0.030 and near_to_core_ratio >= 0.10):
        glare_tier = "DUAL_NEAR_AND_BROAD"
    elif bloom_detected:
        glare_tier = "SINGLE_SOFT_GLOW"
    else:
        glare_tier = "NONE"

    return {
        "linear_luminance_percentiles": {
            "p05_shadow_floor": round(p05, 4),
            "p20_fill_level": round(p20, 4),
            "p50_midtone": round(p50, 4),
            "p85_key_level": round(p85, 4),
            "p99_highlight_core": round(p99, 4),
        },
        "key_to_fill_linear_ratio": key_to_fill_ratio,
        "dominant_light_direction_2d_deg": light_azimuth_deg,
        "lighting_color_and_rig_profile": {
            "key_lit_hex": key_hex,
            "fill_shadow_hex": fill_hex,
            "key_warmth_index": round(key_warmth, 3),
            "fill_warmth_index": round(fill_warmth, 3),
            "key_fill_warmth_delta": warmth_delta,
            "split_tone_detected": split_tone_detected,
            "rim_light_detected": rim_light_detected,
            "rim_boundary_excess_lum": rim_excess,
            "estimated_key_elevation": key_elevation,
        },
        "atmospheric_depth_haze": {
            "daylight_atmospheric_haze_detected": daylight_haze_detected,
            "distant_black_floor_lift": black_floor_lift,
            "distant_contrast_decay_ratio": contrast_decay,
            "recommended_blender_mist_note": (
                "Enable World Mist Pass or low-density distant Principled Volume haze to lift distant shadow floor"
                if daylight_haze_detected else "NONE"
            ),
        },
        "ground_reflection_profile": {
            "ground_reflection_detected": ground_refl_detected,
            "reflection_type": refl_type,
            "mirror_ncc": mirror_ncc,
            "vertical_streak_ratio": v_streak_ratio,
            "oriented_streak_coherence": oriented_streak_coherence,
            "recommended_ground_roughness": rec_ground_rough,
        },
        "emissive_spill_profile": {
            "emissive_indirect_spill_detected": emissive_indirect_spill_detected,
            "emissive_core_fraction": round(em_core_frac, 4),
            "spill_ring_linear_lum": round(spill_ring_lum, 4),
            "spill_ring_sat": round(spill_ring_sat, 4),
            "spill_ring_chromatic_energy": round(spill_ring_chroma, 4),
            "non_core_crushed_black_ratio": round(non_core_crushed_black, 4),
        },
        "regional_tonal_contrast_profile": {
            "quadrant_highlight_to_median": quad_contrasts,
            "active_quadrant_median_contrast": active_quad_med,
            "global_highlight_to_median": global_hi_to_med,
        },
        "shadow_and_key_light_hardness": {
            "shadow_penumbra_width_px": penumbra_px,
            "estimated_key_light_hardness": key_hardness,
            "recommended_blender_rig_note": rec_area_note,
        },
        "vignette_profile": {
            "vignette_detected": vignette_detected,
            "center_nonblack_linear_lum": round(c_med, 4),
            "corner_linear_lum": round(k_med, 4),
            "corner_to_center_ratio": c2k_ratio,
            "corner_pure_black_fraction": round(corner_void_frac, 3),
            "recommended_compositor_vignette_strength": vig_strength,
        },
        "radial_halo_profile": {
            "has_localized_light_cores": has_localized_light_cores,
            "core_area_fraction": round(core_area_frac, 4),
            "core_linear_lum": round(l_core, 4),
            "near_bloom_linear_lum": round(l_near, 4),
            "near_bloom_excess": round(near_bloom_excess, 4),
            "near_to_core_ratio": round(near_to_core_ratio, 4),
            "far_volumetric_linear_lum": round(l_far, 4),
            "far_volumetric_excess": round(far_volumetric_excess, 4),
            "far_to_core_ratio": round(far_to_core_ratio, 4),
            "ambient_floor_linear_lum": round(l_ambient, 4),
            "halo_decay_length_lambda_px": lambda_fog_px,
            "recommended_clean_plate_pad_px": clean_pad_px,
            "core_to_rim_saturation_ratio": core_sat_rolloff,
            "bloom_detected": bloom_detected,
            "volume_fog_detected": volume_fog_detected,
            "emissive_core_detected": emissive_core_detected,
            "radial_lum_curve_0_to_35px": [round(float(v), 4) for v in radial_lum[:36]],
        },
        "recommended_blender_lighting_and_volume": {
            "view_transform": "AgX",
            "look": "AgX - Medium High Contrast" if key_to_fill_ratio >= 3.0 else "AgX - Base Contrast",
            "enable_emissive_materials": emissive_core_detected,
            "enable_emissive_indirect_spill": emissive_indirect_spill_detected,
            "non_additive_emission_strength_fg": rec_emis_strength,
            "non_additive_emission_gamma": rec_emis_gamma,
            "enable_volume_fog": volume_fog_detected,
            "principled_volume_density": round(est_density, 4),
            "principled_volume_anisotropy": round(est_anisotropy, 3),
            "enable_bloom_or_soft_glow": bloom_detected,
            "recommended_soft_glow_mix": rec_soft_glow_mix,
            "world_background_strength": round(float(np.clip(p05 * 8.0, 0.05, 1.2)), 3),
        },
        "recommended_blender_render_requirements": {
            "policy": "SCHEME_A_PHYSICS_DRIVEN_NO_SPEED_COMPROMISE",
            "require_cycles": require_cycles,
            "allow_eevee_no_rt": False,
            "min_diffuse_bounces": 2 if emissive_indirect_spill_detected else 1,
            "min_glossy_bounces": 2 if (ground_refl_detected or emissive_indirect_spill_detected or key_to_fill_ratio >= 2.5) else 1,
            "min_transmission_bounces": 2 if ground_refl_detected else 1,
            "min_volume_bounces": 1 if volume_fog_detected else 0,
            "compositor_glare_tier": glare_tier,
            "anti_compromise_reasons": req_cycles_reasons,
        },
    }


def analyze_optics_and_motion_3d(frames_bgr, center_idx=0, cam_info=None, light_info=None):
    """Pillar 5 & 6: Measure multi-depth parallax layers, stepped vs full animation cadence,
    evidence-gated Motion Blur Shutter, Depth-of-Field spatial focus separation, radial
    Chromatic Aberration (Dispersion), and Lift/Gamma/Gain color grading.
    """
    bgr = frames_bgr[center_idx]
    h, w = bgr.shape[:2]
    diag = math.hypot(w, h)
    gray_c = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)

    # 1. Color Grading: Lift (P08 shadow RGB), Gamma (P50 midtone RGB), Gain (P96 highlight RGB)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    lum_srgb = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    def sample_percentile_rgb(lo_q, hi_q):
        lo_v, hi_v = np.percentile(lum_srgb, [lo_q, hi_q])
        mask = (lum_srgb >= lo_v) & (lum_srgb <= hi_v)
        if not np.any(mask):
            return [0.0, 0.0, 0.0], "#000000"
        med = np.median(rgb[mask], axis=0)
        return [round(float(v) / 255.0, 4) for v in med], rgb_to_hex(med)

    lift_rgb, lift_hex = sample_percentile_rgb(3, 12)
    gamma_rgb, gamma_hex = sample_percentile_rgb(42, 58)
    gain_rgb, gain_hex = sample_percentile_rgb(90, 98)

    # 2. Evidence-Gated Depth-of-Field (DOF) Detection across a 3x3 spatial grid
    sm_c = cv.GaussianBlur(gray_c, (5, 5), 0.75)
    dx_c = cv.Sobel(sm_c, cv.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
    dy_c = cv.Sobel(sm_c, cv.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
    zone_q30s = []
    for r_idx in range(3):
        for c_idx in range(3):
            z_mask = np.zeros((h, w), dtype=np.uint8)
            y0_z, y1_z = int(r_idx * h / 3), int((r_idx + 1) * h / 3)
            x0_z, x1_z = int(c_idx * w / 3), int((c_idx + 1) * w / 3)
            z_mask[y0_z:y1_z, x0_z:x1_z] = 255
            z_edges = sample(gray_c, dx_c, dy_c, z_mask, 12)
            if len(z_edges) >= 8:
                zone_q30s.append(float(np.percentile(z_edges[:, 2], 30)))

    is_ortho = (cam_info is not None and cam_info.get("projection_type") == "ORTHO")
    if len(zone_q30s) >= 2:
        dof_sharp_min = float(min(zone_q30s))
        dof_soft_max = float(max(zone_q30s))
        dof_gap_px = round(dof_soft_max - dof_sharp_min, 3)
        dof_detected = bool((not is_ortho) and (dof_gap_px >= 0.85) and (dof_sharp_min <= 4.0))
        dof_evidence = "separated" if dof_detected else "deep_focus_or_uniform"
    else:
        dof_sharp_min = 0.0
        dof_soft_max = 0.0
        dof_gap_px = 0.0
        dof_detected = False
        dof_evidence = "insufficient_edges"

    # 3. Radial Chromatic Aberration (Red vs Blue channel radial sub-pixel shift near frame corners)
    r_ch = rgb[:, :, 0]
    b_ch = rgb[:, :, 2]
    cy, cx = h * 0.5, w * 0.5
    dispersion_shifts = []
    for angle_deg in range(15, 360, 30):
        rad = math.radians(angle_deg)
        ux, uy = math.cos(rad), math.sin(rad)
        r_max = min(
            (w - 12 - cx) / abs(ux) if abs(ux) > 1e-3 else 9999.0,
            (h - 12 - cy) / abs(uy) if abs(uy) > 1e-3 else 9999.0,
        )
        if r_max < 120:
            continue
        radii = np.arange(r_max * 0.52, r_max * 0.94, 0.25, dtype=np.float32)
        if len(radii) < 64:
            continue
        xs = (cx + radii * ux).astype(np.float32)
        ys = (cy + radii * uy).astype(np.float32)
        prof_r = cv.remap(r_ch, xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_CUBIC).ravel()
        prof_b = cv.remap(b_ch, xs.reshape(1, -1), ys.reshape(1, -1), cv.INTER_CUBIC).ravel()
        gr = np.gradient(prof_r)
        gb = np.gradient(prof_b)
        if np.std(gr) < 1.2 or np.std(gb) < 1.2:
            continue
        gr_n = (gr - np.mean(gr)) / (np.std(gr) + 1e-6)
        gb_n = (gb - np.mean(gb)) / (np.std(gb) + 1e-6)
        lags = np.arange(-16, 17)
        corrs = [float(np.mean(np.roll(gr_n, lag) * gb_n)) for lag in lags]
        best_idx = int(np.argmax(corrs))
        if corrs[best_idx] > 0.35:
            shift_px = abs(float(lags[best_idx]) * 0.25)
            dispersion_shifts.append(shift_px)

    ca_shift_px = float(np.median(dispersion_shifts)) if len(dispersion_shifts) >= 3 else 0.0
    dispersion_detected = bool(len(dispersion_shifts) >= 3 and ca_shift_px >= 0.45)
    blender_dispersion = (
        round(float(np.clip(ca_shift_px / (0.28 * diag), 0.001, 0.035)), 4)
        if dispersion_detected else 0.0
    )

    # 4. Multi-frame Motion, Stepped Cadence, Depth Parallax Layers, and Evidence-Gated Motion Blur
    motion_info = {
        "window_frames_analyzed": len(frames_bgr),
        "animation_cadence": "single_frame",
        "median_flow_speed_px_per_frame": 0.0,
        "motion_blur_detected": False,
        "estimated_motion_blur_shutter": 0.0,
        "estimated_depth_parallax_layers": 1,
    }

    if len(frames_bgr) >= 2:
        grays = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in frames_bgr]
        frame_diffs = [float(np.mean(np.abs(grays[i + 1].astype(np.float32) - grays[i].astype(np.float32)))) for i in range(len(grays) - 1)]
        duplicate_ratio = float(np.mean([d < 0.45 for d in frame_diffs]))
        if duplicate_ratio > 0.42:
            cadence = "stepped_stop_motion_or_on_2s"
        else:
            cadence = "full_framerate_continuous"

        g0 = grays[center_idx]
        g1 = grays[min(len(grays) - 1, center_idx + 1)] if center_idx + 1 < len(grays) else grays[center_idx - 1]
        mask_all = np.full((h, w), 255, dtype=np.uint8)
        pts0 = pick(g0, mask_all, spacing=18)
        a_pts, b_pts = flow(g0, g1, pts0)
        if len(a_pts) >= 8:
            vecs = b_pts - a_pts
            speeds = np.linalg.norm(vecs, axis=1)
            med_speed = float(np.median(speeds))
            p20_spd, p85_spd = float(np.percentile(speeds, 20)), float(np.percentile(speeds, 85))
            # Multi-depth parallax ratio
            parallax_spread = p85_spd / max(0.35, p20_spd)
            est_depth_layers = 3 if parallax_spread > 2.2 else (2 if parallax_spread > 1.35 else 1)

            # Directional edge smear check: compare edges aligned with motion vs orthogonal to motion
            edges_sample = sample(gray_c, dx_c, dy_c, mask_all, 10)
            motion_blur_detected = False
            shutter_est = 0.0
            dir_smear_gap = 0.0
            if cadence == "full_framerate_continuous" and med_speed >= 0.85 and len(edges_sample) >= 8:
                med_vec = np.median(vecs, axis=0)
                med_vec_norm = float(np.linalg.norm(med_vec))
                w_p30 = float(np.percentile(edges_sample[:, 2], 30))
                w_p75 = float(np.percentile(edges_sample[:, 2], 75))
                if med_vec_norm >= 0.5:
                    u_flow = med_vec / med_vec_norm
                    ex = np.clip(np.rint(edges_sample[:, 0]).astype(int), 0, w - 1)
                    ey = np.clip(np.rint(edges_sample[:, 1]).astype(int), 0, h - 1)
                    gx_e = dx_c[ey, ex]
                    gy_e = dy_c[ey, ex]
                    gnorm = np.hypot(gx_e, gy_e) + 1e-6
                    align = np.abs((gx_e * u_flow[0] + gy_e * u_flow[1]) / gnorm)
                    along_w = edges_sample[align >= 0.65, 2]
                    perp_w = edges_sample[align <= 0.35, 2]
                    if len(along_w) >= 4 and len(perp_w) >= 4:
                        dir_smear_gap = float(np.median(along_w) - np.median(perp_w))
                # Trigger motion blur only when genuine directional smear along the flow vector is observed
                # (keeps crisp stop-motion or motion-blur-disabled CG renders cleanly at False!)
                if (dir_smear_gap >= 0.30) or (dir_smear_gap >= 0.12 and w_p75 >= 3.8 and w_p30 >= 2.4):
                    motion_blur_detected = True
                    shutter_est = float(np.clip((w_p75 - w_p30 + max(0.4, dir_smear_gap)) / max(2.0, med_speed * 0.45), 0.20, 1.0))

            motion_info = {
                "window_frames_analyzed": len(frames_bgr),
                "animation_cadence": cadence,
                "duplicate_frame_ratio": round(duplicate_ratio, 3),
                "median_flow_speed_px_per_frame": round(med_speed, 2),
                "flow_speed_p20_p85_px": [round(p20_spd, 2), round(p85_spd, 2)],
                "parallax_speed_ratio": round(parallax_spread, 3),
                "estimated_depth_parallax_layers": est_depth_layers,
                "directional_smear_gap_px": round(dir_smear_gap, 3),
                "motion_blur_detected": motion_blur_detected,
                "estimated_motion_blur_shutter": round(shutter_est, 2),
                "sample_flow_vectors": [
                    {"x": round(float(a[0]), 1), "y": round(float(a[1]), 1), "dx": round(float(v[0]), 2), "dy": round(float(v[1]), 2)}
                    for a, v in zip(a_pts[:30], vecs[:30])
                ],
            }

    bloom_on = bool(light_info and light_info.get("radial_halo_profile", {}).get("bloom_detected", False))
    soft_glow_mix = (
        light_info.get("recommended_blender_lighting_and_volume", {}).get("recommended_soft_glow_mix", 0.0)
        if light_info else 0.0
    )

    # 5. Universal Global Film Grain / Paper Overlay Texture Detector
    # Uses a smoothed structural gradient mask so real object edges are excluded without
    # censoring the 1-3px film/paper grain itself, and a 5x5 sigma=1.2 high-pass kernel
    # that survives standard H.264/HEVC 4:2:0 video compression.
    gray_n = gray_c / 255.0
    hp = gray_n - cv.GaussianBlur(gray_n, (5, 5), 1.2)
    g_smooth = cv.GaussianBlur(gray_c, (0, 0), 3.0)
    gmag_struct = np.hypot(
        cv.Sobel(g_smooth, cv.CV_32F, 1, 0, ksize=3),
        cv.Sobel(g_smooth, cv.CV_32F, 0, 1, ksize=3),
    )
    flat_mid = (gmag_struct < 14.0) & (gray_c > 16.0) & (gray_c < 242.0)
    if int(np.sum(flat_mid)) >= 250:
        global_grain_rms = float(np.std(hp[flat_mid]))
        q_stds = []
        for ry0, ry1 in ((0, h // 2), (h // 2, h)):
            for cx0, cx1 in ((0, w // 2), (w // 2, w)):
                qm = flat_mid[ry0:ry1, cx0:cx1]
                if int(np.sum(qm)) >= 40:
                    q_stds.append(float(np.std(hp[ry0:ry1, cx0:cx1][qm])))
        quad_unif = round(float(min(q_stds) / max(1e-5, max(q_stds))), 3) if len(q_stds) >= 2 else 1.0
    else:
        global_grain_rms = 0.0
        quad_unif = 0.0

    grain_detected = bool(global_grain_rms >= 0.0070)
    rec_grain_mix = round(float(np.clip(global_grain_rms * 7.5, 0.04, 0.35)), 3) if grain_detected else 0.0

    # 6. Temporal Dynamics: Camera Shake Spectrum, Temporal Exposure/Color Flicker, Texture Boiling & Mixed Cadence
    shake_accel_rms = 0.0
    camera_shake_detected = False
    shake_type = "NONE"
    max_lum_jump_ratio = 0.0
    exposure_flicker_detected = False
    color_temp_drift = 0.0
    window_lum_series = []
    boil_hf_rms = 0.0
    texture_boiling_detected = False
    mixed_cadence_detected = False

    if len(frames_bgr) >= 3:
        # Frame-by-frame linear luminance & color temperature series
        warmth_series = []
        grays_win = []
        for fb in frames_bgr:
            frgb = cv.cvtColor(fb, cv.COLOR_BGR2RGB).astype(np.float32)
            flin = srgb_to_linear(frgb)
            flum = 0.2126 * flin[:, :, 0] + 0.7152 * flin[:, :, 1] + 0.0722 * flin[:, :, 2]
            window_lum_series.append(round(float(np.median(flum)), 4))
            m_rgb = np.mean(frgb.reshape(-1, 3), axis=0)
            warmth_series.append(float((m_rgb[0] - m_rgb[2]) / max(15.0, float(np.mean(m_rgb)))))
            grays_win.append(cv.cvtColor(fb, cv.COLOR_BGR2GRAY))

        lum_arr = np.asarray(window_lum_series, dtype=np.float64)
        base_lum = max(0.02, float(np.median(lum_arr)))
        lum_jumps = np.abs(np.diff(lum_arr)) / base_lum
        max_lum_jump_ratio = round(float(np.max(lum_jumps)), 3) if len(lum_jumps) else 0.0
        exposure_flicker_detected = bool(max_lum_jump_ratio >= 0.18)
        color_temp_drift = round(float(max(warmth_series) - min(warmth_series)), 3) if warmth_series else 0.0

        # Per-pair global median motion for camera shake acceleration spectrum
        step_vecs = []
        mask_full_w = np.full((h, w), 255, dtype=np.uint8)
        for fi in range(len(grays_win) - 1):
            pts_f = pick(grays_win[fi], mask_full_w, spacing=24)
            if len(pts_f) >= 8:
                pa, pb = flow(grays_win[fi], grays_win[fi + 1], pts_f)
                if len(pa) >= 6:
                    dv = pb - pa
                    step_vecs.append(np.median(dv, axis=0))
        if len(step_vecs) >= 3:
            sv = np.asarray(step_vecs, dtype=np.float64)
            acc = np.diff(sv, axis=0)
            acc_mags = np.linalg.norm(acc, axis=1)
            shake_accel_rms = round(float(np.sqrt(np.mean(acc_mags ** 2))), 3)
            # Check sign reversals in x or y velocity (oscillatory jitter vs smooth monotonic pan)
            reversals = int(np.sum((sv[:-1, 0] * sv[1:, 0]) < -0.25) + np.sum((sv[:-1, 1] * sv[1:, 1]) < -0.25))
            if shake_accel_rms >= 2.8 and reversals >= 1:
                camera_shake_detected = True
                shake_type = "IMPACT_HIGH_FREQ_SHAKE"
            elif shake_accel_rms >= 0.95 and reversals >= 2:
                camera_shake_detected = True
                shake_type = "HANDHELD_LOW_FREQ_DRIFT"

        # Texture / Line Boiling on macro-stationary regions between consecutive non-duplicate frames
        nxt_i = min(len(grays_win) - 1, center_idx + 1) if center_idx + 1 < len(grays_win) else max(0, center_idx - 1)
        g0_f = grays_win[center_idx].astype(np.float32) / 255.0
        g1_f = grays_win[nxt_i].astype(np.float32) / 255.0
        macro_diff = np.abs(cv.GaussianBlur(g0_f, (9, 9), 2.5) - cv.GaussianBlur(g1_f, (9, 9), 2.5))
        hp0 = g0_f - cv.GaussianBlur(g0_f, (5, 5), 1.2)
        hp1 = g1_f - cv.GaussianBlur(g1_f, (5, 5), 1.2)
        static_mid = (macro_diff < 0.012) & (g0_f > 0.08) & (g0_f < 0.94)
        if int(np.sum(static_mid)) >= 400:
            boil_hf_rms = round(float(np.std((hp1 - hp0)[static_mid])), 4)
            # Distinguish real stop-motion paper/line boil from subtle H.264 P-frame noise
            texture_boiling_detected = bool(boil_hf_rms >= 0.0155)

        # Mixed cadence: background moves smoothly on 1s while a distinct sub-region freezes/steps on 2s
        if len(step_vecs) >= 3:
            bg_moving_frames = int(np.sum(np.linalg.norm(np.asarray(step_vecs), axis=1) >= 0.8))
            if bg_moving_frames >= len(step_vecs) - 1 and motion_info.get("duplicate_frame_ratio", 0.0) < 0.2:
                # Check if a localized interior block has alternate-frame freeze while global scene moves
                g_m1 = grays_win[max(0, center_idx - 1)].astype(np.float32)
                g_0 = grays_win[center_idx].astype(np.float32)
                g_p1 = grays_win[min(len(grays_win) - 1, center_idx + 1)].astype(np.float32)
                d_prev = np.abs(g_0 - g_m1)
                d_next = np.abs(g_p1 - g_0)
                freeze_then_jump = float(np.mean((d_prev < 2.0) & (d_next > 14.0)))
                jump_then_freeze = float(np.mean((d_prev > 14.0) & (d_next < 2.0)))
                mixed_cadence_detected = bool(max(freeze_then_jump, jump_then_freeze) >= 0.045)

    # 7. Advanced Optical & Stylized Post-FX Signatures: Film Halation, Anamorphic Streak, Scanline/Halftone
    lin_c = srgb_to_linear(rgb)
    lum_c = 0.2126 * lin_c[:, :, 0] + 0.7152 * lin_c[:, :, 1] + 0.0722 * lin_c[:, :, 2]
    p995_c = float(np.percentile(lum_c, 99.5))
    p99_c = float(np.percentile(lum_c, 99.0))
    p85_c = float(np.percentile(lum_c, 85.0))
    has_cores = bool(light_info and light_info.get("radial_halo_profile", {}).get("has_localized_light_cores", False))

    red_minus_gb_halo = 0.0
    halation_detected = False
    hv_halo_ratio = 1.0
    anamorphic_streak_detected = False

    if (has_cores or p995_c >= 0.55) and max(p99_c, p995_c) >= 0.35:
        c_thr = max(0.35, min(0.78, p85_c + 0.35 * max(0.0, p995_c - p85_c)))
        core_bin = (lum_c >= c_thr).astype(np.uint8)
        if 20 <= int(np.sum(core_bin)) <= int(0.18 * h * w):
            dist_c = cv.distanceTransform(1 - core_bin, cv.DIST_L2, 5)
            ring_2_9 = (dist_c >= 2.0) & (dist_c <= 9.5)
            bg_ref = (dist_c >= 24.0) & (dist_c <= 36.0)
            if int(np.sum(ring_2_9)) >= 60 and int(np.sum(bg_ref)) >= 60:
                # Only count halation around neutral/bright cores (exclude inherently red neon tubes)
                core_r_dom = float(np.median(lin_c[:, :, 0][core_bin > 0]) - 0.5 * (
                    np.median(lin_c[:, :, 1][core_bin > 0]) + np.median(lin_c[:, :, 2][core_bin > 0])
                ))
                r_ex = float(np.median(lin_c[:, :, 0][ring_2_9]) - np.median(lin_c[:, :, 0][bg_ref]))
                g_ex = float(np.median(lin_c[:, :, 1][ring_2_9]) - np.median(lin_c[:, :, 1][bg_ref]))
                b_ex = float(np.median(lin_c[:, :, 2][ring_2_9]) - np.median(lin_c[:, :, 2][bg_ref]))
                red_minus_gb_halo = round(r_ex - 0.5 * (g_ex + b_ex), 4)
                # Film halation occurs around genuine bright highlight cores (p995 >= 0.72) with neutral core
                # and red-shifted rim, NOT on scenes whose surfaces are broadly painted red (e.g. red/white stripes)
                bg_red_dom = float(np.median(lin_c[:, :, 0][bg_ref]) - 0.5 * (
                    np.median(lin_c[:, :, 1][bg_ref]) + np.median(lin_c[:, :, 2][bg_ref])
                ))
                global_red_albedo_frac = float(np.mean((lin_c[:, :, 0] - 0.5 * (lin_c[:, :, 1] + lin_c[:, :, 2])) > 0.08))
                halation_detected = bool(
                    max(p99_c, p995_c) >= 0.72
                    and red_minus_gb_halo >= 0.036
                    and core_r_dom < 0.20
                    and bg_red_dom < 0.035
                    and global_red_albedo_frac < 0.15
                    and r_ex >= 0.045
                )

            # Anamorphic horizontal streak vs vertical glow around bright cores
            k_h = cv.getStructuringElement(cv.MORPH_RECT, (41, 3))
            k_v = cv.getStructuringElement(cv.MORPH_RECT, (3, 41))
            k_c = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
            inner_ex = cv.dilate(core_bin, k_c) > 0
            h_wing = (cv.dilate(core_bin, k_h) > 0) & (~inner_ex)
            v_wing = (cv.dilate(core_bin, k_v) > 0) & (~inner_ex)
            if int(np.sum(h_wing)) >= 80 and int(np.sum(v_wing)) >= 80:
                bg_floor = float(np.percentile(lum_c, 25))
                h_ex = max(0.0, float(np.percentile(lum_c[h_wing], 92)) - bg_floor)
                v_ex = max(0.0, float(np.percentile(lum_c[v_wing], 92)) - bg_floor)
                hv_halo_ratio = round((h_ex + 1e-4) / (v_ex + 1e-4), 2)
                anamorphic_streak_detected = bool(h_ex >= 0.045 and hv_halo_ratio >= 2.25)

    # Stylized periodic overlay detector (CRT scanlines / comic halftone dots via 2D FFT high-freq peak)
    # Check spatial support before calling a local periodic surface a full-frame overlay.
    ps = min(192, min(h,w)//2)
    periodic_regions = []
    if ps >= 32:
        for top in np.linspace(0,h-ps,3,dtype=int):
            for left in np.linspace(0,w-ps,3,dtype=int):
                patch_fft = gray_c[top:top+ps,left:left+ps]
                if patch_fft.std() < 6: continue
                win_2d = np.outer(np.hanning(ps), np.hanning(ps)).astype(np.float32)
                fmag = np.abs(np.fft.fftshift(np.fft.fft2((patch_fft - np.mean(patch_fft)) * win_2d)))
                mid_f = ps // 2
                fy, fx = np.ogrid[:ps, :ps]
                frad = np.hypot(fx - mid_f, fy - mid_f)
                # Scanline check: sharp periodic spike strictly on vertical frequency axis (|fx-mid|<=2, high fy)
                scan_band = (np.abs(fx - mid_f) <= 2) & (np.abs(fy - mid_f) >= ps // 6) & (np.abs(fy - mid_f) <= int(ps * 0.48))
                hf_annulus = (frad >= ps // 6) & (frad <= int(ps * 0.48))
                hf_med = max(1e-3, float(np.median(fmag[hf_annulus])))
                scan_peak_ratio = round(float(np.max(fmag[scan_band])) / hf_med, 1) if np.any(scan_band) else 0.0
                # Halftone dot grid check: off-axis diagonal periodic spike (|fx-mid|>8 and |fy-mid|>8)
                diag_band = hf_annulus & (np.abs(fx - mid_f) >= 10) & (np.abs(fy - mid_f) >= 10)
                diag_peak_ratio = round(float(np.max(fmag[diag_band])) / hf_med, 1) if np.any(diag_band) else 0.0
                periodic_regions.append(dict(x=int(left),y=int(top),size=ps,scan=scan_peak_ratio,diagonal=diag_peak_ratio))
    scan_peak_ratio=max((p['scan'] for p in periodic_regions),default=0.)
    diag_peak_ratio=max((p['diagonal'] for p in periodic_regions),default=0.)
    scans=sum(p['scan']>=52 for p in periodic_regions)
    dots=sum(p['diagonal']>=75 for p in periodic_regions)
    stylized_overlay_type='CRT_SCANLINES' if scans>=4 else 'HALFTONE_DOTS' if dots>=4 else 'NONE'
    stylized_overlay_detected=stylized_overlay_type!='NONE'

    return {
        "color_grading_lift_gamma_gain": {
            "lift_shadow_srgb": lift_rgb,
            "lift_shadow_hex": lift_hex,
            "gamma_midtone_srgb": gamma_rgb,
            "gamma_midtone_hex": gamma_hex,
            "gain_highlight_srgb": gain_rgb,
            "gain_highlight_hex": gain_hex,
        },
        "depth_of_field_analysis": {
            "dof_detected": dof_detected,
            "dof_evidence": dof_evidence,
            "sharpest_zone_q30_px": round(dof_sharp_min, 3),
            "softest_zone_q30_px": round(dof_soft_max, 3),
            "cross_zone_blur_gap_px": dof_gap_px,
            "sampled_zones_count": len(zone_q30s),
        },
        "film_and_paper_grain": {
            "film_or_paper_grain_detected": grain_detected,
            "global_grain_rms": round(global_grain_rms, 4),
            "quadrant_uniformity": quad_unif,
            "recommended_compositor_grain_mix": rec_grain_mix,
        },
        "lens_optics_and_compositor": {
            "enable_bloom_or_soft_glow": bloom_on,
            "recommended_glare_type": "Fog Glow" if bloom_on else "NONE",
            "recommended_soft_glow_screen_factor": soft_glow_mix,
            "dispersion_detected": dispersion_detected,
            "radial_chromatic_aberration_px": round(ca_shift_px, 3),
            "recommended_blender_lens_dispersion": blender_dispersion,
            "recommended_blender_lens_distortion": round(blender_dispersion * 0.65, 4) if dispersion_detected else 0.0,
        },
        "advanced_optics_profile": {
            "halation_detected": halation_detected,
            "red_minus_gb_halo_excess": red_minus_gb_halo,
            "anamorphic_streak_detected": anamorphic_streak_detected,
            "horiz_to_vert_halo_ratio": hv_halo_ratio,
            "stylized_overlay_detected": stylized_overlay_detected,
            "stylized_overlay_type": stylized_overlay_type,
            "periodic_region_samples": periodic_regions,
            "fft_scanline_and_halftone_peak_ratios": [scan_peak_ratio, diag_peak_ratio],
        },
        "temporal_dynamics_profile": {
            "camera_shake_detected": camera_shake_detected,
            "shake_type": shake_type,
            "shake_accel_rms_px": shake_accel_rms,
            "texture_boiling_detected": texture_boiling_detected,
            "stationary_hf_boil_rms": boil_hf_rms,
            "mixed_cadence_detected": mixed_cadence_detected,
            "temporal_exposure_flicker_detected": exposure_flicker_detected,
            "max_frame_lum_jump_ratio": max_lum_jump_ratio,
            "window_color_temp_drift": color_temp_drift,
            "window_median_lum_series": window_lum_series,
        },
        "motion_and_parallax_3d": motion_info,
    }


def build_lookdev_feature_gates(cam_info, mat_info, light_info, optics_info):
    """Assemble explicit, evidence-gated boolean activation flags for Blender Look-Dev & Compositor.
    Prevents both omitting effects present in the source AND fabricating effects absent in the source.
    """
    rad = light_info["radial_halo_profile"]
    rec_lv = light_info["recommended_blender_lighting_and_volume"]
    vig = light_info.get("vignette_profile", {})
    shd = light_info.get("shadow_and_key_light_hardness", {})
    lcr = light_info.get("lighting_color_and_rig_profile", {})
    haze = light_info.get("atmospheric_depth_haze", {})
    gref = light_info.get("ground_reflection_profile", {})
    esp = light_info.get("emissive_spill_profile", {})
    rcon = light_info.get("regional_tonal_contrast_profile", {})
    tex = mat_info.get("surface_texture_profile", {})
    ink = mat_info.get("stylized_ink_outline", {})
    dof = optics_info["depth_of_field_analysis"]
    grn = optics_info.get("film_and_paper_grain", {})
    opt = optics_info["lens_optics_and_compositor"]
    adv = optics_info.get("advanced_optics_profile", {})
    tmp = optics_info.get("temporal_dynamics_profile", {})
    mot = optics_info["motion_and_parallax_3d"]

    return {
        "enable_volume_fog": bool(rad.get("volume_fog_detected", False)),
        "enable_daylight_depth_haze": bool(haze.get("daylight_atmospheric_haze_detected", False)),
        "enable_bloom_or_soft_glow": bool(rad.get("bloom_detected", False)),
        "enable_dof": bool(dof.get("dof_detected", False)),
        "enable_motion_blur": bool(mot.get("motion_blur_detected", False)),
        "enable_lens_dispersion": bool(opt.get("dispersion_detected", False)),
        "enable_emissive_materials": bool(rad.get("emissive_core_detected", False)),
        "enable_emissive_indirect_spill": bool(esp.get("emissive_indirect_spill_detected", False)),
        "enable_surface_grunge_bump": bool(tex.get("surface_grunge_detected", False)),
        "enable_ink_outline": bool(ink.get("ink_outline_detected", False)),
        "enable_vignette": bool(vig.get("vignette_detected", False)),
        "enable_film_or_paper_grain": bool(grn.get("film_or_paper_grain_detected", False)),
        "enable_split_tone_lighting": bool(lcr.get("split_tone_detected", False)),
        "enable_rim_backlight": bool(lcr.get("rim_light_detected", False)),
        "enable_ground_reflection": bool(gref.get("ground_reflection_detected", False)),
        "enable_film_halation": bool(adv.get("halation_detected", False)),
        "enable_anamorphic_streak": bool(adv.get("anamorphic_streak_detected", False)),
        "enable_texture_boiling": bool(tmp.get("texture_boiling_detected", False)),
        "enable_camera_shake": bool(tmp.get("camera_shake_detected", False)),
        "shading_paradigm": mat_info["shading_paradigm"],
        "surface_finish_class": tex.get("surface_finish_class", "CLEAN_SMOOTH"),
        "key_light_hardness": shd.get("estimated_key_light_hardness", "MEDIUM_SOFT_AREA"),
        "key_elevation": lcr.get("estimated_key_elevation", "SIDE_KEY"),
        "projection_type": cam_info["projection_type"],
        "evidence_summary": {
            "near_bloom_excess": rad.get("near_bloom_excess", 0.0),
            "far_volumetric_excess": rad.get("far_volumetric_excess", 0.0),
            "emissive_spill_chromatic_energy": esp.get("spill_ring_chromatic_energy", 0.0),
            "non_core_crushed_black_ratio": esp.get("non_core_crushed_black_ratio", 0.0),
            "oriented_streak_coherence": gref.get("oriented_streak_coherence", 0.0),
            "active_quadrant_median_contrast": rcon.get("active_quadrant_median_contrast", 1.0),
            "principled_volume_density": rec_lv.get("principled_volume_density", 0.0),
            "daylight_haze_floor_lift": haze.get("distant_black_floor_lift", 0.0),
            "ground_reflection_type": gref.get("reflection_type", "NONE"),
            "key_fill_warmth_delta": lcr.get("key_fill_warmth_delta", 0.0),
            "rim_boundary_excess_lum": lcr.get("rim_boundary_excess_lum", 0.0),
            "dof_blur_gap_px": dof.get("cross_zone_blur_gap_px", 0.0),
            "motion_speed_px_per_frame": mot.get("median_flow_speed_px_per_frame", 0.0),
            "motion_blur_shutter": mot.get("estimated_motion_blur_shutter", 0.0),
            "camera_shake_accel_rms_px": tmp.get("shake_accel_rms_px", 0.0),
            "texture_boil_hf_rms": tmp.get("stationary_hf_boil_rms", 0.0),
            "chromatic_aberration_px": opt.get("radial_chromatic_aberration_px", 0.0),
            "halation_red_excess": adv.get("red_minus_gb_halo_excess", 0.0),
            "anamorphic_hv_ratio": adv.get("horiz_to_vert_halo_ratio", 1.0),
            "surface_mid_texture_rms": tex.get("mid_texture_rms_5_16px", 0.0),
            "surface_macro_grunge_rms": tex.get("macro_grunge_rms_16_48px", 0.0),
            "ink_outline_width_px": ink.get("estimated_outline_width_px", 0.0),
            "shadow_penumbra_width_px": shd.get("shadow_penumbra_width_px", 0.0),
            "vignette_corner_to_center_ratio": vig.get("corner_to_center_ratio", 1.0),
            "global_grain_rms": grn.get("global_grain_rms", 0.0),
        },
    }


def assess_reconstructability_signals(cam_info, mat_info, light_info, optics_info, paradigm_info):
    """Compute objective geometric and temporal reconstructability signals to assist pre-flight
    capability boundary triage (references/capabilities.md). Never guesses semantic categories
    like live-action humans or hand-drawn anime — always flags requires_agent_visual_boundary_check=True.
    """
    vp_dict = cam_info.get("vanishing_points_px", {})
    vp_count = sum(1 for v in vp_dict.values() if v is not None)
    lc = cam_info.get("line_clusters", {})
    total_lines = int(lc.get("vertical_count", 0)) + int(lc.get("cluster_a_count", 0)) + int(lc.get("cluster_b_count", 0))
    focal_method = cam_info.get("focal_estimation_method", "default_35mm_prior")
    proj_type = cam_info.get("projection_type", "PERSP")

    detected_paradigm = paradigm_info.get("detected_paradigm", "3D_BLENDER_SCENE")
    p95 = paradigm_info.get("single_plane_homography_p95_px")
    inlier_ratio = float(paradigm_info.get("single_plane_inlier_ratio", 0.0) or 0.0)
    top2_cov = float(paradigm_info.get("top2_histogram_bin_coverage", 0.0) or 0.0)

    mot = optics_info.get("motion_and_parallax_3d", {})
    n_win = int(mot.get("window_frames_analyzed", 1))
    med_speed = float(mot.get("median_flow_speed_px_per_frame", 0.0) or 0.0)
    flow_samples = len(mot.get("sample_flow_vectors", []))

    # 2. Temporal stability hint (evaluated before finalizing geometric confidence)
    if n_win < 2:
        temp_hint = "single_frame_only"
    elif flow_samples < 8:
        temp_hint = "tracking_failed_or_nonrigid"
    elif med_speed > 35.0:
        temp_hint = "extreme_fast_motion_or_cut"
    elif p95 is not None and inlier_ratio < 0.20 and vp_count == 0:
        temp_hint = "low_rigidity_or_heavy_parallax"
    else:
        temp_hint = "stable"

    # 1. Objective geometric structure confidence
    if temp_hint == "tracking_failed_or_nonrigid" and focal_method != "orthogonal_vanishing_points":
        geom_conf = "low"
    elif detected_paradigm == "2D_PLANAR_UI_MOTION" and (
        (p95 is not None and p95 <= 2.75 and inlier_ratio >= 0.75) or (top2_cov >= 0.68 and total_lines >= 6)
    ):
        geom_conf = "high"
    elif (
        (focal_method == "orthogonal_vanishing_points" and vp_count >= 2 and total_lines >= 30)
        or (proj_type == "ORTHO" and total_lines >= 20)
        or (vp_count >= 2 and total_lines >= 40 and temp_hint == "stable")
    ):
        geom_conf = "high"
    elif (
        (vp_count >= 1 and total_lines >= 10)
        or (p95 is not None and inlier_ratio >= 0.35 and total_lines >= 8)
        or (top2_cov >= 0.65 and total_lines >= 6)
    ):
        geom_conf = "medium"
    else:
        geom_conf = "low"

    # 3. Suggested downstream engine routing
    if geom_conf == "low" or temp_hint in ("tracking_failed_or_nonrigid", "extreme_fast_motion_or_cut"):
        suggested_engine = "Visual_Review_Required"
    elif detected_paradigm == "2D_PLANAR_UI_MOTION":
        suggested_engine = "Remotion_AE_2D"
    elif detected_paradigm == "HYBRID_2D_OVER_3D":
        suggested_engine = "Hybrid_2D_over_3D"
    else:
        suggested_engine = "Blender_3D"

    # 4. Objective automated risk warnings
    warnings = []
    if geom_conf == "low":
        warnings.append(
            "Weak geometric structure (no reliable vanishing points or stable planar homography): "
            "may be organic live-action, fluid/VFX, noisy, or non-rigid footage; verify against L3 boundaries."
        )
    if focal_method == "default_35mm_prior" and detected_paradigm != "2D_PLANAR_UI_MOTION":
        warnings.append(
            "Insufficient orthogonal vanishing points for exact focal solve; fell back to 35mm prior."
        )
    if temp_hint == "single_frame_only":
        warnings.append("Single-frame analysis only; depth parallax and motion blur cannot be measured.")
    elif temp_hint == "tracking_failed_or_nonrigid":
        warnings.append(
            "Inter-frame optical flow failed (<8 stable tracks); likely unstructured noise, blank frame, "
            "extreme non-rigid motion, or hard cut inside analysis window."
        )
    elif temp_hint == "extreme_fast_motion_or_cut":
        warnings.append(
            f"Extreme inter-frame motion ({med_speed:.1f} px/frame); verify no hard cut or heavy whip-pan inside window."
        )
    elif temp_hint == "low_rigidity_or_heavy_parallax":
        warnings.append(
            f"Low planar rigidity (inlier_ratio={inlier_ratio:.2f}) without vanishing-point support; "
            "confirm whether shot contains heavy organic/non-rigid deformation."
        )

    return {
        "geometric_structure_confidence": geom_conf,
        "temporal_stability_hint": temp_hint,
        "suggested_engine": suggested_engine,
        "metrics_used": {
            "vanishing_points_count": vp_count,
            "structural_line_segments_count": total_lines,
            "focal_estimation_method": focal_method,
            "single_plane_inlier_ratio": round(inlier_ratio, 3),
            "median_flow_speed_px_per_frame": round(med_speed, 2),
        },
        "automated_warnings": warnings,
        "requires_agent_visual_boundary_check": True,
    }


def save_diagnostics_3d(out_dir, bgr, cam_info, mat_info, light_info, optics_info):
    """Render a 4-panel visual diagnostic chart summarizing the 3D scene decomposition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    rgb_disp = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).copy()

    # Overlay detected line segments, 3D planar object corners, and optical flow vectors on Panel 0
    for seg in cam_info["line_clusters"]["sample_segments"]:
        cv.line(
            rgb_disp,
            (int(seg["x1"]), int(seg["y1"])),
            (int(seg["x2"]), int(seg["y2"])),
            (60, 220, 255), 2, cv.LINE_AA
        )
    if "planar_object_3d" in cam_info:
        c2d = cam_info["planar_object_3d"]["corners_2d_px"]
        poly = np.array([c2d["TL"], c2d["TR"], c2d["BR"], c2d["BL"]], dtype=np.int32)
        cv.polylines(rgb_disp, [poly], True, (50, 255, 110), 3, cv.LINE_AA)

    for fv in optics_info["motion_and_parallax_3d"].get("sample_flow_vectors", []):
        p0 = (int(fv["x"]), int(fv["y"]))
        p1 = (int(fv["x"] + fv["dx"] * 3.0), int(fv["y"] + fv["dy"] * 3.0))
        cv.arrowedLine(rgb_disp, p0, p1, (255, 190, 50), 2, cv.LINE_AA, tipLength=0.3)

    axes[0, 0].imshow(rgb_disp)
    axes[0, 0].set_title(
        f"1. Camera & 3D Geometry: {cam_info['projection_type']} | "
        f"f={cam_info['estimated_focal_mm']}mm | Pitch={cam_info['camera_pitch_deg']}°"
    )
    axes[0, 0].axis("off")

    # Panel 1: Radial Light / Halo / Volumetric Scattering Profile
    curve = light_info["radial_halo_profile"]["radial_lum_curve_0_to_35px"]
    rs = np.arange(len(curve))
    axes[0, 1].plot(rs, curve, color="#FF4520", lw=2.2, label="Measured Linear Luminance L(r)")
    axes[0, 1].axhline(light_info["radial_halo_profile"]["ambient_floor_linear_lum"], color="#8899AA", ls="--", label="Ambient Shadow Floor")
    axes[0, 1].axvspan(0, 2.5, color="#FFD0B0", alpha=0.3, label="1. Core (AgX Rolloff)")
    axes[0, 1].axvspan(2.5, 10.0, color="#FFA040", alpha=0.2, label="2. Near Bloom / Black Pro-Mist")
    axes[0, 1].axvspan(10.0, 35.0, color="#806040", alpha=0.15, label=f"3. Volume Fog (λ={light_info['radial_halo_profile']['halo_decay_length_lambda_px']}px)")
    axes[0, 1].set_title(
        f"2. Light & Volumetric Profile (Anisotropy g={light_info['recommended_blender_lighting_and_volume']['principled_volume_anisotropy']}, "
        f"Density={light_info['recommended_blender_lighting_and_volume']['principled_volume_density']})"
    )
    axes[0, 1].set_xlabel("Distance from Light/Tube Ridge (px)")
    axes[0, 1].set_ylabel("Linear Luminance")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc="upper right", fontsize=8.5)

    # Panel 2: Lift / Gamma / Gain & Material BSDF Summary
    lgg = optics_info["color_grading_lift_gamma_gain"]
    swatches = np.zeros((100, 300, 3), dtype=np.float32)
    swatches[:, :100] = lgg["lift_shadow_srgb"]
    swatches[:, 100:200] = lgg["gamma_midtone_srgb"]
    swatches[:, 200:] = lgg["gain_highlight_srgb"]
    axes[1, 0].imshow(np.clip(swatches, 0.0, 1.0))
    axes[1, 0].set_xticks([50, 150, 250])
    axes[1, 0].set_xticklabels([
        f"Lift (Shadows)\n{lgg['lift_shadow_hex']}",
        f"Gamma (Midtones)\n{lgg['gamma_midtone_hex']}",
        f"Gain (Highlights)\n{lgg['gain_highlight_hex']}",
    ])
    axes[1, 0].set_yticks([])
    axes[1, 0].set_title(
        f"3. Color Grading & Material ({mat_info['shading_paradigm']}, "
        f"Roughness≈{mat_info['estimated_principled_bsdf']['roughness_median']})"
    )

    # Panel 3: Recommended Blender 5.x Parameter Sheet (Evidence-Gated)
    axes[1, 1].axis("off")
    rec_vol = light_info["recommended_blender_lighting_and_volume"]
    rec_opt = optics_info["lens_optics_and_compositor"]
    rec_mot = optics_info["motion_and_parallax_3d"]
    rec_dof = optics_info.get("depth_of_field_analysis", {})
    rec_tex = mat_info.get("surface_texture_profile", {})
    rec_ink = mat_info.get("stylized_ink_outline", {})
    rec_shd = light_info.get("shadow_and_key_light_hardness", {})
    rec_vig = light_info.get("vignette_profile", {})
    rec_grn = optics_info.get("film_and_paper_grain", {})
    rec_lcr = light_info.get("lighting_color_and_rig_profile", {})
    rec_haze = light_info.get("atmospheric_depth_haze", {})
    rec_gref = light_info.get("ground_reflection_profile", {})
    rec_adv = optics_info.get("advanced_optics_profile", {})
    rec_tmp = optics_info.get("temporal_dynamics_profile", {})
    vol_status = (
        f"ON (Density={rec_vol['principled_volume_density']}, g={rec_vol['principled_volume_anisotropy']})"
        if rec_vol.get("enable_volume_fog") else "OFF (clean / no volume scatter)"
    )
    bloom_status = (
        f"ON ({rec_opt['recommended_glare_type']}, SoftMix={rec_opt['recommended_soft_glow_screen_factor']})"
        if rec_opt.get("enable_bloom_or_soft_glow") else "OFF (crisp edges)"
    )
    dof_status = (
        f"ON (gap={rec_dof.get('cross_zone_blur_gap_px', 0.0)}px)"
        if rec_dof.get("dof_detected") else "OFF (deep focus / uniform)"
    )
    mb_status = (
        f"ON (Shutter={rec_mot['estimated_motion_blur_shutter']})"
        if rec_mot.get("motion_blur_detected") else "OFF (static / stepped / crisp)"
    )
    tex_status = (
        f"{rec_tex.get('surface_finish_class', 'CLEAN')} "
        f"(mid={rec_tex.get('mid_texture_rms_5_16px', 0.0)}, macro={rec_tex.get('macro_grunge_rms_16_48px', 0.0)}, "
        f"bump={rec_tex.get('recommended_bump_strength', 0.0)})"
    )
    ink_status = (
        f"ON (~{rec_ink.get('estimated_outline_width_px', 0.0)}px contour)"
        if rec_ink.get("ink_outline_detected") else "OFF"
    )
    vig_status = (
        f"ON (corner/center={rec_vig.get('corner_to_center_ratio', 1.0)}, str={rec_vig.get('recommended_compositor_vignette_strength', 0.0)})"
        if rec_vig.get("vignette_detected") else "OFF"
    )
    grn_status = (
        f"ON (rms={rec_grn.get('global_grain_rms', 0.0)}, mix={rec_grn.get('recommended_compositor_grain_mix', 0.0)})"
        if rec_grn.get("film_or_paper_grain_detected") else "OFF"
    )
    rig_extras = [
        f"SplitTone={'ON' if rec_lcr.get('split_tone_detected') else 'OFF'}({rec_lcr.get('key_lit_hex','#')}/{rec_lcr.get('fill_shadow_hex','#')})",
        f"Rim={'ON' if rec_lcr.get('rim_light_detected') else 'OFF'}",
        f"DayHaze={'ON' if rec_haze.get('daylight_atmospheric_haze_detected') else 'OFF'}",
        f"GroundRefl={rec_gref.get('reflection_type', 'NONE')}",
    ]
    adv_extras = [
        f"Halation={'ON' if rec_adv.get('halation_detected') else 'OFF'}",
        f"Streak={'ON' if rec_adv.get('anamorphic_streak_detected') else 'OFF'}",
        f"Shake={rec_tmp.get('shake_type', 'NONE')}",
        f"Boil={'ON' if rec_tmp.get('texture_boiling_detected') else 'OFF'}",
    ]
    lines_txt = [
        "4. Evidence-Gated Blender 5.x Reconstruction Parameters",
        "-" * 56,
        f"• Projection & Lens : {cam_info['projection_type']} | Focal = {cam_info['estimated_focal_mm']} mm (36mm)",
        f"• Camera Tilt / Roll: Pitch = {cam_info['camera_pitch_deg']}° | Roll = {cam_info['camera_roll_deg']}°",
        f"• Shading & Rough   : {mat_info['shading_paradigm']} | Rough={mat_info['estimated_principled_bsdf']['roughness_median']}",
        f"• Surface Texture   : {tex_status}",
        f"• Stylized Outline  : {ink_status}",
        f"• Key Light / Shadow: {rec_shd.get('estimated_key_light_hardness', 'MEDIUM')} ({rec_lcr.get('estimated_key_elevation','SIDE')}, penumbra={rec_shd.get('shadow_penumbra_width_px', 0.0)}px)",
        f"• Light/Env Gates   : {' | '.join(rig_extras)}",
        f"• Principled Volume : {vol_status}",
        f"• Bloom / Soft Glow : {bloom_status}",
        f"• Depth of Field    : {dof_status}",
        f"• Motion Blur       : {mb_status} | {rec_mot['animation_cadence']}",
        f"• Vignette / Grain  : Vig={vig_status} | Grain={grn_status}",
        f"• Adv FX / Dynamics : {' | '.join(adv_extras)}",
        f"• Depth Parallax    : {rec_mot.get('estimated_depth_parallax_layers', 1)} layer(s) | "
        f"Speed = {rec_mot['median_flow_speed_px_per_frame']} px/f",
    ]
    if "planar_object_3d" in cam_info:
        p3d = cam_info["planar_object_3d"]
        lines_txt.extend([
            f"• Planar Object 3D  : D_L/D_R = {p3d['foreshortening_depth_ratio_L_over_R']} | "
            f"Focus Dist = {p3d['explicit_focus_distance_m']} m",
            f"• Clean Plate Margin: +{light_info['radial_halo_profile']['recommended_clean_plate_pad_px']} px",
        ])
    axes[1, 1].text(0.02, 0.96, "\n".join(lines_txt), va="top", ha="left", fontsize=8.0, family="monospace")

    fig.savefig(Path(out_dir) / "diagnostics_3d.png", dpi=145)
    plt.close(fig)


def decompose_3d_objects_and_bom(out_dir, frames_bgr, center_idx, cam_info, mat_info, light_info, objects_plan=None):
    """Discover or measure all distinct 3D objects in the scene, compute their 3D camera-space position,
    metric dimensions (W_m, H_m, D_m), motion vector, material profile, Blender modeling recipe, and
    group them into Multi-Agent Work Packages (assets_bom.json + assets_bom.md + objects_3d_map.png + crops).
    """
    out_path = Path(out_dir)
    crops_dir = out_path / "objects"
    crops_dir.mkdir(parents=True, exist_ok=True)

    bgr = frames_bgr[center_idx]
    h, w = bgr.shape[:2]
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    focal_mm = float(cam_info.get("estimated_focal_mm") or 35.0)
    fov_scale = 36.0 / max(1e-3, focal_mm)

    # Compute sparse optical flow and global background homography to detect independently moving 3D objects
    flow_pts0 = np.empty((0, 2), dtype=np.float32)
    flow_vecs = np.empty((0, 2), dtype=np.float32)
    flow_res = np.empty((0,), dtype=np.float32)
    H_bg = np.eye(3, dtype=np.float64)

    if len(frames_bgr) >= 2:
        g0 = gray
        g1_idx = min(len(frames_bgr) - 1, center_idx + 1) if center_idx + 1 < len(frames_bgr) else max(0, center_idx - 1)
        g1 = cv.cvtColor(frames_bgr[g1_idx], cv.COLOR_BGR2GRAY)
        mask_all = np.full((h, w), 255, dtype=np.uint8)
        a_pts, b_pts = flow(g0, g1, pick(g0, mask_all, spacing=14))
        if len(a_pts) >= 8:
            flow_pts0 = a_pts
            flow_vecs = b_pts - a_pts
            fit = solve_pair(a_pts, b_pts, strict=False)
            if fit is not None and fit.get("H") is not None:
                H_bg = np.asarray(fit["H"], dtype=np.float64)
                hom = np.column_stack([a_pts, np.ones(len(a_pts))]) @ H_bg.T
                pred = hom[:, :2] / np.maximum(1e-6, hom[:, 2:3])
                flow_res = np.linalg.norm(b_pts - pred, axis=1)
            else:
                med_v = np.median(flow_vecs, axis=0)
                flow_res = np.linalg.norm(flow_vecs - med_v, axis=1)

    raw_candidates = []

    if objects_plan and isinstance(objects_plan, list) and len(objects_plan) > 0:
        # Mode B: Plan-specified or agent-refined semantic 3D objects
        for item in objects_plan:
            bbox = item.get("bbox")
            if bbox is None and item.get("corners"):
                cs = np.asarray(item["corners"], dtype=np.float32)
                bbox = [float(np.min(cs[:, 0])), float(np.min(cs[:, 1])), float(np.max(cs[:, 0])), float(np.max(cs[:, 1]))]
            if not bbox:
                continue
            # Support normalized [0..1] or pixel coordinates
            if max(bbox) <= 1.05:
                x0, y0, x1, y1 = int(round(bbox[0] * w)), int(round(bbox[1] * h)), int(round(bbox[2] * w)), int(round(bbox[3] * h))
            else:
                x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            x0, x1 = int(np.clip(x0, 0, w - 2)), int(np.clip(x1, x0 + 8, w))
            y0, y1 = int(np.clip(y0, 0, h - 2)), int(np.clip(y1, y0 + 8, h))
            raw_candidates.append({
                "id": item.get("id") or f"obj_{len(raw_candidates)+1:02d}",
                "name": item.get("name") or f"3D_Object_{len(raw_candidates)+1:02d}",
                "category": item.get("category"),
                "bbox_px": [x0, y0, x1, y1],
                "depth_m_override": item.get("depth_m"),
                "depth_Thickness_m_override": item.get("thickness_m"),
                "modeling_recipe_override": item.get("modeling_recipe"),
                "work_package_override": item.get("work_package"),
            })
    else:
        # Mode A: Scene-Agnostic Automatic 3D Object Candidate Discovery
        # Works across emissive night scenes, daylight exteriors, studio products, and motion-graphics 3D scenes.
        hue_deg = hsv[:, :, 0] * 2.0  # OpenCV H is 0..180 -> 0..360 deg
        sat_n = hsv[:, :, 1] / 255.0
        val_n = hsv[:, :, 2] / 255.0

        boxes_found = []
        covered_mask = np.zeros((h, w), dtype=np.uint8)

        # 1) Discover distinct Emissive / High-Saliency Accent Clusters (only if localized emissive cores exist)
        has_emissive = bool(light_info.get("radial_halo_profile", {}).get("emissive_core_detected", False))
        if has_emissive:
            bright_emissive = (val_n > 0.56) & (sat_n > 0.36)
            hue_bands = [
                ("Emissive_Warm", bright_emissive & ((hue_deg < 48) | (hue_deg >= 340)), (25, 13)),
                ("Emissive_Magenta", bright_emissive & (hue_deg >= 275) & (hue_deg < 340), (27, 13)),
                ("Emissive_Cool", bright_emissive & (hue_deg >= 170) & (hue_deg < 275), (15, 21)),
                ("Emissive_Green", bright_emissive & (hue_deg >= 48) & (hue_deg < 170), (21, 11)),
            ]
            for band_label, band_mask, ksize in hue_bands:
                kern = cv.getStructuringElement(cv.MORPH_RECT, ksize)
                closed = cv.morphologyEx(band_mask.astype(np.uint8) * 255, cv.MORPH_CLOSE, kern)
                num_labels, _, stats_cc, _ = cv.connectedComponentsWithStats(closed)
                for lbl in range(1, num_labels):
                    x0, y0, bw, bh, area = [int(v) for v in stats_cc[lbl]]
                    if bw < int(0.045 * w) or bh < int(0.038 * h) or (bw * bh) < int(0.0035 * w * h):
                        continue
                    if bw > int(0.92 * w) and bh > int(0.85 * h):
                        continue
                    boxes_found.append((bw * bh, [x0, y0, x0 + bw, y0 + bh], "EMISSIVE_OR_ACCENT", band_label))
                    covered_mask[y0:y0 + bh, x0:x0 + bw] = 255

        # 2) Discover Non-Emissive Hero / Salient 3D Subjects via border-background contrast & edge density
        # Essential for daylight exteriors, studio product renders, clay/low-poly, and NPR scenes!
        border_pixels = np.concatenate([
            rgb[:max(4, h // 16), :, :].reshape(-1, 3),
            rgb[-max(4, h // 16):, :, :].reshape(-1, 3),
            rgb[:, :max(4, w // 16), :].reshape(-1, 3),
            rgb[:, -max(4, w // 16):, :].reshape(-1, 3),
        ], axis=0)
        bg_med_rgb = np.median(border_pixels, axis=0)
        color_dist = np.linalg.norm(rgb - bg_med_rgb, axis=2)
        edges_canny = cv.Canny(gray, 50, 140)
        edge_dens = cv.GaussianBlur(edges_canny.astype(np.float32) / 255.0, (21, 21), 5.0)
        saliency = (color_dist > max(28.0, float(np.percentile(color_dist, 72)))) & (edge_dens > 0.045)
        saliency_u8 = cv.morphologyEx(
            (saliency & (covered_mask == 0)).astype(np.uint8) * 255,
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_RECT, (23, 23)),
        )
        n_sal, _, st_sal, _ = cv.connectedComponentsWithStats(saliency_u8)
        for lbl in range(1, n_sal):
            x0, y0, bw, bh, area = [int(v) for v in st_sal[lbl]]
            if bw < int(0.07 * w) or bh < int(0.07 * h) or (bw * bh) < int(0.008 * w * h):
                continue
            if bw > int(0.92 * w) and bh > int(0.88 * h):
                continue
            boxes_found.append((bw * bh * 0.90, [x0, y0, x0 + bw, y0 + bh], "HERO_SUBJECT", "Hero_Subject"))

        # 3) Discover Independently Moving Dynamic 3D Objects anywhere in frame via flow homography residual
        if len(flow_pts0) >= 8 and len(flow_res) == len(flow_pts0):
            moving_mask = flow_res > max(1.8, float(np.percentile(flow_res, 75)))
            mov_pts = flow_pts0[moving_mask]
            if len(mov_pts) >= 3:
                canvas_mov = np.zeros((h, w), dtype=np.uint8)
                rad_px = int(0.025 * min(w, h))
                for pt in np.rint(mov_pts).astype(int):
                    px, py = int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))
                    if covered_mask[py, px] == 0:
                        cv.circle(canvas_mov, (px, py), rad_px, 255, -1)
                canvas_mov = cv.morphologyEx(canvas_mov, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (21, 17)))
                n_m, _, st_m, _ = cv.connectedComponentsWithStats(canvas_mov)
                for lbl in range(1, n_m):
                    x0, y0, bw, bh, area = [int(v) for v in st_m[lbl]]
                    if bw < int(0.055 * w) or bh < int(0.05 * h):
                        continue
                    boxes_found.append((bw * bh * 0.95, [x0, y0, x0 + bw, y0 + bh], "DYNAMIC_OBJECT", "Dynamic_Object"))

        # Deduplicate overlapping boxes via IoU
        boxes_found.sort(key=lambda item: item[0], reverse=True)
        kept = []
        for score_area, b, cat, sub_lbl in boxes_found:
            overlap = False
            for _, kb, _, _ in kept:
                ix0, iy0 = max(b[0], kb[0]), max(b[1], kb[1])
                ix1, iy1 = min(b[2], kb[2]), min(b[3], kb[3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                min_area = min((b[2] - b[0]) * (b[3] - b[1]), (kb[2] - kb[0]) * (kb[3] - kb[1]))
                if inter / max(1.0, float(min_area)) > 0.55:
                    overlap = True
                    break
            if not overlap and len(kept) < 12:
                kept.append((score_area, b, cat, sub_lbl))

        for idx, (_, b, cat, sub_lbl) in enumerate(kept, 1):
            raw_candidates.append({
                "id": f"obj_{idx:02d}",
                "name": f"{sub_lbl}_{idx:02d}",
                "category": cat,
                "bbox_px": b,
            })

        # 4) Add generic Environment Structure & Ground/Backdrop zones when non-uniform visual structure exists
        upper_std = float(np.std(gray[:int(0.55 * h), :]))
        lower_std = float(np.std(gray[int(0.55 * h):, :]))
        if upper_std > 12.0:
            raw_candidates.append({
                "id": f"obj_{len(raw_candidates)+1:02d}",
                "name": "Environment_Structure_Zone",
                "category": "ENVIRONMENT_STRUCTURE",
                "bbox_px": [0, 0, int(0.50 * w), int(0.58 * h)],
            })
        if lower_std > 8.0 or len(raw_candidates) == 0:
            raw_candidates.append({
                "id": f"obj_{len(raw_candidates)+1:02d}",
                "name": "Ground_or_Backdrop_Surface",
                "category": "GROUND_OR_BACKDROP",
                "bbox_px": [int(0.05 * w), int(0.55 * h), int(0.95 * w), h - 1],
            })

    # Normalize legacy category aliases to canonical universal families while preserving user's label
    def canonical_family(cat_str):
        c = (cat_str or "").upper()
        if c in ("EMISSIVE_OR_ACCENT", "EMISSIVE_SIGN_OR_FIXTURE"):
            return "EMISSIVE_OR_ACCENT"
        if c in ("HERO_SUBJECT",):
            return "HERO_SUBJECT"
        if c in ("DYNAMIC_OBJECT", "DYNAMIC_VEHICLE_OR_PROP"):
            return "DYNAMIC_OBJECT"
        if c in ("ENVIRONMENT_STRUCTURE", "MODULAR_BUILDING_FACADE"):
            return "ENVIRONMENT_STRUCTURE"
        if c in ("INSTANCE_OR_SCATTER_GROUP", "CROWD_OR_INSTANCE_GROUP"):
            return "INSTANCE_OR_SCATTER_GROUP"
        return "GROUND_OR_BACKDROP"

    shading_paradigm = mat_info.get("shading_paradigm", "PBR_PRINCIPLED_BSDF")
    shader_base = (
        "Shader to RGB + ColorRamp (Constant/Toon)"
        if shading_paradigm == "NPR_TOON_SHADER_TO_RGB"
        else "Principled BSDF"
    )
    scene_rough_med = mat_info.get("estimated_principled_bsdf", {}).get("roughness_median", 0.4)
    scene_rough_rng = mat_info.get("estimated_principled_bsdf", {}).get("roughness_range", [0.2, 0.6])

    # Measure 3D spatial properties, materials, motion, and Blender modeling recipes for every object
    measured_objects = []
    for idx, cand in enumerate(raw_candidates, 1):
        x0, y0, x1, y1 = cand["bbox_px"]
        bw, bh = max(8, x1 - x0), max(8, y1 - y0)
        cx_px, cy_px = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        area_ratio = round((bw * bh) / float(w * h), 4)

        patch_rgb = rgb[y0:y1, x0:x1]
        patch_hsv = hsv[y0:y1, x0:x1]
        patch_bgr = bgr[y0:y1, x0:x1]

        # Dominant bright/featured color & emission fraction
        sat_p = patch_hsv[:, :, 1] / 255.0
        val_p = patch_hsv[:, :, 2] / 255.0
        emis_frac = float(np.mean((val_p > 0.68) & (sat_p > 0.42)))
        feat_mask = (val_p > 0.35) & (sat_p > 0.25)
        if np.any(feat_mask):
            dom_rgb = np.median(patch_rgb[feat_mask], axis=0)
        else:
            dom_rgb = np.median(patch_rgb.reshape(-1, 3), axis=0)
        dom_hex = rgb_to_hex(dom_rgb)

        # Category classification if not pre-assigned
        cat = cand.get("category")
        if not cat:
            if emis_frac > 0.12:
                cat = "EMISSIVE_OR_ACCENT"
            elif cy_px > 0.58 * h and area_ratio > 0.16:
                cat = "GROUND_OR_BACKDROP"
            elif cy_px < 0.52 * h and area_ratio > 0.14:
                cat = "ENVIRONMENT_STRUCTURE"
            else:
                cat = "HERO_SUBJECT"
        fam = canonical_family(cat)

        # Check optical flow vectors inside bbox
        vx_med, vy_med, res_med = 0.0, 0.0, 0.0
        if len(flow_pts0) > 0:
            in_box = (flow_pts0[:, 0] >= x0) & (flow_pts0[:, 0] <= x1) & (flow_pts0[:, 1] >= y0) & (flow_pts0[:, 1] <= y1)
            if np.any(in_box):
                vx_med = float(np.median(flow_vecs[in_box, 0]))
                vy_med = float(np.median(flow_vecs[in_box, 1]))
                if len(flow_res) == len(flow_pts0):
                    res_med = float(np.median(flow_res[in_box]))

        is_independent_moving = bool(res_med > 2.4 or fam in ("DYNAMIC_OBJECT", "INSTANCE_OR_SCATTER_GROUP"))
        motion_state = "INDEPENDENT_MOVING" if is_independent_moving else "STATIC_SCENE_STRUCTURE"

        # Estimate 3D depth Z_m in camera space
        if cand.get("depth_m_override") is not None:
            depth_m = float(cand["depth_m_override"])
        elif fam in ("HERO_SUBJECT", "EMISSIVE_OR_ACCENT"):
            depth_m = float(np.clip(11.5 / math.sqrt(max(0.015, area_ratio * 3.5)), 4.0, 28.0))
        elif fam == "ENVIRONMENT_STRUCTURE":
            depth_m = 22.0
        elif fam == "GROUND_OR_BACKDROP":
            depth_m = 16.5
        elif fam == "INSTANCE_OR_SCATTER_GROUP":
            depth_m = 15.0
        else:
            depth_m = float(np.clip(16.0 - 4.0 * ((cy_px / float(h)) - 0.5), 6.0, 24.0))

        if depth_m <= 11.5:
            depth_layer = "FG_HERO"
        elif depth_m <= 18.5:
            depth_layer = "MG_SUBJECT_OR_PROP"
        else:
            depth_layer = "BG_ENVIRONMENT"

        # Compute 3D Camera-Space center (X_m, Y_m, Z_m) and real-world dimensions (W_m, H_m, D_m)
        X_m = ((cx_px / float(w)) - 0.5) * depth_m * fov_scale
        Y_m = (0.5 - (cy_px / float(h))) * depth_m * fov_scale * (float(h) / float(w))
        Z_m = -depth_m
        W_m = (bw / float(w)) * depth_m * fov_scale
        H_m = (bh / float(h)) * depth_m * fov_scale * (float(h) / float(w))
        if cand.get("depth_Thickness_m_override") is not None:
            D_m = float(cand["depth_Thickness_m_override"])
        elif fam == "EMISSIVE_OR_ACCENT":
            D_m = 0.18
        elif fam in ("HERO_SUBJECT", "DYNAMIC_OBJECT"):
            D_m = round(max(0.4, min(W_m, H_m) * 0.65), 2)
        elif fam == "ENVIRONMENT_STRUCTURE":
            D_m = 4.0
        else:
            D_m = 0.5

        # Assign Scene-Agnostic Blender Modeling Recipe & Subagent Work Package
        if cand.get("modeling_recipe_override"):
            recipe = cand["modeling_recipe_override"]
        elif fam == "EMISSIVE_OR_ACCENT":
            recipe = f"Multi-layer Relief/Solidify Mesh + Non-Additive Opaque Alpha {shader_base} + Calibrated Emission"
        elif fam == "HERO_SUBJECT":
            recipe = f"Primary 3D Mesh (Subdiv / Hard-Surface + Bevel) + UV Texture / Procedural {shader_base}"
        elif fam == "DYNAMIC_OBJECT":
            recipe = f"Articulated / Rigid 3D Asset + Bevel + Independent Keyframe/Path Animation + {shader_base}"
        elif fam == "ENVIRONMENT_STRUCTURE":
            recipe = f"Modular Structural Unit + Loop-Cut / Extrude + Array / Geometry Nodes Instancing + {shader_base}"
        elif fam == "INSTANCE_OR_SCATTER_GROUP":
            recipe = f"Base Asset Variant(s) + Collection Instance / Geometry Nodes Point Distribute + {shader_base}"
        else:
            recipe = (
                f"Subdivided Ground/Backdrop Surface + {shader_base} "
                f"(Roughness≈{scene_rough_med}, range {scene_rough_rng})"
            )

        if cand.get("work_package_override"):
            wp = cand["work_package_override"]
        elif fam in ("HERO_SUBJECT", "EMISSIVE_OR_ACCENT"):
            wp = "WP1_Hero_&_Emissive_Assets"
        elif fam == "DYNAMIC_OBJECT":
            wp = "WP2_Dynamic_Objects_&_Props"
        elif fam == "ENVIRONMENT_STRUCTURE":
            wp = "WP3_Environment_Structures"
        else:
            wp = "WP4_Ground_Lighting_&_Atmosphere"

        crop_filename = f"objects/{cand['id']}.png"
        cv.imwrite(str(out_path / crop_filename), patch_bgr)

        is_emissive_obj = bool(
            (emis_frac > 0.12 or fam == "EMISSIVE_OR_ACCENT")
            and light_info.get("radial_halo_profile", {}).get("emissive_core_detected", False)
        )
        # Per-object independent material profile (prevents mixed materials collapsing into one global average)
        patch_mat = analyze_patch_material_profile(patch_bgr, light_info)
        obj_rough = patch_mat["recommended_roughness"] if patch_mat else round(float(scene_rough_med), 3)
        obj_finish = patch_mat["surface_finish_class"] if patch_mat else mat_info.get("surface_texture_profile", {}).get("surface_finish_class", "CLEAN_SMOOTH")
        obj_grunge = patch_mat["surface_grunge_detected"] if patch_mat else bool(mat_info.get("surface_texture_profile", {}).get("surface_grunge_detected", False))
        obj_bump = patch_mat["recommended_bump_strength"] if patch_mat else float(mat_info.get("surface_texture_profile", {}).get("recommended_bump_strength", 0.0))
        obj_metal = patch_mat["metallic_hint"] if patch_mat else "DIELECTRIC"
        obj_ink = patch_mat["ink_outline_detected"] if patch_mat else False
        obj_clearcoat = patch_mat["clearcoat_or_glossy_hint"] if patch_mat else False

        # Enrich generic Ground/Backdrop recipe with per-object measured roughness & ground reflection if detected
        gref_info = light_info.get("ground_reflection_profile", {})
        if not cand.get("modeling_recipe_override") and fam == "GROUND_OR_BACKDROP":
            g_r = gref_info.get("recommended_ground_roughness") if gref_info.get("ground_reflection_detected") else obj_rough
            refl_tag = f" + {gref_info.get('reflection_type')}" if gref_info.get("ground_reflection_detected") else ""
            recipe = f"Subdivided Ground/Backdrop Surface + {shader_base} (Roughness≈{g_r}, {obj_finish}{refl_tag})"

        measured_objects.append({
            "index": idx,
            "id": cand["id"],
            "name": cand["name"],
            "category": cat,
            "canonical_family": fam,
            "depth_layer": depth_layer,
            "work_package": wp,
            "bbox_px": [x0, y0, x1, y1],
            "screen_area_ratio": area_ratio,
            "camera_space_location_3d_m": [round(X_m, 3), round(Y_m, 3), round(Z_m, 3)],
            "estimated_dimensions_3d_m": {
                "width_m": round(W_m, 3),
                "height_m": round(H_m, 3),
                "depth_m": round(D_m, 3),
            },
            "motion_3d": {
                "state": motion_state,
                "velocity_px_per_frame": [round(vx_med, 2), round(vy_med, 2)],
                "residual_against_bg_px": round(res_med, 2),
            },
            "material_summary": {
                "dominant_color_hex": dom_hex,
                "emissive_pixel_fraction": round(emis_frac, 3),
                "is_emissive_light_source": is_emissive_obj,
                "recommended_emission_strength": (
                    light_info["recommended_blender_lighting_and_volume"]["non_additive_emission_strength_fg"]
                    if is_emissive_obj else 0.0
                ),
                "recommended_roughness": obj_rough,
                "surface_finish_class": obj_finish,
                "surface_grunge_detected": obj_grunge,
                "recommended_bump_strength": obj_bump,
                "metallic_hint": obj_metal,
                "ink_outline_detected": obj_ink,
                "clearcoat_or_glossy_hint": obj_clearcoat,
            },
            "blender_modeling_recipe": recipe,
            "reference_crop": crop_filename,
        })

    # Render annotated 3D Object Inventory Map (objects_3d_map.png)
    vis = bgr.copy()
    cat_colors_bgr = {
        "HERO_SUBJECT": (70, 210, 255),
        "EMISSIVE_OR_ACCENT": (60, 230, 255),
        "DYNAMIC_OBJECT": (80, 255, 120),
        "ENVIRONMENT_STRUCTURE": (255, 180, 70),
        "GROUND_OR_BACKDROP": (220, 130, 255),
        "INSTANCE_OR_SCATTER_GROUP": (130, 220, 255),
    }
    for obj in measured_objects:
        x0, y0, x1, y1 = obj["bbox_px"]
        col = cat_colors_bgr.get(obj["canonical_family"], (100, 255, 200))
        cv.rectangle(vis, (x0, y0), (x1, y1), col, 2, cv.LINE_AA)
        dims = obj["estimated_dimensions_3d_m"]
        z_abs = abs(obj["camera_space_location_3d_m"][2])
        tag = f"#{obj['index']} {obj['id']} ({dims['width_m']:.1f}x{dims['height_m']:.1f}m @ Z={z_abs:.1f}m)"
        (tw, th), _ = cv.getTextSize(tag, cv.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ty = max(th + 6, y0 + th + 4 if y0 < 24 else y0 - 4)
        cv.rectangle(vis, (x0, ty - th - 4), (min(w - 1, x0 + tw + 8), ty + 3), (18, 18, 22), -1)
        cv.putText(vis, tag, (x0 + 4, ty - 1), cv.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv.LINE_AA)

    cv.imwrite(str(out_path / "objects_3d_map.png"), vis)

    # Group objects by Work Package for Multi-Agent Parallel Modeling
    work_packages = {}
    for obj in measured_objects:
        wp = obj["work_package"]
        work_packages.setdefault(wp, []).append(obj["id"])

    bom_data = {
        "object_count": len(measured_objects),
        "annotated_inventory_map": "objects_3d_map.png",
        "work_packages_sharding": [
            {
                "work_package_id": wp,
                "object_ids": oids,
                "object_count": len(oids),
                "target_collection_name": f"Col_{wp}",
            }
            for wp, oids in sorted(work_packages.items())
        ],
        "objects": measured_objects,
    }

    # Also write human- and agent-readable Markdown BOM table (assets_bom.md)
    md_lines = [
        "# 3D 场景物件建模清单与多 Agent 分工表 (Blender Asset BOM)",
        "",
        "## 一、 多 Agent 并行建模分包 (Work Packages)",
        "",
        "| 分包编号 (Work Package) | Blender 集合名 | 包含 3D 物件 ID | 建议负责内容 |",
        "| --- | --- | --- | --- |",
    ]
    for wp_item in bom_data["work_packages_sharding"]:
        md_lines.append(
            f"| `{wp_item['work_package_id']}` | `{wp_item['target_collection_name']}` | "
            f"{', '.join(f'`{i}`' for i in wp_item['object_ids'])} | 独立建模并导出对应 Collection，原点统一置于相机空间或构件中心 |"
        )

    md_lines.extend([
        "",
        "## 二、 逐物件三维坐标、尺寸、材质与 Blender 建模配方",
        "",
        "| # | 物件 ID 与名称 | 类别 / 深度层 | 画幅框 `[x0,y0,x1,y1]` | 相机空间 3D 位置 `(X,Y,Z)` m | 估计尺寸 `W×H×D` m | 运动状态 `(vx,vy)` px/f | 主色 / 发光 | Blender 建模与材质配方 | 参考裁剪图 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for obj in measured_objects:
        loc = obj["camera_space_location_3d_m"]
        dm = obj["estimated_dimensions_3d_m"]
        mot = obj["motion_3d"]
        mat = obj["material_summary"]
        mat_tags = []
        if mat["is_emissive_light_source"]:
            mat_tags.append(f"发光 `{mat['recommended_emission_strength']}`")
        else:
            mat_tags.append(f"Rough `{mat['recommended_roughness']}`")
        mat_tags.append(f"`{mat.get('surface_finish_class', 'CLEAN')}`")
        if mat.get("recommended_bump_strength", 0.0) > 0:
            mat_tags.append(f"Bump `{mat['recommended_bump_strength']}`")
        if mat.get("ink_outline_detected"):
            mat_tags.append("`INK`")
        if mat.get("metallic_hint") == "METALLIC_OR_COLORED_SPEC":
            mat_tags.append("`SPEC/MET`")
        emis_str = " / ".join(mat_tags)
        md_lines.append(
            f"| {obj['index']} | **`{obj['id']}`**<br>{obj['name']} | `{obj['category']}`<br>(`{obj['depth_layer']}`) | "
            f"`{obj['bbox_px']}` | `({loc[0]}, {loc[1]}, {loc[2]})` | `{dm['width_m']}×{dm['height_m']}×{dm['depth_m']}` | "
            f"`{mot['state']}`<br>`{mot['velocity_px_per_frame']}` | `{mat['dominant_color_hex']}`<br>{emis_str} | "
            f"{obj['blender_modeling_recipe']} | `{obj['reference_crop']}` |"
        )

    (out_path / "assets_bom.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return bom_data


def _find_frame_png(scene3d_json_path):
    parent = Path(scene3d_json_path).parent
    pngs = sorted(parent.glob("frame_*.png"))
    return pngs[0] if pngs else None


def _compute_ssim_gray(a_f, b_f):
    """Lightweight global SSIM on [0, 1] float grayscale images using 11x11 Gaussian window."""
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    mu_a = cv.GaussianBlur(a_f, (11, 11), 1.5)
    mu_b = cv.GaussianBlur(b_f, (11, 11), 1.5)
    sig_a2 = cv.GaussianBlur(a_f * a_f, (11, 11), 1.5) - mu_a * mu_a
    sig_b2 = cv.GaussianBlur(b_f * b_f, (11, 11), 1.5) - mu_b * mu_b
    sig_ab = cv.GaussianBlur(a_f * b_f, (11, 11), 1.5) - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sig_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sig_a2 + sig_b2 + c2)
    ssim_map = num / np.maximum(1e-8, den)
    return float(np.mean(ssim_map))


def compare_scene3d_pair(
    orig_json_path, render_json_path, out_dir=None, *, reference_video=None, render_video=None
):
    """Automated Bidirectional Look-Dev, Photometry, Material & Structure Gate Judge.
    Compares an original video's scene3d.json against a replica render's scene3d.json,
    returning an objective PASS / FAIL scorecard with every failing dimension and actionable fix hint.
    """
    orig_p = Path(orig_json_path).resolve()
    rend_p = Path(render_json_path).resolve()
    orig = json.loads(orig_p.read_text(encoding="utf-8"))
    rend = json.loads(rend_p.read_text(encoding="utf-8"))

    checks = []

    def add_check(category, metric, status, orig_val, rend_val, detail, fix_hint=""):
        checks.append({
            "category": category,
            "metric": metric,
            "status": status,  # PASS | WARN | FAIL
            "orig": orig_val,
            "render": rend_val,
            "detail": detail,
            "fix_hint": fix_hint,
        })

    # 1. 18-Gate Evidence Parity
    og = orig.get("lookdev_feature_gates", {})
    rg = rend.get("lookdev_feature_gates", {})
    critical_gates = {
        "enable_volume_fog": "按原片体积散射密度开启/关闭 Principled Volume",
        "enable_daylight_depth_haze": "按原片远景雾度开启/关闭 Mist Pass 或低密度大气雾",
        "enable_bloom_or_soft_glow": "按原片近场辉晕开启/关闭合成器 Fog Glow / 柔焦层",
        "enable_dof": "按原片焦点深度差开启/关闭相机真实景深 (Camera DOF + f-stop)",
        "enable_motion_blur": "按原片快门值开启/关闭 Render Motion Blur (Shutter)",
        "enable_emissive_materials": "仅对原片真实自发光构件开启 Emission，严禁拿 Emission 替代受光打光",
        "enable_emissive_indirect_spill": "开启 Cycles 真实间接漫反射/镜面弹射（diffuse/glossy >= 2），严禁使用关光追 EEVEE 导致自发光无法照亮周边几何",
        "enable_surface_grunge_bump": "按原片肌理等级配置宏/中/微多尺度 Object 坐标贴图与 Bump",
        "enable_ink_outline": "按原片描边宽度开启/关闭 Grease Pencil Line Art / Freestyle / Inverted Hull",
        "enable_vignette": "按原片四角衰减比例在合成器叠加/移除径向暗角",
        "enable_film_or_paper_grain": "按原片全屏颗粒 RMS 在合成器叠加/移除统一颗粒层",
        "enable_split_tone_lighting": "按原片冷暖极性匹配主光与辅光/环境光色温",
        "enable_rim_backlight": "按原片轮廓逆光增减独立背光 Rim Light",
        "enable_ground_reflection": "按原片地面倒影类型调整地坪 Roughness 与镜面/湿地反射",
        "enable_film_halation": "按原片高光红晕增减合成器红色通道辉晕",
        "enable_anamorphic_streak": "按原片水平拉丝炫光增减 Glare Streaks",
    }
    advisory_gates = {
        "enable_lens_dispersion": "调整合成器 Lens Distortion Dispersion",
        "enable_texture_boiling": "检查定格动画静止面纹理随机跳帧",
        "enable_camera_shake": "核对相机抖动/Noise F-Curve 曲线",
    }

    for gkey, hint in {**critical_gates, **advisory_gates}.items():
        if gkey not in og or gkey not in rg:
            continue
        ov, rv = bool(og[gkey]), bool(rg[gkey])
        if ov == rv:
            add_check("gates", gkey, "PASS", ov, rv, "MATCH")
        else:
            direction = "MISSING_IN_RENDER" if ov else "HALLUCINATED_IN_RENDER"
            sev = "FAIL" if gkey in critical_gates else "WARN"
            add_check("gates", gkey, sev, ov, rv, direction, hint)

    # 2. Bidirectional Photometry, Shadow Softness & Halo Checks
    op4 = orig.get("pillar_4_lighting_and_volumetrics", {})
    rp4 = rend.get("pillar_4_lighting_and_volumetrics", {})
    olum = op4.get("linear_luminance_percentiles", {})
    rlum = rp4.get("linear_luminance_percentiles", {})

    o_p50, r_p50 = float(olum.get("p50_midtone", 0.0)), float(rlum.get("p50_midtone", 0.0))
    if max(o_p50, r_p50) >= 0.012:
        rel_p50 = (r_p50 - o_p50) / max(0.012, o_p50)
        st = "PASS" if abs(rel_p50) <= 0.45 else "FAIL"
        add_check(
            "photometry", "p50_midtone", st, o_p50, r_p50,
            f"rel_diff={rel_p50:+.1%}",
            "调高/压低真实环境辅光、主光能量或曝光（严禁给非自发光面混大比例 Emission 提亮）" if st == "FAIL" else "",
        )

    o_p85, r_p85 = float(olum.get("p85_key_level", 0.0)), float(rlum.get("p85_key_level", 0.0))
    if max(o_p85, r_p85) >= 0.04:
        rel_p85 = (r_p85 - o_p85) / max(0.04, o_p85)
        st = "PASS" if abs(rel_p85) <= 0.40 else "FAIL"
        add_check(
            "photometry", "p85_key_level", st, o_p85, r_p85,
            f"rel_diff={rel_p85:+.1%}",
            "调整主光强度与 AgX 曝光，使亮部线性电平贴合原片 ±40%" if st == "FAIL" else "",
        )

    o_pen = float(op4.get("shadow_and_key_light_hardness", {}).get("shadow_penumbra_width_px", 0.0))
    r_pen = float(rp4.get("shadow_and_key_light_hardness", {}).get("shadow_penumbra_width_px", 0.0))
    if o_pen >= 1.0 and r_pen >= 0.5:
        pen_ratio = r_pen / o_pen
        st = "PASS" if (0.55 <= pen_ratio <= 1.85) else "FAIL"
        detail = f"ratio={pen_ratio:.2f}x ({'TOO_SHARP_CG' if pen_ratio < 0.55 else 'TOO_BLURRY' if pen_ratio > 1.85 else 'OK'})"
        add_check(
            "photometry", "shadow_penumbra_width_px", st, o_pen, r_pen, detail,
            "调大主光软影半径并开启真实景深/柔焦/运动模糊，消除过锐 CG 矢量刀切边" if pen_ratio < 0.55 else (
                "减小主光面积并核对对焦距离，收紧阴影与光学边缘" if pen_ratio > 1.85 else ""
            ),
        )

    orad = op4.get("radial_halo_profile", {})
    rrad = rp4.get("radial_halo_profile", {})
    if og.get("enable_bloom_or_soft_glow"):
        o_nb, r_nb = float(orad.get("near_bloom_linear_lum", 0.0)), float(rrad.get("near_bloom_linear_lum", 0.0))
        if max(o_nb, r_nb) >= 0.04:
            rel_nb = (r_nb - o_nb) / max(0.04, o_nb)
            st = "PASS" if abs(rel_nb) <= 0.35 else ("WARN" if abs(rel_nb) <= 0.50 else "FAIL")
            add_check(
                "photometry", "near_bloom_linear_lum", st, o_nb, r_nb, f"rel_diff={rel_nb:+.1%}",
                "调整合成器 Glare 阈值/强度或叠加宽场 Bloom 散射层，严禁仅用高阈值弱 Glare 敷衍" if st != "PASS" else "",
            )

    if og.get("enable_volume_fog"):
        o_far = float(orad.get("far_volumetric_excess", 0.0))
        r_far = float(rrad.get("far_volumetric_excess", 0.0))
        if o_far >= 0.010:
            rel_far = (r_far - o_far) / o_far
            st_far = "FAIL" if (rel_far < -0.55 or rel_far > 1.20) else ("WARN" if abs(rel_far) > 0.40 else "PASS")
            add_check(
                "photometry", "far_volumetric_excess", st_far, round(o_far, 4), round(r_far, 4),
                f"rel_diff={rel_far:+.1%}",
                "补齐 Principled Volume 散射密度/各向异性及 Cycles volume_bounces >= 1，匹配远场光雾衰减" if st_far != "PASS" else "",
            )

    # Indirect emissive chromatic spill & relative non-core crushed-black check (source-relative, safe for dark-void products)
    oesp = op4.get("emissive_spill_profile", {})
    resp = rp4.get("emissive_spill_profile", {})
    if og.get("enable_emissive_indirect_spill") and oesp and resp:
        o_chr = float(oesp.get("spill_ring_chromatic_energy", 0.0))
        r_chr = float(resp.get("spill_ring_chromatic_energy", 0.0))
        if o_chr >= 0.015:
            chr_ratio = r_chr / o_chr
            st_chr = "FAIL" if chr_ratio < 0.55 else ("WARN" if chr_ratio > 2.20 else "PASS")
            add_check(
                "photometry", "emissive_spill_chromatic_energy", st_chr, round(o_chr, 4), round(r_chr, 4),
                f"ratio={chr_ratio:.2f}x",
                "自发光周边彩色溢光严重衰减：确认启用 Cycles 真实间接弹射（禁止无光追 EEVEE）并检查周边材质反照率与粗糙度" if st_chr == "FAIL" else "",
            )

    if "non_core_crushed_black_ratio" in oesp and "non_core_crushed_black_ratio" in resp:
        o_void = float(oesp["non_core_crushed_black_ratio"])
        r_void = float(resp["non_core_crushed_black_ratio"])
        void_delta = r_void - o_void
        void_ratio = r_void / max(0.05, o_void)
        st_void = "FAIL" if (void_delta >= 0.08 and void_ratio >= 1.35) else "PASS"
        if st_void == "FAIL" or o_void >= 0.05:
            add_check(
                "photometry", "non_core_crushed_black_ratio", st_void, round(o_void, 3), round(r_void, 3),
                f"delta={void_delta:+.3f} (ratio={void_ratio:.2f}x)",
                "非光源区死黑塌陷明显多于原片：开启真实间接光弹射与环境底光，避免未受光建筑/阴影成纯黑空洞" if st_void == "FAIL" else "",
            )

    # 4-Quadrant regional tonal dynamic range (catches flat-wash lighting anywhere in frame)
    orcon = op4.get("regional_tonal_contrast_profile", {}).get("quadrant_highlight_to_median", {})
    rrcon = rp4.get("regional_tonal_contrast_profile", {}).get("quadrant_highlight_to_median", {})
    if orcon and rrcon:
        worst_q = None
        worst_ratio = 999.0
        worst_vals = (0.0, 0.0)
        for qname, oq in orcon.items():
            rq = rrcon.get(qname)
            if oq is not None and rq is not None and float(oq) >= 2.8:
                q_ratio = float(rq) / float(oq)
                if q_ratio < worst_ratio:
                    worst_ratio = q_ratio
                    worst_q = qname
                    worst_vals = (float(oq), float(rq))
        if worst_q is not None:
            st_q = "FAIL" if worst_ratio < 0.55 else ("WARN" if worst_ratio < 0.70 else "PASS")
            add_check(
                "photometry", "regional_tonal_contrast", st_q, round(worst_vals[0], 2), round(worst_vals[1], 2),
                f"worst_quad={worst_q} ratio={worst_ratio:.2f}x",
                "局部象限明暗反差被洗平（若同帧 SSIM/Edge IoU 偏低请先修几何对位；否则缩减超大面积泛光补光并恢复定向光与表面高光反射）" if st_q != "PASS" else "",
            )

    # 3. Material, Bidirectional Texture Spectrum, Zone-Split & Stretch Anisotropy
    op3 = orig.get("pillar_3_material_and_bsdf", {})
    rp3 = rend.get("pillar_3_material_and_bsdf", {})
    o_sh, r_sh = op3.get("shading_paradigm"), rp3.get("shading_paradigm")
    if o_sh and r_sh:
        st = "PASS" if o_sh == r_sh else "FAIL"
        add_check("material", "shading_paradigm", st, o_sh, r_sh, "MATCH" if st == "PASS" else "MISMATCH",
                  "核对着色范式（Principled BSDF vs NPR Shader to RGB）" if st == "FAIL" else "")

    o_r = float(op3.get("estimated_principled_bsdf", {}).get("roughness_median", 0.4))
    r_r = float(rp3.get("estimated_principled_bsdf", {}).get("roughness_median", 0.4))
    dr = r_r - o_r
    st_r = "PASS" if abs(dr) <= 0.12 else "FAIL"
    add_check(
        "material", "roughness_median", st_r, o_r, r_r,
        f"delta={dr:+.3f} ({'TOO_GLOSSY_PLASTIC' if dr < -0.12 else 'TOO_MATTE' if dr > 0.12 else 'OK'})",
        "提高材质 Roughness 并接入纸纹/磨损粗糙度贴图（勿固定光滑常数）" if dr < -0.12 else (
            "降低材质 Roughness 恢复高光质感" if dr > 0.12 else ""
        ),
    )

    otex = op3.get("surface_texture_profile", {})
    rtex = rp3.get("surface_texture_profile", {})
    o_cls, r_cls = otex.get("surface_finish_class"), rtex.get("surface_finish_class")
    if o_cls and r_cls:
        st = "PASS" if o_cls == r_cls else "FAIL"
        add_check("material", "surface_finish_class", st, o_cls, r_cls, "MATCH" if st == "PASS" else "CLASS_MISMATCH",
                  "按原片表面分级重做宏/中/微三尺度肌理与 Bump" if st == "FAIL" else "")

    # Bidirectional wear speckle check (catches BOTH missing wear AND 3x over-speckled snowflake cheat!)
    o_spk = float(otex.get("wear_speckle_area_ratio", 0.0))
    r_spk = float(rtex.get("wear_speckle_area_ratio", 0.0))
    if max(o_spk, r_spk) >= 0.045:
        spk_ratio = r_spk / max(0.015, o_spk)
        if o_spk >= 0.045 and spk_ratio < 0.50:
            add_check("material", "wear_speckle_area_ratio", "FAIL", o_spk, r_spk,
                      f"ratio={spk_ratio:.2f}x (UNDER_TEXTURED)",
                      "补充可见磨损/刷痕贴图与 Bump，避免表面过滑")
        elif r_spk >= 0.095 and spk_ratio > 1.75:
            add_check("material", "wear_speckle_area_ratio", "FAIL", o_spk, r_spk,
                      f"ratio={spk_ratio:.2f}x (OVER_SPECKLED_NOISE)",
                      "降低高频孤立小白点阈值，改用大/中尺度笔刷与边缘做旧替代均匀雪花斑")
        else:
            add_check("material", "wear_speckle_area_ratio", "PASS", o_spk, r_spk, f"ratio={spk_ratio:.2f}x")

    # Zone-split bright surface texture (catches smooth white letter faces hiding behind noisy colored walls!)
    o_bf, r_bf = otex.get("bright_surface_fine_rms"), rtex.get("bright_surface_fine_rms")
    if o_bf is not None and r_bf is not None and o_bf >= 0.0045:
        bf_ratio = float(r_bf) / max(1e-4, float(o_bf))
        st_bf = "PASS" if (0.55 <= bf_ratio <= 1.85) else "FAIL"
        add_check(
            "material", "bright_surface_fine_rms", st_bf, o_bf, r_bf,
            f"ratio={bf_ratio:.2f}x ({'BRIGHT_FACE_TOO_SMOOTH' if bf_ratio < 0.55 else 'BRIGHT_FACE_TOO_NOISY' if bf_ratio > 1.85 else 'OK'})",
            "白色/亮面材质缺失纸纹颗粒：须对亮色面独立挂载粗纸纹/画布贴图与 Roughness 扰动" if bf_ratio < 0.55 else (
                "降低亮面高频噪点强度" if bf_ratio > 1.85 else ""
            ),
        )

    # Texture stretch anisotropy (catches Generated-coord stretching along extruded side walls)
    o_aniso = float(otex.get("texture_stretch_anisotropy", 1.0))
    r_aniso = float(rtex.get("texture_stretch_anisotropy", 1.0))
    if r_aniso > max(2.15, o_aniso * 1.40):
        add_check(
            "material", "texture_stretch_anisotropy", "FAIL", o_aniso, r_aniso,
            f"STRETCHED_TEXTURE ({r_aniso:.2f} vs {o_aniso:.2f})",
            "贴图/噪点坐标发生方向拉伸：将 ShaderNodeTexCoord 从 Generated 改为 Object 坐标或做 Box 投射/展开 UV",
        )
    else:
        add_check("material", "texture_stretch_anisotropy", "PASS", o_aniso, r_aniso, f"ratio={r_aniso/max(1e-3,o_aniso):.2f}x")

    # 4. Visual Frame Structure & Silhouette Alignment (when frame_*.png exists beside both JSONs)
    orig_img_p = _find_frame_png(orig_p)
    rend_img_p = _find_frame_png(rend_p)
    structure_info = {}
    diff_vis_bgr = None
    orig_bgr = rend_bgr = None
    if orig_img_p and rend_img_p:
        orig_bgr = cv.imread(str(orig_img_p))
        rend_bgr = cv.imread(str(rend_img_p))
        if orig_bgr is not None and rend_bgr is not None:
            th, tw = orig_bgr.shape[:2]
            if rend_bgr.shape[:2] != (th, tw):
                rend_bgr = cv.resize(rend_bgr, (tw, th), interpolation=cv.INTER_AREA)
            og_gray = cv.cvtColor(orig_bgr, cv.COLOR_BGR2GRAY)
            rg_gray = cv.cvtColor(rend_bgr, cv.COLOR_BGR2GRAY)
            ssim_val = round(_compute_ssim_gray(og_gray.astype(np.float32) / 255.0, rg_gray.astype(np.float32) / 255.0), 4)

            # Dilated edge silhouette IoU (4px tolerance captures pose, font shape, framing & clipping errors)
            oe = cv.Canny(og_gray, 50, 140) > 0
            re_m = cv.Canny(rg_gray, 50, 140) > 0
            k_tol = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
            oe_dil = cv.dilate(oe.astype(np.uint8), k_tol) > 0
            re_dil = cv.dilate(re_m.astype(np.uint8), k_tol) > 0
            inter_e = int(np.sum((oe & re_dil) | (re_m & oe_dil)))
            union_e = max(1, int(np.sum(oe | re_m)))
            edge_iou = round(inter_e / float(union_e), 4)

            # Per-quadrant mean absolute RGB difference (0..255) to localize mismatch regions
            abs_diff = np.mean(np.abs(orig_bgr.astype(np.float32) - rend_bgr.astype(np.float32)), axis=2)
            quad_diffs = {
                "top_left": round(float(np.mean(abs_diff[:th // 2, :tw // 2])), 1),
                "top_right": round(float(np.mean(abs_diff[:th // 2, tw // 2:])), 1),
                "bottom_left": round(float(np.mean(abs_diff[th // 2:, :tw // 2])), 1),
                "bottom_right": round(float(np.mean(abs_diff[th // 2:, tw // 2:])), 1),
            }
            worst_quad = max(quad_diffs.items(), key=lambda x: x[1])
            structure_info = {
                "ssim_global": ssim_val,
                "edge_silhouette_iou_4px": edge_iou,
                "mean_abs_rgb_diff_0_255": round(float(np.mean(abs_diff)), 2),
                "quadrant_rgb_diff_0_255": quad_diffs,
                "worst_mismatch_quadrant": worst_quad[0],
            }
            st_ssim = "PASS" if ssim_val >= 0.72 else ("WARN" if ssim_val >= 0.55 else "FAIL")
            add_check("structure", "ssim_global", st_ssim, 1.0, ssim_val,
                      f"worst_quad={worst_quad[0]}({worst_quad[1]})",
                      f"重点核查 {worst_quad[0]} 象限的构图、造型/姿势、字体与曝光差异" if st_ssim != "PASS" else "")
            st_iou = "PASS" if edge_iou >= 0.55 else ("WARN" if edge_iou >= 0.38 else "FAIL")
            add_check("structure", "edge_silhouette_iou_4px", st_iou, 1.0, edge_iou,
                      f"edge_overlap={edge_iou:.1%}",
                      "几何轮廓、角色姿势、字形或穿模与原片存在明显错位，对照热力图逐件修正" if st_iou != "PASS" else "")

            diff_u8 = np.clip(abs_diff * 2.5, 0, 255).astype(np.uint8)
            diff_vis_bgr = cv.applyColorMap(diff_u8, cv.COLORMAP_INFERNO)

    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    passes = [c for c in checks if c["status"] == "PASS"]
    verdict = "PASS" if len(fails) == 0 else "FAIL"

    report = {
        "verdict": verdict,
        "summary": {
            "total_checks": len(checks),
            "pass_count": len(passes),
            "warn_count": len(warns),
            "fail_count": len(fails),
        },
        "orig_scene3d_json": str(orig_p),
        "render_scene3d_json": str(rend_p),
        "structure_alignment": structure_info,
        "failed_checks": fails,
        "warning_checks": warns,
        "all_checks": checks,
    }

    from lookdiff import compare_metadata, markdown as lookdiff_markdown
    report["local_appearance"] = compare_metadata(
        orig, rend, out_dir, reference_video=reference_video, render_video=render_video
    )

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "compare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Markdown Scorecard
        md = [
            f"# 3D 复刻双向闭环验收报告 — `{verdict}`",
            "",
            f"- **原片测量**：`{orig_p}`",
            f"- **复刻测量**：`{rend_p}`",
            f"- **判定汇总**：`PASS {len(passes)}` / `WARN {len(warns)}` / **`FAIL {len(fails)}`**",
        ]
        if structure_info:
            md.append(
                f"- **结构与轮廓对齐**：`SSIM = {structure_info['ssim_global']}`，"
                f"`Edge IoU(4px) = {structure_info['edge_silhouette_iou_4px']}`，"
                f"最大偏差象限 **`{structure_info['worst_mismatch_quadrant']}`**"
            )
        md.extend(["", "## 必修正项（FAIL）", ""])
        if fails:
            md.append("| 维度 | 指标 | 原片 | 复刻 | 偏差说明 | 建议修正动作 |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for c in fails:
                md.append(f"| `{c['category']}` | `{c['metric']}` | `{c['orig']}` | `{c['render']}` | {c['detail']} | {c['fix_hint']} |")
        else:
            md.append("无 FAIL 项，满足双向闭环交付门槛。")
        if warns:
            md.extend(["", "## 关注项（WARN）", "", "| 维度 | 指标 | 原片 | 复刻 | 偏差说明 | 建议 |", "| --- | --- | --- | --- | --- | --- |"])
            for c in warns:
                md.append(f"| `{c['category']}` | `{c['metric']}` | `{c['orig']}` | `{c['render']}` | {c['detail']} | {c['fix_hint']} |")
        md.extend(lookdiff_markdown(report["local_appearance"]))
        (out_p / "compare_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        # 4-panel visual judge chart (compare_report.png)
        if orig_bgr is not None and rend_bgr is not None and diff_vis_bgr is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
            axes[0].imshow(cv.cvtColor(orig_bgr, cv.COLOR_BGR2RGB))
            axes[0].set_title("1. Original Source Frame", fontsize=10, fontweight="bold")
            axes[0].axis("off")
            axes[1].imshow(cv.cvtColor(rend_bgr, cv.COLOR_BGR2RGB))
            axes[1].set_title("2. Replica Render Frame", fontsize=10, fontweight="bold")
            axes[1].axis("off")
            axes[2].imshow(cv.cvtColor(diff_vis_bgr, cv.COLOR_BGR2RGB))
            axes[2].set_title(f"3. Diff Heatmap (SSIM={structure_info.get('ssim_global',0):.2f}, EdgeIoU={structure_info.get('edge_silhouette_iou_4px',0):.2f})", fontsize=9.5, fontweight="bold")
            axes[2].axis("off")
            axes[3].axis("off")
            color_v = "#C62828" if verdict == "FAIL" else "#2E7D32"
            lines_s = [
                f"VERDICT: {verdict} (FAIL={len(fails)}, WARN={len(warns)})",
                "-" * 42,
            ]
            for c in fails[:10]:
                lines_s.append(f"[FAIL] {c['metric']}: {c['orig']} -> {c['render']}")
                lines_s.append(f"       ({c['detail']})")
            for c in warns[:4]:
                lines_s.append(f"[WARN] {c['metric']}: {c['orig']} -> {c['render']}")
            if not fails and not warns:
                lines_s.append("All 18 gates, photometry, material &")
                lines_s.append("structure alignment checks PASSED.")
            axes[3].text(0.02, 0.96, "\n".join(lines_s), va="top", ha="left", fontsize=7.8, family="monospace", color=color_v if verdict == "FAIL" else "#1B5E20")
            fig.savefig(out_p / "compare_report.png", dpi=140)
            plt.close(fig)

    return report


