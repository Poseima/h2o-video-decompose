"""Multi-genre, zero-overfit unit tests for scene3d lighting, spill, reflection, contrast & lookdev keyframe selection."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import scene3d_core as s3  # noqa: E402
import decompose as dec  # noqa: E402


def _make_white_studio_frame(h=360, w=360):
    """Genre 1: Clean bright white studio cyclorama with a soft neutral grey cylinder."""
    img = np.full((h, w, 3), 232, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - w * 0.5, yy - h * 0.52)
    mask = dist <= 68
    shade = np.clip(140 + 55 * ((xx - w * 0.45) / 68.0), 85, 210)
    for c in range(3):
        img[:, :, c] = np.where(mask, shade.astype(np.uint8), img[:, :, c])
    return img


def _make_black_void_product_frame(h=360, w=360):
    """Genre 2: Floating neutral metallic/matte product on pure black studio void (no emissive, no fog)."""
    img = np.full((h, w, 3), 3, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    mask = (np.abs(xx - w * 0.5) <= 62) & (np.abs(yy - h * 0.5) <= 46)
    shade = np.clip(70 + 80 * ((xx - (w * 0.5 - 62)) / 124.0), 40, 175)
    for c in range(3):
        img[:, :, c] = np.where(mask, shade.astype(np.uint8), img[:, :, c])
    return img


def _make_flat_npr_frame(h=360, w=360):
    """Genre 3: Clean flat 2D/NPR cel-shaded scene (sharp blocks, no glow, no fog)."""
    img = np.full((h, w, 3), (210, 195, 170), dtype=np.uint8)
    cv.rectangle(img, (40, 60), (170, 290), (95, 130, 195), -1)
    cv.rectangle(img, (190, 90), (320, 280), (130, 170, 120), -1)
    cv.circle(img, (180, 180), 45, (185, 145, 90), -1)
    return img


def _make_emissive_spill_scene(h=360, w=360, with_gi_and_fog=True):
    """Genre 4: Dim interior/street with localized colored luminous fixtures.
    When with_gi_and_fog=True: casts chromatic indirect bounce + smooth radial bloom/fog + ambient walls.
    When with_gi_and_fog=False: sharp thin emissive bars floating in crushed pitch-black void (no-RT EEVEE artifact).
    """
    lin = np.full((h, w, 3), 0.022 if with_gi_and_fog else 0.002, dtype=np.float32)
    # Subtle textured walls (lit by bounce when with_gi_and_fog, crushed black when False)
    if with_gi_and_fog:
        lin[:, :, 0] += 0.018
        lin[:, :, 1] += 0.010
        lin[:, :, 2] += 0.006

    # Two localized colored luminous bars (warm amber-red and cyan)
    bars = [
        ((70, 110, 155, 126), np.array([1.0, 0.30, 0.08], dtype=np.float32)),
        ((205, 165, 290, 181), np.array([0.12, 0.78, 1.0], dtype=np.float32)),
    ]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    for (x0, y0, x1, y1), col in bars:
        lin[y0:y1, x0:x1] = col * 0.92
        if with_gi_and_fog:
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            dx = np.maximum(0.0, np.maximum(x0 - xx, xx - x1))
            dy = np.maximum(0.0, np.maximum(y0 - yy, yy - y1))
            dist = np.hypot(dx, dy)
            near_glow = 0.24 * np.exp(-dist / 7.5)
            far_spill = 0.095 * np.exp(-dist / 26.0)
            for ch in range(3):
                lin[:, :, ch] += col[ch] * (near_glow + far_spill)

    srgb = np.clip(np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(srgb, cv.COLOR_RGB2BGR)


def _make_contrast_vs_washed_frame(h=360, w=360, washed_flat=False):
    """Genre 5: High-contrast moody surface with diagonal glossy streaks vs flat-washed fill."""
    lin = np.full((h, w, 3), 0.055 if washed_flat else 0.024, dtype=np.float32)
    # Upper-left directional highlight & lower-right diagonal glossy reflection streaks
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    diag = (xx * 0.7 + yy * 0.7)
    if not washed_flat:
        # Crisp diagonal specular bands in bottom-right & directional key in top-left
        streak = 0.20 * np.exp(-((diag - 340.0) ** 2) / (2.0 * 9.0 ** 2))
        streak += 0.16 * np.exp(-((diag - 395.0) ** 2) / (2.0 * 7.0 ** 2))
        mask_br = (xx > w * 0.45) & (yy > h * 0.52)
        for ch, gain in enumerate((1.0, 0.85, 0.65)):
            lin[:, :, ch] += np.where(mask_br, streak * gain, 0.0)
        key_spot = 0.22 * np.exp(-(((xx - 90.0) ** 2 + (yy - 90.0) ** 2) / (2.0 * 35.0 ** 2)))
        lin[:, :, 0] += key_spot
        lin[:, :, 1] += key_spot * 0.9
        lin[:, :, 2] += key_spot * 0.8
    else:
        # Giant flat fill washes out contrast everywhere
        lin[:, :, :] = 0.058 + 0.012 * np.sin(xx / 40.0)[:, :, None] * 0.0
        lin[:, :, 0] += 0.012 * ((xx / w))
        lin[:, :, 1] += 0.011 * ((xx / w))
        lin[:, :, 2] += 0.010 * ((xx / w))
    srgb = np.clip(np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(srgb, cv.COLOR_RGB2BGR)


def _wrap_scene3d_json(bgr, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_dir / "frame_000000.png"), bgr)
    cam = s3.analyze_camera_and_projection(bgr)
    mat = s3.analyze_material_and_shading(bgr)
    light = s3.analyze_lighting_and_volumetrics(bgr)
    opt = s3.analyze_optics_and_motion_3d([bgr, bgr], center_idx=0, cam_info=cam, light_info=light)
    gates = s3.build_lookdev_feature_gates(cam, mat, light, opt)
    payload = {
        "target_frame": 0,
        "pillar_1_camera_and_projection": cam,
        "pillar_3_material_and_bsdf": mat,
        "pillar_4_lighting_and_volumetrics": light,
        "pillar_5_and_6_motion_and_optics": opt,
        "lookdev_feature_gates": gates,
    }
    p = out_dir / "scene3d.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


class Scene3DMultiGenreTests(unittest.TestCase):
    def test_white_studio_has_zero_night_or_spill_false_positives(self):
        img = _make_white_studio_frame()
        li = s3.analyze_lighting_and_volumetrics(img)
        self.assertFalse(li["radial_halo_profile"]["bloom_detected"])
        self.assertFalse(li["radial_halo_profile"]["volume_fog_detected"])
        self.assertFalse(li["emissive_spill_profile"]["emissive_indirect_spill_detected"])
        self.assertFalse(li["recommended_blender_render_requirements"]["require_cycles"])
        self.assertEqual(li["recommended_blender_render_requirements"]["compositor_glare_tier"], "NONE")

    def test_black_void_product_does_not_false_trigger_crushed_black(self):
        orig = _make_black_void_product_frame()
        rend = _make_black_void_product_frame()
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            o_json = _wrap_scene3d_json(orig, tp / "orig")
            r_json = _wrap_scene3d_json(rend, tp / "rend")
            rep = s3.compare_scene3d_pair(o_json, r_json, out_dir=tp / "cmp")
            failed_metrics = {c["metric"] for c in rep["failed_checks"]}
            self.assertNotIn("non_core_crushed_black_ratio", failed_metrics)
            self.assertNotIn("enable_emissive_indirect_spill", failed_metrics)
            self.assertEqual(rep["verdict"], "PASS")

    def test_flat_npr_does_not_force_volumetrics_or_cycles(self):
        img = _make_flat_npr_frame()
        li = s3.analyze_lighting_and_volumetrics(img)
        self.assertFalse(li["radial_halo_profile"]["volume_fog_detected"])
        self.assertFalse(li["emissive_spill_profile"]["emissive_indirect_spill_detected"])
        self.assertFalse(li["recommended_blender_render_requirements"]["require_cycles"])

    def test_emissive_spill_and_fog_catches_no_gi_degradation(self):
        orig = _make_emissive_spill_scene(with_gi_and_fog=True)
        degraded = _make_emissive_spill_scene(with_gi_and_fog=False)

        orig_li = s3.analyze_lighting_and_volumetrics(orig)
        self.assertTrue(orig_li["radial_halo_profile"]["bloom_detected"])
        self.assertTrue(orig_li["radial_halo_profile"]["volume_fog_detected"])
        self.assertTrue(orig_li["emissive_spill_profile"]["emissive_indirect_spill_detected"])
        self.assertTrue(orig_li["recommended_blender_render_requirements"]["require_cycles"])
        self.assertEqual(orig_li["recommended_blender_render_requirements"]["compositor_glare_tier"], "DUAL_NEAR_AND_BROAD")

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            o_json = _wrap_scene3d_json(orig, tp / "orig")
            r_json = _wrap_scene3d_json(degraded, tp / "rend")
            rep = s3.compare_scene3d_pair(o_json, r_json, out_dir=tp / "cmp")
            failed = {c["metric"] for c in rep["failed_checks"]}
            self.assertEqual(rep["verdict"], "FAIL")
            self.assertIn("enable_emissive_indirect_spill", failed)
            self.assertIn("emissive_spill_chromatic_energy", failed)
            self.assertIn("non_core_crushed_black_ratio", failed)
            self.assertIn("near_bloom_linear_lum", failed)

    def test_regional_contrast_catches_flat_washed_lighting(self):
        orig = _make_contrast_vs_washed_frame(washed_flat=False)
        washed = _make_contrast_vs_washed_frame(washed_flat=True)
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            o_json = _wrap_scene3d_json(orig, tp / "orig")
            r_json = _wrap_scene3d_json(washed, tp / "rend")
            rep = s3.compare_scene3d_pair(o_json, r_json, out_dir=tp / "cmp")
            failed = {c["metric"] for c in rep["failed_checks"]}
            self.assertEqual(rep["verdict"], "FAIL")
            self.assertIn("regional_tonal_contrast", failed)

    def test_lookdev_keyframe_selector_hits_emissive_peak_and_low_key(self):
        # Synthetic 90-frame multi-regime sequence:
        # 0..29: bright white studio | 30..59: peak emissive scene (peak at f45) | 60..89: low-key moody frame
        records = []
        f_white = _make_white_studio_frame(120, 120)
        f_emissive = _make_emissive_spill_scene(120, 120, with_gi_and_fog=True)
        f_dark = _make_contrast_vs_washed_frame(120, 120, washed_flat=False)
        for idx in range(90):
            img = f_white if idx < 30 else (f_emissive if idx < 60 else f_dark)
            em, con, lk, p50 = dec._frame_optical_signature(img)
            records.append(dict(frame=idx, time_s=idx / 30.0, emissive_score=em, contrast_score=con, low_key_score=lk, p50_midtone=p50))
        kfs = dec._select_lookdev_keyframes(records, fps=30.0)
        roles = {k["role"] for k in kfs}
        self.assertIn("PEAK_EMISSIVE_OR_HIGHLIGHT", roles)
        em_frames = [k["frame"] for k in kfs if k["role"] == "PEAK_EMISSIVE_OR_HIGHLIGHT"]
        self.assertTrue(30 <= em_frames[0] < 60)
        self.assertTrue(any(k["frame"] >= 60 for k in kfs))


if __name__ == "__main__":
    unittest.main()
