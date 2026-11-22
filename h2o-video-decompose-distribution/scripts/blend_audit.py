#!/usr/bin/env python3
"""Headless static audit of a Blender (.blend) project against source scene3d.json requirements.

Enforces the No-Speed-Compromise Rendering Policy (Scheme A):
- Physically required multi-bounce emissive GI, volumetric scattering, or complex glossy/wet
  reflections require Cycles with adequate bounces and denoising/samples.
- Standard PBR/NPR scenes without those may use Cycles or EEVEE with raytracing + GI/probes.
- EEVEE with use_raytracing=False is strictly forbidden for final 3D delivery.
- Scale-invariant lighting rig checks avoid overfitting to any specific scene size or genre.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

_BLENDER_INSPECTOR_PY = r'''
import bpy
import json
import math
import sys
from mathutils import Vector

def fmt_val(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, str):
        return val
    try:
        return [round(float(x), 4) for x in val]
    except Exception:
        return str(val)

def iter_fcurves(anim_data):
    if anim_data is None or anim_data.action is None:
        return
    action = anim_data.action
    if hasattr(action, "fcurves") and action.fcurves:
        for fc in action.fcurves:
            yield fc
        return
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in getattr(layer, "strips", []):
                for bag in getattr(strip, "channelbags", []):
                    for fc in getattr(bag, "fcurves", []):
                        yield fc

def walk_nodes(node_tree, visited=None):
    if node_tree is None:
        return
    if visited is None:
        visited = set()
    ptr = node_tree.as_pointer()
    if ptr in visited:
        return
    visited.add(ptr)
    for node in node_tree.nodes:
        yield node, node_tree
        if getattr(node, "node_tree", None) is not None and not node.mute:
            yield from walk_nodes(node.node_tree, visited)

def get_compositor_tree(scene):
    if hasattr(scene, "compositing_node_group") and scene.compositing_node_group is not None:
        return scene.compositing_node_group
    if getattr(scene, "use_nodes", False) and getattr(scene, "node_tree", None) is not None:
        return scene.node_tree
    return None

def has_active_volume(node_tree):
    if node_tree is None:
        return []
    found = []
    for node, _ in walk_nodes(node_tree):
        if node.mute:
            continue
        if node.type in ("VOLUME_PRINCIPLED", "PRINCIPLED_VOLUME", "VOLUME_SCATTER") or node.bl_idname in (
            "ShaderNodeVolumePrincipled", "ShaderNodeVolumeScatter"
        ):
            dens_in = node.inputs.get("Density")
            density = 0.0
            linked = False
            if dens_in is not None:
                linked = bool(dens_in.is_linked)
                if not linked:
                    density = float(dens_in.default_value)
            if linked or density > 1e-5:
                aniso_in = node.inputs.get("Anisotropy")
                aniso = float(aniso_in.default_value) if (aniso_in and not aniso_in.is_linked) else None
                found.append({
                    "node_type": node.bl_idname,
                    "density": round(density, 5) if not linked else "linked",
                    "anisotropy": round(aniso, 3) if aniso is not None else "linked",
                })
    return found

def inspect_scene(scene):
    r = scene.render
    vm = scene.view_settings
    info = {
        "name": scene.name,
        "engine": r.engine,
        "resolution": [r.resolution_x, r.resolution_y, r.resolution_percentage],
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "use_motion_blur": bool(r.use_motion_blur),
        "motion_blur_shutter": round(float(r.motion_blur_shutter), 4),
        "use_compositing": bool(r.use_compositing),
        "view_transform": vm.view_transform,
        "look": vm.look,
        "exposure": round(float(vm.exposure), 4),
    }

    if r.engine == "CYCLES":
        c = scene.cycles
        info["cycles"] = {
            "samples": int(c.samples),
            "use_adaptive_sampling": bool(c.use_adaptive_sampling),
            "use_denoising": bool(c.use_denoising),
            "max_bounces": int(c.max_bounces),
            "diffuse_bounces": int(c.diffuse_bounces),
            "glossy_bounces": int(c.glossy_bounces),
            "transmission_bounces": int(c.transmission_bounces),
            "volume_bounces": int(c.volume_bounces),
            "transparent_max_bounces": int(c.transparent_max_bounces),
        }
    else:
        e = scene.eevee
        info["eevee"] = {
            "taa_render_samples": int(getattr(e, "taa_render_samples", 64)),
            "use_raytracing": bool(getattr(e, "use_raytracing", False)),
            "use_fast_gi": bool(getattr(e, "use_fast_gi", False)),
            "use_shadows": bool(getattr(e, "use_shadows", True)),
            "use_volumetric_shadows": bool(getattr(e, "use_volumetric_shadows", False)),
        }

    # Active camera & DOF
    cam_obj = scene.camera
    if cam_obj and cam_obj.type == "CAMERA" and cam_obj.data:
        cd = cam_obj.data
        info["camera"] = {
            "name": cam_obj.name,
            "type": cd.type,
            "lens": round(float(cd.lens), 3),
            "use_dof": bool(cd.dof.use_dof),
            "focus_distance": round(float(cd.dof.focus_distance), 3),
            "fstop": round(float(cd.dof.aperture_fstop), 3),
            "location": [round(float(x), 3) for x in cam_obj.matrix_world.translation],
        }
        cam_loc = cam_obj.matrix_world.translation.copy()
    else:
        info["camera"] = None
        cam_loc = Vector((0.0, 0.0, 0.0))

    # Active light probes
    probes = []
    for obj in scene.objects:
        if obj.type == "LIGHT_PROBE" and not obj.hide_render:
            probes.append({
                "name": obj.name,
                "probe_type": obj.data.type if obj.data else "UNKNOWN",
            })
    info["light_probes"] = probes

    # Reference scale: median distance from camera to render-visible mesh objects (excluding huge backdrop domes)
    mesh_sizes = []
    mesh_dists = []
    mesh_entries = []
    for obj in scene.objects:
        if obj.type == "MESH" and not obj.hide_render and obj.data and len(obj.data.vertices) > 0:
            dims = obj.dimensions
            diag = float(math.sqrt(dims.x**2 + dims.y**2 + dims.z**2))
            if diag > 1e-4:
                center = obj.matrix_world.translation
                dist = float((center - cam_loc).length)
                mesh_sizes.append(diag)
                mesh_entries.append((diag, dist))
    if mesh_sizes:
        mesh_sizes_sorted = sorted(mesh_sizes)
        med_size = mesh_sizes_sorted[len(mesh_sizes_sorted) // 2]
        for diag, dist in mesh_entries:
            if diag <= max(10.0, med_size * 8.0) and dist > 0.1:
                mesh_dists.append(dist)
    cam_ref_dist = sorted(mesh_dists)[len(mesh_dists) // 2] if mesh_dists else 10.0
    info["scene_scale_ref_dist"] = round(float(cam_ref_dist), 3)

    # Lights
    lights = []
    for obj in scene.objects:
        if obj.type == "LIGHT" and not obj.hide_render and obj.data:
            ld = obj.data
            linfo = {
                "name": obj.name,
                "type": ld.type,
                "energy": round(float(ld.energy), 3),
                "color": [round(float(c), 3) for c in ld.color],
                "use_shadow": bool(ld.use_shadow),
                "diffuse_factor": round(float(getattr(ld, "diffuse_factor", 1.0)), 3),
                "specular_factor": round(float(getattr(ld, "specular_factor", 1.0)), 3),
            }
            if ld.type == "AREA":
                s1 = float(ld.size)
                s2 = float(ld.size_y) if ld.shape in ("RECTANGLE", "ELLIPSE") else s1
                linfo["size_max"] = round(max(s1, s2), 3)
                linfo["size_to_ref_dist_ratio"] = round(max(s1, s2) / max(0.5, cam_ref_dist), 3)
            lights.append(linfo)
    info["lights"] = lights

    # Volumes (World + render-visible objects)
    volumes = []
    if scene.world and scene.world.node_tree:
        for v in has_active_volume(scene.world.node_tree):
            v["source"] = f"World:{scene.world.name}"
            volumes.append(v)

    used_mats = set()
    for obj in scene.objects:
        if obj.type in ("MESH", "CURVE", "FONT", "SURFACE", "VOLUME") and not obj.hide_render:
            for slot in obj.material_slots:
                if slot.material:
                    used_mats.add(slot.material)
                    if slot.material.node_tree:
                        for v in has_active_volume(slot.material.node_tree):
                            v["source"] = f"Object:{obj.name} ({slot.material.name})"
                            volumes.append(v)
    info["volumes"] = volumes

    # Materials: emissive count, suspicious emission-mix cheats, high clearcoat count
    emissive_count = 0
    emission_mix_cheats = []
    high_clearcoat_mats = []
    for mat in sorted(used_mats, key=lambda m: m.name):
        if not mat.node_tree:
            continue
        is_emissive = False
        has_principled = False
        has_emission_node = False
        for node, _ in walk_nodes(mat.node_tree):
            if node.mute:
                continue
            if node.type == "BSDF_PRINCIPLED":
                has_principled = True
                es = node.inputs.get("Emission Strength")
                if es is not None and (es.is_linked or float(es.default_value) > 0.01):
                    is_emissive = True
                cw = node.inputs.get("Coat Weight")
                cr = node.inputs.get("Coat Roughness")
                if cw is not None and cr is not None and not cw.is_linked and not cr.is_linked:
                    if float(cw.default_value) >= 0.2 and float(cr.default_value) <= 0.05:
                        high_clearcoat_mats.append({
                            "material": mat.name,
                            "coat_weight": round(float(cw.default_value), 3),
                            "coat_roughness": round(float(cr.default_value), 3),
                        })
            elif node.type == "EMISSION":
                has_emission_node = True
                st = node.inputs.get("Strength")
                if st is not None and (st.is_linked or float(st.default_value) > 0.01):
                    is_emissive = True
            elif node.type == "MIX_SHADER":
                fac = node.inputs.get("Fac")
                if fac is not None and not fac.is_linked and float(fac.default_value) > 0.15:
                    s1 = node.inputs[1].links[0].from_node.type if node.inputs[1].is_linked else ""
                    s2 = node.inputs[2].links[0].from_node.type if node.inputs[2].is_linked else ""
                    if ("BSDF_PRINCIPLED" in (s1, s2)) and ("EMISSION" in (s1, s2)):
                        if (s2 == "EMISSION" and float(fac.default_value) > 0.15) or (
                            s1 == "EMISSION" and float(fac.default_value) < 0.85
                        ):
                            emission_mix_cheats.append({
                                "material": mat.name,
                                "mix_factor": round(float(fac.default_value), 3),
                            })
        if is_emissive:
            emissive_count += 1
    info["material_stats"] = {
        "used_count": len(used_mats),
        "emissive_count": emissive_count,
        "emission_mix_cheats": emission_mix_cheats[:15],
        "high_clearcoat_mats": high_clearcoat_mats[:15],
    }

    # Compositor (recursive through node groups + animation check)
    comp = get_compositor_tree(scene)
    glare_nodes = []
    blur_nodes = []
    colorspace_transforms = []
    animated_comp_paths = []
    if comp is not None:
        for fc in iter_fcurves(comp.animation_data):
            if len(fc.keyframe_points) >= 2:
                vals = [kp.co[1] for kp in fc.keyframe_points]
                if max(vals) - min(vals) > 1e-4:
                    animated_comp_paths.append(fc.data_path)
        for node, _ in walk_nodes(comp):
            if node.mute:
                continue
            if node.type == "GLARE":
                g = {"name": node.name, "label": node.label}
                for key in ("Type", "Threshold", "Strength", "Size", "Saturation"):
                    inp = node.inputs.get(key)
                    if inp is not None:
                        g[key.lower()] = fmt_val(inp.default_value) if not inp.is_linked else "linked"
                if "type" not in g and hasattr(node, "glare_type"):
                    g["type"] = node.glare_type
                if "threshold" not in g and hasattr(node, "threshold"):
                    g["threshold"] = round(float(node.threshold), 4)
                glare_nodes.append(g)
            elif node.type == "BLUR":
                sz = node.inputs.get("Size")
                blur_nodes.append({
                    "name": node.name,
                    "label": node.label,
                    "size": fmt_val(sz.default_value) if (sz and not sz.is_linked) else getattr(node, "size_x", None),
                })
            elif node.type in ("CONVERT_COLORSPACE", "CONVERT_COLOR_SPACE"):
                to_cs = getattr(node, "to_color_space", None)
                colorspace_transforms.append({"name": node.name, "to": str(to_cs)})
    info["compositor"] = {
        "has_tree": comp is not None,
        "glare_nodes": glare_nodes,
        "blur_nodes": blur_nodes,
        "colorspace_transforms": colorspace_transforms,
        "animated_paths_count": len(animated_comp_paths),
        "animated_paths_sample": animated_comp_paths[:15],
    }
    return info

def select_target_scenes(requested):
    if requested:
        return [bpy.data.scenes[name] for name in requested if name in bpy.data.scenes]
    # Auto-detect: if active scene is a VSE master with Scene Strips, collect referenced 3D scenes
    active = bpy.context.scene
    vse_scenes = []
    if active and active.sequence_editor:
        strips = getattr(active.sequence_editor, "strips", None) or getattr(active.sequence_editor, "sequences", [])
        seen = set()
        for st in strips:
            sc = getattr(st, "scene", None)
            if sc is not None and sc.name not in seen and sc.name != active.name:
                seen.add(sc.name)
                vse_scenes.append(sc)
    if vse_scenes:
        return vse_scenes
    # Otherwise inspect scenes that actually have a camera and meshes
    three_d = [
        s for s in bpy.data.scenes
        if s.camera is not None and any(o.type == "MESH" and not o.hide_render for o in s.objects)
    ]
    return three_d if three_d else ([active] if active else list(bpy.data.scenes))

if __name__ == "__main__":
    argv = sys.argv
    idx = argv.index("--")
    out_path = argv[idx + 1]
    requested_scenes = argv[idx + 2:]
    targets = select_target_scenes(requested_scenes)
    payload = {
        "blend_file": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "all_scenes": [s.name for s in bpy.data.scenes],
        "audited_scenes": [inspect_scene(s) for s in targets],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
'''


def find_blender_executable(explicit: str | Path | None = None) -> Path:
    """Locate Blender binary across macOS, Windows, and PATH."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified Blender executable not found: {p}")

    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).exists():
        return Path(env_bin)

    which = shutil.which("blender")
    if which:
        return Path(which)

    candidates: list[Path] = []
    system = platform.system().lower()
    if system == "darwin":
        candidates.extend([
            Path("/opt/homebrew/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        ])
    elif system == "windows":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        bf = pf / "Blender Foundation"
        if bf.exists():
            for child in sorted(bf.glob("Blender*"), reverse=True):
                candidates.append(child / "blender.exe")
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Blender executable not found in PATH or standard install locations. Pass --blender /path/to/blender."
    )


def inspect_blend_metadata(
    blend_path: str | Path,
    scene_names: list[str] | None = None,
    blender_bin: str | Path | None = None,
    timeout_s: int = 90,
) -> dict:
    """Run headless Blender to extract scene rendering/lighting/compositor metadata."""
    blend_p = Path(blend_path).resolve()
    if not blend_p.exists():
        raise FileNotFoundError(f".blend file not found: {blend_p}")
    exe = find_blender_executable(blender_bin)

    with tempfile.TemporaryDirectory(prefix="h2o_blend_audit_") as tmp:
        tmp_p = Path(tmp)
        script_p = tmp_p / "inspect_blend.py"
        out_p = tmp_p / "blend_meta.json"
        script_p.write_text(_BLENDER_INSPECTOR_PY, encoding="utf-8")
        cmd = [str(exe), "-b", str(blend_p), "--python-exit-code", "1", "--python", str(script_p), "--", str(out_p)]
        if scene_names:
            cmd.extend(scene_names)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0 or not out_p.exists():
            raise RuntimeError(
                f"Blender inspection failed (rc={proc.returncode}).\nSTDOUT:\n{proc.stdout[-800:]}\nSTDERR:\n{proc.stderr[-800:]}"
            )
        return json.loads(out_p.read_text(encoding="utf-8"))


def _aggregate_source_requirements(scene3d_docs: list[dict]) -> dict:
    """Aggregate physical light-transport requirements from one or more source scene3d.json files."""
    if not scene3d_docs:
        return {
            "source_count": 0,
            "is_flat_npr_only": False,
            "require_cycles": False,
            "allow_eevee_no_rt": False,
            "min_diffuse_bounces": 1,
            "min_glossy_bounces": 1,
            "min_transmission_bounces": 1,
            "min_volume_bounces": 0,
            "needs_indirect_gi": False,
            "needs_volume_fog": False,
            "needs_bloom": False,
            "compositor_glare_tier": "NONE",
            "bloom_regime_flips": False,
            "exposure_regime_shift": False,
            "all_require_dof": False,
            "all_require_motion_blur": False,
            "is_high_contrast_directional": False,
            "reasons": [],
        }

    bloom_flags = []
    p50_vals = []
    dof_flags = []
    mb_flags = []
    flat_npr_flags = []
    high_contrast_flags = []
    req_cycles = False
    needs_gi = False
    needs_vol = False
    needs_bloom = False
    glare_tier_rank = {"NONE": 0, "SINGLE_SOFT_GLOW": 1, "DUAL_NEAR_AND_BROAD": 2}
    best_glare_tier = "NONE"
    min_diff = 1
    min_gloss = 1
    min_trans = 1
    min_vol = 0
    reasons: list[str] = []

    for doc in scene3d_docs:
        gates = doc.get("lookdev_feature_gates", {})
        p3 = doc.get("pillar_3_material_and_bsdf", {})
        p4 = doc.get("pillar_4_lighting_and_volumetrics", {})
        rad = p4.get("radial_halo_profile", {})
        req = p4.get("recommended_blender_render_requirements", {})
        lum = p4.get("linear_luminance_percentiles", {})
        shd = p4.get("shadow_and_key_light_hardness", {})

        paradigm = p3.get("shading_paradigm", "PBR_PRINCIPLED_BSDF")
        is_npr_flat = paradigm in ("NPR_TOON_OR_FLAT", "FLAT_2D_UNLIT")
        flat_npr_flags.append(is_npr_flat)

        has_vol = bool(gates.get("enable_volume_fog", False))
        has_bloom = bool(gates.get("enable_bloom_or_soft_glow", False))
        has_spill = bool(gates.get("enable_emissive_indirect_spill", False))
        has_gref = bool(gates.get("enable_ground_reflection", False))
        has_glass = bool(p3.get("special_material_signals", {}).get("transmission_glass_hint", False))
        has_em = bool(gates.get("enable_emissive_materials", False))
        k2f = float(p4.get("key_to_fill_linear_ratio", 2.0))
        hardness = shd.get("estimated_key_light_hardness", "MEDIUM_SOFT_AREA")

        bloom_flags.append(has_bloom)
        dof_flags.append(bool(gates.get("enable_dof", False)))
        mb_flags.append(bool(gates.get("enable_motion_blur", False)))
        p50 = float(lum.get("p50_midtone", 0.05))
        if p50 > 0:
            p50_vals.append(p50)
        if hardness != "BROAD_DIFFUSE_OR_HDRI" and k2f >= 3.0:
            high_contrast_flags.append(True)
        else:
            high_contrast_flags.append(False)

        if has_vol:
            needs_vol = True
            req_cycles = True
            min_vol = max(min_vol, 1)
            reasons.append("原片实测含体积雾散射(enable_volume_fog=True)")
        if has_spill:
            needs_gi = True
            req_cycles = True
            min_diff = max(min_diff, 2)
            min_gloss = max(min_gloss, 2)
            reasons.append("原片实测含自发光周边间接溢光(enable_emissive_indirect_spill=True)")
        if has_gref:
            req_cycles = True
            min_gloss = max(min_gloss, 2)
            reasons.append("原片实测含地面/光泽镜面反射(enable_ground_reflection=True)")
        if has_glass:
            req_cycles = True
            min_trans = max(min_trans, 2)
            min_gloss = max(min_gloss, 2)
            reasons.append("原片实测含透明/玻璃折射反射(transmission_glass_hint=True)")
        if has_em or k2f >= 3.0:
            needs_gi = True

        # Respect explicit machine-readable requirements emitted by upgraded scene3d_core.py
        if req:
            if req.get("require_cycles"):
                req_cycles = True
            min_diff = max(min_diff, int(req.get("min_diffuse_bounces", 1)))
            min_gloss = max(min_gloss, int(req.get("min_glossy_bounces", 1)))
            min_trans = max(min_trans, int(req.get("min_transmission_bounces", 1)))
            min_vol = max(min_vol, int(req.get("min_volume_bounces", 0)))
            tier = req.get("compositor_glare_tier", "NONE")
            if glare_tier_rank.get(tier, 0) > glare_tier_rank.get(best_glare_tier, 0):
                best_glare_tier = tier

        if has_bloom:
            needs_bloom = True
            far_ratio = float(rad.get("far_to_core_ratio", 0.0))
            near_ratio = float(rad.get("near_to_core_ratio", 0.0))
            inferred_tier = (
                "DUAL_NEAR_AND_BROAD"
                if (has_vol or (far_ratio >= 0.030 and near_ratio >= 0.10))
                else "SINGLE_SOFT_GLOW"
            )
            if glare_tier_rank[inferred_tier] > glare_tier_rank[best_glare_tier]:
                best_glare_tier = inferred_tier

    bloom_flips = len(set(bloom_flags)) > 1
    exp_shift = bool(len(p50_vals) >= 2 and (max(p50_vals) / max(1e-4, min(p50_vals))) >= 2.5)

    return {
        "source_count": len(scene3d_docs),
        "is_flat_npr_only": all(flat_npr_flags) if flat_npr_flags else False,
        "require_cycles": req_cycles,
        "allow_eevee_no_rt": False,
        "min_diffuse_bounces": min_diff,
        "min_glossy_bounces": min_gloss,
        "min_transmission_bounces": min_trans,
        "min_volume_bounces": min_vol,
        "needs_indirect_gi": needs_gi,
        "needs_volume_fog": needs_vol,
        "needs_bloom": needs_bloom,
        "compositor_glare_tier": best_glare_tier,
        "bloom_regime_flips": bloom_flips,
        "exposure_regime_shift": exp_shift,
        "all_require_dof": all(dof_flags) if dof_flags else False,
        "all_require_motion_blur": all(mb_flags) if mb_flags else False,
        "is_high_contrast_directional": any(high_contrast_flags),
        "reasons": sorted(set(reasons)),
    }


def evaluate_blend_audit(blend_meta: dict, scene3d_docs: list[dict]) -> dict:
    """Evaluate extracted .blend metadata against aggregated source requirements (Scheme A)."""
    req = _aggregate_source_requirements(scene3d_docs)
    scene_reports = []
    all_checks = []

    for sc in blend_meta.get("audited_scenes", []):
        checks = []
        sname = sc["name"]

        def add(category: str, metric: str, status: str, expected, actual, detail: str, fix_hint: str = ""):
            item = {
                "scene": sname,
                "category": category,
                "metric": metric,
                "status": status,  # PASS | WARN | FAIL
                "expected": expected,
                "actual": actual,
                "detail": detail,
                "fix_hint": fix_hint,
            }
            checks.append(item)
            all_checks.append(item)

        engine = sc.get("engine", "UNKNOWN")
        is_cycles = engine == "CYCLES"
        is_eevee = "EEVEE" in engine

        # 1. Render Pipeline & No-Speed-Compromise Gate (Scheme A)
        if is_eevee:
            ev = sc.get("eevee", {})
            use_rt = bool(ev.get("use_raytracing", False))
            use_gi = bool(ev.get("use_fast_gi", False))
            probes = sc.get("light_probes", [])
            samples = int(ev.get("taa_render_samples", 64))

            if req["require_cycles"]:
                add(
                    "render_pipeline",
                    "engine_fidelity_lock",
                    "FAIL",
                    "CYCLES",
                    engine,
                    "CYCLES_REQUIRED_FOR_LIGHTING_FIDELITY: " + "; ".join(req["reasons"]),
                    "切换至 Cycles（严禁为追求速度降级为 EEVEE 牺牲间接溢光、体积散射或实景反射）",
                )
            elif not use_rt and not req["is_flat_npr_only"]:
                add(
                    "render_pipeline",
                    "eevee_raytracing_enabled",
                    "FAIL",
                    True,
                    False,
                    "FORBIDDEN_EEVEE_NO_RAYTRACING",
                    "禁止在最终 3D 交付中使用关光追的 EEVEE：开启 scene.eevee.use_raytracing = True 或切换至 Cycles",
                )
            else:
                add("render_pipeline", "engine_fidelity_lock", "PASS", "CYCLES_OR_EEVEE_RT", f"{engine}(RT={use_rt})", "OK")

            if not use_rt and req["require_cycles"]:
                add(
                    "render_pipeline",
                    "eevee_raytracing_enabled",
                    "FAIL",
                    True,
                    False,
                    "FORBIDDEN_EEVEE_NO_RAYTRACING",
                    "当前工程关闭了 EEVEE Raytracing，自发光与表面反射链路完全失效；请切回 Cycles",
                )

            if req["needs_indirect_gi"] and (not use_gi) and len(probes) == 0:
                add(
                    "render_pipeline",
                    "eevee_gi_and_probes",
                    "FAIL",
                    "use_fast_gi=True or active LIGHT_PROBE",
                    f"use_fast_gi={use_gi}, probes=0",
                    "FORBIDDEN_EEVEE_NO_GI_OR_PROBES",
                    "无 Fast GI 且无反射/辐照探头导致间接光与反射坍塌；请切回 Cycles 或开启 Fast GI 并布置探头",
                )

            if samples < 16:
                add(
                    "render_pipeline",
                    "render_samples_floor",
                    "FAIL",
                    ">=16 (recommended >=32)",
                    samples,
                    "SPEED_COMPROMISE_LOW_SAMPLES",
                    "提高 EEVEE 渲染采样（scene.eevee.taa_render_samples >= 32），禁止低采样妥协",
                )
            elif samples < 32:
                add(
                    "render_pipeline",
                    "render_samples_floor",
                    "WARN",
                    ">=32",
                    samples,
                    "MODERATE_LOW_SAMPLES",
                    "建议将 EEVEE 正式渲染采样提升至 >= 32 以消除光影与抗锯齿闪烁",
                )
            else:
                add("render_pipeline", "render_samples_floor", "PASS", ">=32", samples, "OK")

        elif is_cycles:
            cy = sc.get("cycles", {})
            add("render_pipeline", "engine_fidelity_lock", "PASS", "CYCLES", "CYCLES", "OK")

            diff_b = int(cy.get("diffuse_bounces", 0))
            gloss_b = int(cy.get("glossy_bounces", 0))
            trans_b = int(cy.get("transmission_bounces", 0))
            vol_b = int(cy.get("volume_bounces", 0))
            samples = int(cy.get("samples", 0))
            denoise = bool(cy.get("use_denoising", False))

            if diff_b < req["min_diffuse_bounces"]:
                add(
                    "render_pipeline",
                    "cycles_diffuse_bounces",
                    "FAIL",
                    f">={req['min_diffuse_bounces']}",
                    diff_b,
                    "INSUFFICIENT_CYCLES_BOUNCES",
                    f"提高 scene.cycles.diffuse_bounces 至 >= {req['min_diffuse_bounces']}，恢复真实间接漫反射",
                )
            else:
                add("render_pipeline", "cycles_diffuse_bounces", "PASS", f">={req['min_diffuse_bounces']}", diff_b, "OK")

            if gloss_b < req["min_glossy_bounces"]:
                add(
                    "render_pipeline",
                    "cycles_glossy_bounces",
                    "FAIL",
                    f">={req['min_glossy_bounces']}",
                    gloss_b,
                    "INSUFFICIENT_CYCLES_BOUNCES",
                    f"提高 scene.cycles.glossy_bounces 至 >= {req['min_glossy_bounces']}，恢复湿面/漆面/玻璃反射",
                )
            else:
                add("render_pipeline", "cycles_glossy_bounces", "PASS", f">={req['min_glossy_bounces']}", gloss_b, "OK")

            if req["min_transmission_bounces"] > 1 and trans_b < req["min_transmission_bounces"]:
                add(
                    "render_pipeline",
                    "cycles_transmission_bounces",
                    "FAIL",
                    f">={req['min_transmission_bounces']}",
                    trans_b,
                    "INSUFFICIENT_CYCLES_BOUNCES",
                    f"提高 scene.cycles.transmission_bounces 至 >= {req['min_transmission_bounces']}",
                )

            if req["min_volume_bounces"] >= 1 and vol_b < req["min_volume_bounces"]:
                add(
                    "render_pipeline",
                    "cycles_volume_bounces",
                    "FAIL",
                    f">={req['min_volume_bounces']}",
                    vol_b,
                    "INSUFFICIENT_CYCLES_BOUNCES",
                    "开启 scene.cycles.volume_bounces >= 1，让光源在体积介质中产生真实散射",
                )

            if samples < 8 or (samples < 16 and not denoise):
                add(
                    "render_pipeline",
                    "render_samples_floor",
                    "FAIL",
                    ">=16 (or >=12 with Denoising)",
                    f"samples={samples}, denoise={denoise}",
                    "SPEED_COMPROMISE_LOW_SAMPLES",
                    "提高 Cycles 采样数并开启 Denoising，禁止以严重噪点换取渲染速度",
                )
            elif samples < 16:
                add(
                    "render_pipeline",
                    "render_samples_floor",
                    "PASS",
                    ">=12 with Denoising",
                    f"samples={samples}, denoise={denoise}",
                    "OK_DENOISED",
                )
            else:
                add("render_pipeline", "render_samples_floor", "PASS", ">=16", f"samples={samples}, denoise={denoise}", "OK")
        else:
            add(
                "render_pipeline",
                "engine_fidelity_lock",
                "FAIL",
                "CYCLES",
                engine,
                "UNSUPPORTED_OR_WORKBENCH_ENGINE",
                "禁止使用 Workbench 等预览引擎出正式 3D 终稿；请切换至 Cycles",
            )

        # 2. Gate-to-Blend Node Parity
        vols = sc.get("volumes", [])
        if req["needs_volume_fog"]:
            if len(vols) == 0:
                add(
                    "gate_parity",
                    "volume_medium_present",
                    "FAIL",
                    "Active Principled Volume / Volume Scatter (Density > 0)",
                    "0 volume nodes",
                    "MISSING_VOLUME_MEDIUM",
                    "原片实测含体积大气散射，须在 World 或受限体积 Mesh 上挂载 Principled Volume / Volume Scatter",
                )
            else:
                add("gate_parity", "volume_medium_present", "PASS", ">=1 active volume", f"{len(vols)} active", "OK")

        comp = sc.get("compositor", {})
        glare_nodes = comp.get("glare_nodes", [])
        blur_nodes = comp.get("blur_nodes", [])
        if req["needs_bloom"]:
            if (not sc.get("use_compositing")) or len(glare_nodes) == 0:
                add(
                    "gate_parity",
                    "compositor_glare_present",
                    "FAIL",
                    "use_compositing=True + active Glare node",
                    f"use_compositing={sc.get('use_compositing')}, glare_count={len(glare_nodes)}",
                    "MISSING_COMPOSITOR_GLARE",
                    "原片实测含近场辉光/柔晕，须在合成器挂载有效的 CompositorNodeGlare",
                )
            else:
                if req["compositor_glare_tier"] == "DUAL_NEAR_AND_BROAD":
                    single_weak = (
                        len(glare_nodes) == 1
                        and len(blur_nodes) == 0
                        and isinstance(glare_nodes[0].get("threshold"), (int, float))
                        and isinstance(glare_nodes[0].get("strength"), (int, float))
                        and float(glare_nodes[0]["threshold"]) >= 0.65
                        and float(glare_nodes[0]["strength"]) <= 0.30
                    )
                    if single_weak:
                        add(
                            "gate_parity",
                            "compositor_glare_depth",
                            "FAIL",
                            "Dual-scale glow (near Fog Glow + broad Bloom/Scatter) or lower threshold",
                            f"1 Glare(Threshold={glare_nodes[0].get('threshold')}, Strength={glare_nodes[0].get('strength')}), 0 Blur",
                            "WEAK_SINGLE_GLARE_FOR_BROAD_HALO",
                            "原片兼具近场辉光与宽场散射，单个高阈值弱 Glare 无法还原光晕层次；请降低阈值并叠加宽场 Bloom 或散射层",
                        )
                    else:
                        add(
                            "gate_parity",
                            "compositor_glare_depth",
                            "PASS",
                            "DUAL_NEAR_AND_BROAD",
                            f"{len(glare_nodes)} Glare, {len(blur_nodes)} Blur",
                            "OK",
                        )
                else:
                    add("gate_parity", "compositor_glare_present", "PASS", ">=1 Glare", f"{len(glare_nodes)} Glare", "OK")

        # Cross-regime dynamic compositor check (WARN to avoid false-killing intentionally split scenes)
        if len(blend_meta.get("audited_scenes", [])) == 1 and (req["bloom_regime_flips"] or req["exposure_regime_shift"]):
            if len(glare_nodes) > 0 and int(comp.get("animated_paths_count", 0)) == 0:
                add(
                    "gate_parity",
                    "cross_regime_compositor_animation",
                    "WARN",
                    "Keyframed Glare/Exposure across lighting regime transition",
                    "0 animated compositor paths",
                    "STATIC_GLARE_ACROSS_LIGHTING_REGIME_CHANGE",
                    "多帧原片跨越了明暗或辉光开关工况，但合成器 Glare/曝光无任何关键帧；请按景别衰减辉光或拆分为独立灯光场景",
                )

        # Camera DOF & Motion Blur consistency (only WARN if missing, unless every source frame agrees)
        cam = sc.get("camera")
        if req["all_require_dof"] and cam is not None and not cam.get("use_dof"):
            add(
                "gate_parity",
                "camera_dof_enabled",
                "WARN",
                True,
                False,
                "DOF_DISABLED_IN_CAMERA",
                "原片所有对照帧均检出景深差，建议在活跃相机开启 dof.use_dof 并设置实测对焦距离",
            )
        if req["all_require_motion_blur"] and not sc.get("use_motion_blur"):
            add(
                "gate_parity",
                "motion_blur_enabled",
                "WARN",
                True,
                False,
                "MOTION_BLUR_DISABLED",
                "原片所有对照帧均检出运动拖影，建议开启 scene.render.use_motion_blur 并匹配快门",
            )

        # 3. Scale-Invariant Lighting Rig Sanity (WARN only — never false-kills legitimate soft skies)
        lights = sc.get("lights", [])
        if req["is_high_contrast_directional"] and lights:
            total_diff_energy = sum(
                float(l["energy"]) * float(l.get("diffuse_factor", 1.0)) for l in lights
            )
            oversized = [
                l for l in lights
                if l.get("type") == "AREA"
                and float(l.get("size_to_ref_dist_ratio", 0.0)) >= 0.65
                and float(l.get("diffuse_factor", 1.0)) >= 0.5
            ]
            oversized_energy = sum(
                float(l["energy"]) * float(l.get("diffuse_factor", 1.0)) for l in oversized
            )
            oversized_share = oversized_energy / max(1e-3, total_diff_energy)
            # Require >= 2 oversized flat-fill area lamps dominating >= 55% of diffuse lamp wattage
            # (a single soft ambient/overcast sky dome won't trigger)
            if len(oversized) >= 2 and oversized_share >= 0.55:
                names = ", ".join(f"{l['name']}(size={l['size_max']},E={l['energy']})" for l in oversized[:5])
                add(
                    "lighting_rig_sanity",
                    "oversized_flat_fill_lamps",
                    "WARN",
                    "Localized proportionate lighting matching source contrast",
                    f"{len(oversized)} oversized AREA lamps ({oversized_share:.0%} diffuse wattage): {names}",
                    "OVERSIZED_FLAT_FILL_LAMPS",
                    "原片为高反差定向/局域光照，但场景存在多盏远超景深尺度的超大面积补光主导漫反射能量，易洗平暗部与立体感；请缩减大面光并依靠真实实景灯与间接弹射",
                )

        # 4. Material Sanity
        mstats = sc.get("material_stats", {})
        if not req["is_flat_npr_only"] and mstats.get("emission_mix_cheats"):
            sample_m = ", ".join(
                f"{m['material']}(fac={m['mix_factor']})" for m in mstats["emission_mix_cheats"][:5]
            )
            add(
                "material_sanity",
                "non_emissive_emission_mix",
                "WARN",
                "No MixShader(Principled, Emission, >0.15) lighting cheats",
                sample_m,
                "SUSPECT_NON_EMISSIVE_EMISSION_CHEAT",
                "核查受光材质是否混入大比例 Emission 充当亮度补偿；非自发光面须靠真实灯光与曝光塑形",
            )
        if is_eevee and (not sc.get("eevee", {}).get("use_raytracing", False)) and len(sc.get("light_probes", [])) == 0:
            if mstats.get("high_clearcoat_mats"):
                sample_c = ", ".join(m["material"] for m in mstats["high_clearcoat_mats"][:5])
                add(
                    "material_sanity",
                    "clearcoat_reflection_support",
                    "WARN",
                    "Raytraced reflections or Reflection Probes for clearcoat materials",
                    f"no-RT EEVEE, 0 probes: {sample_c}",
                    "CLEARCOAT_WITHOUT_RAYTRACED_REFLECTIONS",
                    "清漆/高光漆面在无光追且无探头下无法反射真实环境、易显塑料感；请切回 Cycles 或启用光追/探头",
                )

        sc_fails = [c for c in checks if c["status"] == "FAIL"]
        sc_warns = [c for c in checks if c["status"] == "WARN"]
        scene_reports.append({
            "scene": sname,
            "engine": engine,
            "verdict": "PASS" if not sc_fails else "FAIL",
            "fail_count": len(sc_fails),
            "warn_count": len(sc_warns),
            "checks": checks,
        })

    fails = [c for c in all_checks if c["status"] == "FAIL"]
    warns = [c for c in all_checks if c["status"] == "WARN"]
    passes = [c for c in all_checks if c["status"] == "PASS"]
    verdict = "PASS" if len(fails) == 0 else "FAIL"

    return {
        "verdict": verdict,
        "policy": "SCHEME_A_PHYSICS_DRIVEN_NO_SPEED_COMPROMISE",
        "blend_file": blend_meta.get("blend_file"),
        "blender_version": blend_meta.get("blender_version"),
        "source_requirements": req,
        "summary": {
            "audited_scene_count": len(scene_reports),
            "total_checks": len(all_checks),
            "pass_count": len(passes),
            "warn_count": len(warns),
            "fail_count": len(fails),
        },
        "scenes": scene_reports,
        "failed_checks": fails,
        "warning_checks": warns,
    }


def _render_markdown(report: dict) -> str:
    s = report["summary"]
    req = report["source_requirements"]
    lines = [
        f"# Blender 工程静态审计报告 — `{report['verdict']}`",
        "",
        f"- **工程文件**：`{report['blend_file']}`",
        f"- **Blender 版本**：`{report['blender_version']}`",
        f"- **策略**：`{report['policy']}`",
        f"- **审计汇总**：场景 `{s['audited_scene_count']}` 个 ｜ `PASS {s['pass_count']}` / `WARN {s['warn_count']}` / **`FAIL {s['fail_count']}`**",
        f"- **原片物理光传输需求**：`require_cycles={req['require_cycles']}`、`min_bounces(diff={req['min_diffuse_bounces']}, glossy={req['min_glossy_bounces']}, vol={req['min_volume_bounces']})`、`volume_fog={req['needs_volume_fog']}`、`glare_tier={req['compositor_glare_tier']}`",
        "",
        "## 必修正硬伤（FAIL）",
        "",
    ]
    if report["failed_checks"]:
        lines.append("| 场景 | 类别 | 检查项 | 要求 | 工程实测 | 问题 | 修正建议 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for c in report["failed_checks"]:
            lines.append(
                f"| `{c['scene']}` | `{c['category']}` | `{c['metric']}` | `{c['expected']}` | `{c['actual']}` | {c['detail']} | {c['fix_hint']} |"
            )
    else:
        lines.append("无 FAIL 项，渲染与物理光传输管线符合原片要求。")

    if report["warning_checks"]:
        lines.extend([
            "",
            "## 风险提示与人工复核项（WARN）",
            "",
            "| 场景 | 类别 | 检查项 | 工程实测 | 说明 | 建议 |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for c in report["warning_checks"]:
            lines.append(
                f"| `{c['scene']}` | `{c['category']}` | `{c['metric']}` | `{c['actual']}` | {c['detail']} | {c['fix_hint']} |"
            )
    return "\n".join(lines) + "\n"


def audit_blend_file(
    blend_path: str | Path,
    scene3d_paths: list[str | Path] | None = None,
    out_dir: str | Path | None = None,
    scene_names: list[str] | None = None,
    blender_bin: str | Path | None = None,
) -> dict:
    """Full end-to-end headless .blend audit returning JSON report and writing optional artifacts."""
    docs = []
    for p in (scene3d_paths or []):
        docs.append(json.loads(Path(p).read_text(encoding="utf-8-sig")))
    meta = inspect_blend_metadata(blend_path, scene_names=scene_names, blender_bin=blender_bin)
    report = evaluate_blend_audit(meta, docs)
    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "blend_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_p / "blend_audit.md").write_text(_render_markdown(report), encoding="utf-8")
    return report
