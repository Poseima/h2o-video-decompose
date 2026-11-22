"""Unit tests for headless .blend static auditing (Scheme A: Physics-Driven, No Speed Compromise)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import blend_audit as ba  # noqa: E402

_BUILD_TEST_BLENDS_PY = r'''
import bpy
import sys
from pathlib import Path

out_dir = Path(sys.argv[sys.argv.index("--") + 1])

def reset_empty():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def add_basic_scene_objects(scene):
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
    cam_data = bpy.data.cameras.new("Cam")
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0, -8, 2)
    scene.camera = cam_obj

# 1. Degraded no-RT EEVEE blend (violates Scheme A on emissive+volume scene)
reset_empty()
s = bpy.context.scene
s.name = "Degraded_EEVEE"
add_basic_scene_objects(s)
s.render.engine = "BLENDER_EEVEE"
s.eevee.use_raytracing = False
s.eevee.use_fast_gi = False
s.eevee.taa_render_samples = 12
for idx, (loc, energy, sz) in enumerate([((0, 0, 12), 1500.0, 22.0), ((4, 0, 10), 1400.0, 20.0)]):
    ld = bpy.data.lights.new(f"GiantFill_{idx}", type="AREA")
    ld.energy = energy
    ld.size = sz
    lo = bpy.data.objects.new(f"GiantFill_{idx}", ld)
    lo.location = loc
    s.collection.objects.link(lo)
s.render.use_compositing = True
tree = bpy.data.node_groups.new("CompBad", "CompositorNodeTree")
s.compositing_node_group = tree
tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = tree.nodes.new("CompositorNodeRLayers")
gl = tree.nodes.new("CompositorNodeGlare")
gl.inputs["Type"].default_value = "Fog Glow"
gl.inputs["Threshold"].default_value = 0.75
gl.inputs["Strength"].default_value = 0.25
out_n = tree.nodes.new("NodeGroupOutput")
tree.links.new(rl.outputs["Image"], gl.inputs["Image"])
tree.links.new(gl.outputs["Image"], out_n.inputs["Image"])
bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "degraded_eevee.blend"))

# 2. Compliant Cycles blend (Cycles + bounces + volume + dual glare)
reset_empty()
s = bpy.context.scene
s.name = "Compliant_Cycles"
add_basic_scene_objects(s)
s.render.engine = "CYCLES"
s.cycles.samples = 24
s.cycles.use_denoising = True
s.cycles.diffuse_bounces = 3
s.cycles.glossy_bounces = 3
s.cycles.volume_bounces = 1
bpy.ops.mesh.primitive_cube_add(size=12, location=(0, 0, 4))
vol_obj = bpy.context.active_object
vol_mat = bpy.data.materials.new("AirVol")
vol_mat.use_nodes = True
nt = vol_mat.node_tree
nt.nodes.clear()
pv = nt.nodes.new("ShaderNodeVolumePrincipled")
pv.inputs["Density"].default_value = 0.02
pv.inputs["Anisotropy"].default_value = 0.65
mo = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(pv.outputs["Volume"], mo.inputs["Volume"])
vol_obj.data.materials.append(vol_mat)
s.render.use_compositing = True
tree = bpy.data.node_groups.new("CompGood", "CompositorNodeTree")
s.compositing_node_group = tree
tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = tree.nodes.new("CompositorNodeRLayers")
g1 = tree.nodes.new("CompositorNodeGlare")
g1.inputs["Type"].default_value = "Fog Glow"
g1.inputs["Threshold"].default_value = 0.28
g1.inputs["Strength"].default_value = 0.60
g2 = tree.nodes.new("CompositorNodeGlare")
g2.inputs["Type"].default_value = "Bloom"
g2.inputs["Threshold"].default_value = 0.35
g2.inputs["Strength"].default_value = 0.35
out_n = tree.nodes.new("NodeGroupOutput")
tree.links.new(rl.outputs["Image"], g1.inputs["Image"])
tree.links.new(g1.outputs["Image"], g2.inputs["Image"])
tree.links.new(g2.outputs["Image"], out_n.inputs["Image"])
bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "compliant_cycles.blend"))

# 3. Compliant White Studio with EEVEE RT=True + Fast GI (allowed under Scheme A when no emissive spill/volume/wet reflection)
reset_empty()
s = bpy.context.scene
s.name = "Studio_EEVEE_RT"
add_basic_scene_objects(s)
s.render.engine = "BLENDER_EEVEE"
s.eevee.use_raytracing = True
s.eevee.use_fast_gi = True
s.eevee.taa_render_samples = 32
bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "studio_eevee_rt.blend"))
'''


def _mock_source_scene3d(require_cycles: bool, volume: bool, bloom: bool, spill: bool, glare_tier: str, k2f: float) -> dict:
    return {
        "lookdev_feature_gates": {
            "enable_volume_fog": volume,
            "enable_bloom_or_soft_glow": bloom,
            "enable_emissive_materials": spill,
            "enable_emissive_indirect_spill": spill,
            "enable_ground_reflection": False,
            "enable_dof": False,
            "enable_motion_blur": False,
        },
        "pillar_3_material_and_bsdf": {
            "shading_paradigm": "PBR_PRINCIPLED_BSDF",
            "special_material_signals": {"transmission_glass_hint": False},
        },
        "pillar_4_lighting_and_volumetrics": {
            "key_to_fill_linear_ratio": k2f,
            "linear_luminance_percentiles": {"p50_midtone": 0.03 if require_cycles else 0.45},
            "shadow_and_key_light_hardness": {
                "estimated_key_light_hardness": "MEDIUM_SOFT_AREA" if require_cycles else "BROAD_DIFFUSE_OR_HDRI"
            },
            "radial_halo_profile": {
                "near_to_core_ratio": 0.18 if bloom else 0.0,
                "far_to_core_ratio": 0.05 if volume else 0.0,
            },
            "recommended_blender_render_requirements": {
                "require_cycles": require_cycles,
                "allow_eevee_no_rt": False,
                "min_diffuse_bounces": 2 if require_cycles else 1,
                "min_glossy_bounces": 2 if require_cycles else 1,
                "min_transmission_bounces": 1,
                "min_volume_bounces": 1 if volume else 0,
                "compositor_glare_tier": glare_tier,
            },
        },
    }


class BlendAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.blender = ba.find_blender_executable()
        except FileNotFoundError:
            cls.blender = None
            return
        cls._tmp = tempfile.TemporaryDirectory(prefix="h2o_test_blends_")
        cls.tmp_dir = Path(cls._tmp.name)
        builder = cls.tmp_dir / "build_blends.py"
        builder.write_text(_BUILD_TEST_BLENDS_PY, encoding="utf-8")
        proc = subprocess.run(
            [str(cls.blender), "-b", "--python-exit-code", "1", "--python", str(builder), "--", str(cls.tmp_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed generating test blends: {proc.stderr[-500:]}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    def test_degraded_no_rt_eevee_fails_strict_audit(self):
        if not self.blender:
            self.skipTest("Blender not installed")
        src_p = self.tmp_dir / "src_night.json"
        src_p.write_text(
            json.dumps(_mock_source_scene3d(True, True, True, True, "DUAL_NEAR_AND_BROAD", 6.0)),
            encoding="utf-8",
        )
        rep = ba.audit_blend_file(self.tmp_dir / "degraded_eevee.blend", [src_p], out_dir=self.tmp_dir / "out_bad")
        self.assertEqual(rep["verdict"], "FAIL")
        fail_metrics = {c["metric"] for c in rep["failed_checks"]}
        warn_metrics = {c["metric"] for c in rep["warning_checks"]}
        self.assertIn("engine_fidelity_lock", fail_metrics)
        self.assertIn("eevee_raytracing_enabled", fail_metrics)
        self.assertIn("eevee_gi_and_probes", fail_metrics)
        self.assertIn("render_samples_floor", fail_metrics)
        self.assertIn("volume_medium_present", fail_metrics)
        self.assertIn("compositor_glare_depth", fail_metrics)
        self.assertIn("oversized_flat_fill_lamps", warn_metrics)
        self.assertTrue((self.tmp_dir / "out_bad" / "blend_audit.md").exists())

    def test_compliant_cycles_passes_strict_audit(self):
        if not self.blender:
            self.skipTest("Blender not installed")
        src_p = self.tmp_dir / "src_night.json"
        src_p.write_text(
            json.dumps(_mock_source_scene3d(True, True, True, True, "DUAL_NEAR_AND_BROAD", 6.0)),
            encoding="utf-8",
        )
        rep = ba.audit_blend_file(self.tmp_dir / "compliant_cycles.blend", [src_p], out_dir=self.tmp_dir / "out_good")
        self.assertEqual(rep["verdict"], "PASS", f"Unexpected failures: {rep['failed_checks']}")

    def test_white_studio_with_eevee_rt_passes_under_scheme_a(self):
        if not self.blender:
            self.skipTest("Blender not installed")
        src_p = self.tmp_dir / "src_studio.json"
        src_p.write_text(
            json.dumps(_mock_source_scene3d(False, False, False, False, "NONE", 1.8)),
            encoding="utf-8",
        )
        rep = ba.audit_blend_file(self.tmp_dir / "studio_eevee_rt.blend", [src_p])
        self.assertEqual(rep["verdict"], "PASS", f"Unexpected failures: {rep['failed_checks']}")


if __name__ == "__main__":
    unittest.main()
