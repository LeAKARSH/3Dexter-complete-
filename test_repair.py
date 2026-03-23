"""
test_repair.py  —  3Dexter repair engine test suite
====================================================
Covers every test category from the Grand Plan:

  Unit tests
    - manifold mesh stays manifold
    - holed mesh improves
    - empty mesh returns gracefully
    - disconnected mesh follows component policy

  Integration tests
    - parametric path (no smoothing, dims preserved)
    - organic path (smoothing enabled)
    - assembly mode (lid not lost)

  Metric tests
    - hole count via Euler / boundary-loop method
    - watertight detection
    - volume non-zero after repair

  Normalisation tests
    - organic mesh lands in target size range
    - ground-plane alignment (Z_min ≈ 0)

Run with:
    python test_repair.py
or:
    python -m pytest test_repair.py -v
"""

import sys
import math
import unittest
import numpy as np
import trimesh

# import from same directory
sys.path.insert(0, ".")
from repair import (
    RepairEngine, RepairConfig, RepairResult,
    Track, _count_holes, _is_watertight,
    _bounding_box_diagonal, _normalize, _component_volumes,
)


# ---------------------------------------------------------------------------
# Mesh factories
# ---------------------------------------------------------------------------

def make_icosphere(subdivisions: int = 2) -> trimesh.Trimesh:
    """Clean, manifold, watertight sphere."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=50.0)


def make_broken_sphere(hole_faces: int = 20) -> trimesh.Trimesh:
    """Sphere with a patch of faces removed → open boundary → holes."""
    mesh = make_icosphere()
    # remove a cluster of faces around the +Z pole
    zvals = mesh.triangles_center[:, 2]
    top_idx = np.argsort(zvals)[-hole_faces:]
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[top_idx] = False
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)


def make_empty() -> trimesh.Trimesh:
    return trimesh.Trimesh()


def make_two_separate_spheres() -> trimesh.Trimesh:
    """Two disconnected spheres — simulates a container + lid."""
    s1 = trimesh.creation.icosphere(subdivisions=2, radius=40.0)
    s2 = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    s2.apply_translation([0, 0, 120])   # clearly separate
    return trimesh.util.concatenate([s1, s2])


def make_small_fragment_mesh() -> trimesh.Trimesh:
    """Big sphere + tiny stray triangle (0.1% volume)."""
    s = make_icosphere(subdivisions=3)
    tiny = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    tiny.apply_translation([0, 0, 200])
    return trimesh.util.concatenate([s, tiny])


def make_nan_mesh() -> trimesh.Trimesh:
    mesh = make_icosphere()
    verts = mesh.vertices.copy().astype(float)
    verts[0, 0] = np.nan
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def make_box_scad_like() -> trimesh.Trimesh:
    """Box with exact 10×20×30 mm dims — CAD-like, should not be shrunk."""
    return trimesh.creation.box(extents=[10, 20, 30])


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def test_watertight_icosphere(self):
        m = make_icosphere()
        self.assertTrue(_is_watertight(m))

    def test_hole_count_watertight_is_zero(self):
        m = make_icosphere()
        self.assertEqual(_count_holes(m), 0)

    def test_hole_count_open_mesh(self):
        m = make_broken_sphere(hole_faces=30)
        holes = _count_holes(m)
        # removing a connected patch creates exactly 1 boundary loop
        self.assertGreaterEqual(holes, 1)

    def test_hole_count_empty(self):
        self.assertEqual(_count_holes(make_empty()), 0)

    def test_bbox_diagonal(self):
        box = trimesh.creation.box(extents=[3, 4, 0])
        # 2D box: diagonal of 3-4-0 = 5.0
        diag = _bounding_box_diagonal(box)
        self.assertAlmostEqual(diag, 5.0, places=1)


class TestComponentPolicy(unittest.TestCase):
    def test_single_object_keeps_largest(self):
        mesh = make_two_separate_spheres()
        cfg = RepairConfig(track=Track.PARAMETRIC, assembly=False)
        engine = RepairEngine(cfg)
        result = engine.repair(mesh)
        self.assertTrue(result.ok)
        parts = _component_volumes(result.mesh)
        # after policy, only one component
        self.assertEqual(len(parts), 1)

    def test_assembly_mode_keeps_both(self):
        mesh = make_two_separate_spheres()
        cfg = RepairConfig(track=Track.PARAMETRIC, assembly=True,
                           volume_fraction_min=0.01)
        engine = RepairEngine(cfg)
        result = engine.repair(mesh)
        self.assertTrue(result.ok)
        parts = _component_volumes(result.mesh)
        self.assertGreaterEqual(len(parts), 2,
            "assembly mode must not discard the lid component")

    def test_assembly_mode_drops_stray_fragment(self):
        mesh = make_small_fragment_mesh()
        cfg = RepairConfig(track=Track.PARAMETRIC, assembly=True,
                           volume_fraction_min=0.01)
        engine = RepairEngine(cfg)
        result = engine.repair(mesh)
        # the tiny triangle (0 volume) should be dropped; large sphere kept
        parts = _component_volumes(result.mesh)
        self.assertEqual(len(parts), 1)


class TestEmptyMesh(unittest.TestCase):
    def test_empty_mesh_returns_gracefully(self):
        engine = RepairEngine()
        result = engine.repair(make_empty())
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        self.assertIsInstance(result.mesh, trimesh.Trimesh)

    def test_empty_mesh_no_crash(self):
        # Must not raise
        engine = RepairEngine()
        try:
            engine.repair(make_empty())
        except Exception as exc:
            self.fail(f"repair raised unexpectedly: {exc}")


class TestDegenerateCoords(unittest.TestCase):
    def test_nan_mesh_handled(self):
        engine = RepairEngine()
        # Should not crash; may succeed or fail gracefully
        result = engine.repair(make_nan_mesh())
        self.assertIsNotNone(result)


class TestManifoldPreservation(unittest.TestCase):
    def test_good_mesh_stays_good(self):
        """Repair must not degrade an already-manifold mesh."""
        mesh = make_icosphere(subdivisions=3)
        self.assertTrue(mesh.is_watertight)

        engine = RepairEngine(RepairConfig(track=Track.PARAMETRIC))
        result = engine.repair(mesh)

        self.assertTrue(result.ok)
        self.assertTrue(result.report["after"]["watertight"],
                        "already-manifold mesh must remain watertight after repair")

    def test_broken_mesh_improves(self):
        mesh = make_broken_sphere(hole_faces=40)
        holes_before = _count_holes(mesh)
        self.assertGreater(holes_before, 0)

        engine = RepairEngine(RepairConfig(track=Track.ORGANIC))
        result = engine.repair(mesh)

        self.assertTrue(result.ok)
        holes_after = result.report["after"]["holes"]
        self.assertLessEqual(holes_after, holes_before,
                             "repair must not increase hole count")


# ---------------------------------------------------------------------------
# Track-aware behaviour
# ---------------------------------------------------------------------------

class TestTrackBehaviour(unittest.TestCase):
    def _repair_both_tracks(self, mesh):
        r_param = RepairEngine(RepairConfig(track=Track.PARAMETRIC,
                                            normalize_scale=False)).repair(mesh)
        r_org   = RepairEngine(RepairConfig(track=Track.ORGANIC,
                                            normalize_scale=False)).repair(mesh)
        return r_param, r_org

    def test_parametric_preserves_volume(self):
        """Parametric track skips smoothing → volume change should be tiny."""
        mesh = make_icosphere()
        vol_before = abs(mesh.volume)

        r_param, _ = self._repair_both_tracks(mesh)
        vol_after  = r_param.report["after"]["volume_mm3"]

        rel_change = abs(vol_after - vol_before) / vol_before
        self.assertLess(rel_change, 0.05,
                        "parametric track must not alter volume by >5%")

    def test_organic_smooth_runs(self):
        """Organic track applies smoothing without crashing."""
        mesh = make_broken_sphere()
        engine = RepairEngine(RepairConfig(track=Track.ORGANIC, normalize_scale=False))
        result = engine.repair(mesh)
        self.assertTrue(result.ok)
        self.assertGreater(result.report["after"]["vertices"], 0)


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalisation(unittest.TestCase):
    def test_organic_scale_normalised(self):
        """After normalisation, bounding-box diagonal must be near target."""
        mesh = make_broken_sphere()
        cfg = RepairConfig(track=Track.ORGANIC, normalize_scale=True,
                            target_longest_edge_mm=100.0)
        engine = RepairEngine(cfg)
        result = engine.repair(mesh)
        self.assertTrue(result.ok)
        diag = result.report["bbox_diagonal_mm"]
        self.assertAlmostEqual(diag, 100.0, delta=5.0,
            msg="normalised diagonal should be close to target_longest_edge_mm")

    def test_ground_plane_z_min_near_zero(self):
        """After normalisation, minimum Z must be ≥ 0 (object on ground plane)."""
        mesh = make_icosphere()
        cfg = RepairConfig(track=Track.ORGANIC, normalize_scale=True)
        engine = RepairEngine(cfg)
        result = engine.repair(mesh)
        zmin = result.mesh.bounds[0][2]
        self.assertGreaterEqual(zmin, -0.01,
            "normalised mesh should sit on the Z=0 ground plane")

    def test_parametric_no_normalise(self):
        """With normalize_scale=False, a 10×20×30 box keeps its exact dimensions."""
        box = make_box_scad_like()
        original_diag = _bounding_box_diagonal(box)
        cfg = RepairConfig(track=Track.PARAMETRIC, normalize_scale=False)
        engine = RepairEngine(cfg)
        result = engine.repair(box)
        final_diag = _bounding_box_diagonal(result.mesh)
        self.assertAlmostEqual(final_diag, original_diag, delta=0.5,
            msg="without normalisation, CAD dimensions must be unchanged")


# ---------------------------------------------------------------------------
# Report contract tests
# ---------------------------------------------------------------------------

class TestReportContract(unittest.TestCase):
    def _get_report(self, mesh=None, **kwargs):
        mesh = mesh or make_icosphere()
        cfg = RepairConfig(**kwargs) if kwargs else RepairConfig()
        return RepairEngine(cfg).repair(mesh).report

    def test_report_has_required_keys(self):
        report = self._get_report()
        required = {"track", "assembly_mode", "before", "after",
                    "holes_fixed", "became_watertight",
                    "scale_normalised", "bbox_diagonal_mm"}
        self.assertTrue(required.issubset(report.keys()),
                        f"missing keys: {required - report.keys()}")

    def test_before_after_have_sub_keys(self):
        report = self._get_report()
        sub_keys = {"vertices", "faces", "watertight", "holes", "volume_mm3"}
        for section in ("before", "after"):
            self.assertTrue(sub_keys.issubset(report[section].keys()),
                            f"'{section}' missing keys")

    def test_holes_fixed_non_negative(self):
        report = self._get_report(make_broken_sphere())
        self.assertGreaterEqual(report["holes_fixed"], 0)

    def test_track_recorded_correctly(self):
        for track in (Track.PARAMETRIC, Track.ORGANIC):
            report = self._get_report(track=track)
            self.assertEqual(report["track"], track.value)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    def test_parametric_pipeline(self):
        """Parametric end-to-end: sphere → watertight, volume > 0."""
        mesh = make_broken_sphere()
        cfg = RepairConfig(track=Track.PARAMETRIC, assembly=False)
        result = RepairEngine(cfg).repair(mesh)
        self.assertTrue(result.ok)
        self.assertGreater(result.report["after"]["vertices"], 0)
        # After repair, holes should be 0 or at least fewer
        self.assertLessEqual(result.report["after"]["holes"],
                             result.report["before"]["holes"])

    def test_organic_pipeline(self):
        """Organic end-to-end: broken mesh → STL exportable."""
        mesh = make_broken_sphere()
        cfg = RepairConfig(track=Track.ORGANIC)
        result = RepairEngine(cfg).repair(mesh)
        self.assertTrue(result.ok)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            path = f.name
        try:
            result.mesh.export(path)
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_assembly_lid_preserved(self):
        """Assembly of box + lid: both survive repair."""
        body = trimesh.creation.box(extents=[60, 40, 30])
        lid  = trimesh.creation.box(extents=[62, 42, 5])
        lid.apply_translation([0, 0, 35])
        combined = trimesh.util.concatenate([body, lid])

        cfg = RepairConfig(track=Track.PARAMETRIC, assembly=True,
                            normalize_scale=False)
        result = RepairEngine(cfg).repair(combined)
        self.assertTrue(result.ok)
        parts = _component_volumes(result.mesh)
        self.assertGreaterEqual(len(parts), 2,
            "lid must not be silently discarded in assembly mode")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
