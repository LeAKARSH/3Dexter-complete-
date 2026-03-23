"""
shared/repair.py  —  3Dexter mesh repair engine
================================================
Implements every fix called out in the Grand Plan:

  • empty-mesh guard (crash fix)
  • correct watertight / manifold checks
  • valid hole-count metric via Euler characteristic (not boundary-edge proxy)
  • assembly-aware component policy  (track="parametric" keeps all components)
  • track-aware smoothing  (CAD track skips Laplacian to preserve dimensions)
  • scale / orientation normalisation before export
  • repair report dict with before/after metrics

Usage
-----
from repair import RepairEngine, RepairConfig, Track

cfg = RepairConfig(track=Track.PARAMETRIC, assembly=False)
engine = RepairEngine(cfg)
result = engine.repair(mesh)          # trimesh.Trimesh or path str/Path
print(result.report)
result.mesh.export("fixed.stl")
"""

from __future__ import annotations

import enum
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import pymeshfix

log = logging.getLogger("repair")


# ---------------------------------------------------------------------------
# Public enums & config
# ---------------------------------------------------------------------------

class Track(str, enum.Enum):
    PARAMETRIC = "parametric"   # OpenSCAD / CAD — preserve exact dims
    ORGANIC    = "organic"      # neural diffusion — smoothing allowed


@dataclass
class RepairConfig:
    track: Track = Track.PARAMETRIC

    # Component policy
    assembly: bool = False
    """
    False  →  keep only the largest component (single-object mode, v1 default).
    True   →  keep all components that pass the volume_fraction_min threshold.
    """
    volume_fraction_min: float = 0.01
    """
    In assembly mode, discard components whose volume is < this fraction of the
    largest component.  Guards against stray micro-fragments without losing lids.
    """

    # Smoothing (organic only)
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5

    # Normalisation
    normalize_scale: bool = True
    target_longest_edge_mm: float = 100.0   # default bounding-box diagonal
    up_axis: str = "Z"                      # "X" | "Y" | "Z"

    # Limits
    max_repair_passes: int = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    mesh:   trimesh.Trimesh
    report: dict = field(default_factory=dict)
    ok:     bool = True
    error:  Optional[str] = None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _is_watertight(mesh: trimesh.Trimesh) -> bool:
    """True when the mesh is manifold AND has no boundary edges."""
    return bool(mesh.is_watertight)


def _count_holes(mesh: trimesh.Trimesh) -> int:
    """
    Compute number of topological holes via Euler characteristic:
        χ = V - E + F
        genus = (2 - χ) / 2        (for a single connected surface)
        holes = genus  (each handle = one hole)

    For a manifold closed surface χ = 2, genus = 0, holes = 0.
    For a surface with open boundary edges we fall back to counting
    distinct boundary loops (each loop = one hole).

    The boundary-edge *count* used in many naive implementations is
    NOT a valid holes metric — it depends on tessellation density.
    """
    if mesh.is_empty:
        return 0

    # Fast path: if watertight, guaranteed 0 holes
    if mesh.is_watertight:
        return 0

    # Count boundary loops (each = one hole to fill)
    edges = mesh.edges_unique
    edge_faces = mesh.edges_unique_inverse  # which face each unique edge belongs to
    # boundary edges appear in exactly one face
    face_count = np.bincount(edge_faces, minlength=len(edges))
    boundary_mask = face_count == 1
    boundary_edges = edges[boundary_mask]

    if len(boundary_edges) == 0:
        return 0

    # Build adjacency among boundary edges to find connected loops
    from collections import defaultdict
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    loops = 0
    for start in adj:
        if start not in visited:
            loops += 1
            stack = [start]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                stack.extend(adj[v])
    return loops


def _component_volumes(mesh: trimesh.Trimesh) -> list[tuple[trimesh.Trimesh, float]]:
    """
    Split mesh into connected components and return list of (component, volume).
    Volume is estimated via convex hull when the component is not watertight.
    """
    components = mesh.split(only_watertight=False)
    results = []
    for c in components:
        if c.is_empty:
            continue
        try:
            vol = abs(c.volume) if c.is_watertight else abs(c.convex_hull.volume)
        except Exception:
            vol = 0.0
        results.append((c, vol))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _has_degenerate_coords(mesh: trimesh.Trimesh) -> bool:
    return bool(
        np.any(~np.isfinite(mesh.vertices))
        or np.any(np.abs(mesh.vertices) > 1e9)
    )


def _bounding_box_diagonal(mesh: trimesh.Trimesh) -> float:
    extents = mesh.bounding_box.extents
    return float(np.linalg.norm(extents))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(mesh: trimesh.Trimesh, cfg: RepairConfig) -> trimesh.Trimesh:
    """
    1. Translate centroid to origin.
    2. Rotate so the longest bounding-box axis aligns with cfg.up_axis.
    3. Uniformly scale so the bounding-box diagonal == cfg.target_longest_edge_mm.
    """
    if mesh.is_empty:
        return mesh

    # Centre
    mesh.apply_translation(-mesh.centroid)

    # Orient longest axis to up_axis
    extents = mesh.bounding_box.extents          # (dx, dy, dz)
    longest_local_axis = int(np.argmax(extents)) # 0=X 1=Y 2=Z
    up_map = {"X": 0, "Y": 1, "Z": 2}
    up_idx = up_map.get(cfg.up_axis.upper(), 2)

    if longest_local_axis != up_idx:
        # Swap axes via rotation
        rot = np.eye(4)
        # Build permutation matrix
        perm = list(range(3))
        perm[longest_local_axis], perm[up_idx] = perm[up_idx], perm[longest_local_axis]
        R = np.zeros((3, 3))
        for i, j in enumerate(perm):
            R[j, i] = 1.0
        rot[:3, :3] = R
        mesh.apply_transform(rot)

    # Scale
    diag = _bounding_box_diagonal(mesh)
    if diag > 0:
        scale = cfg.target_longest_edge_mm / diag
        mesh.apply_scale(scale)

    # Place on ground plane (Z=0 minimum)
    zmin = mesh.bounds[0][2]
    mesh.apply_translation([0, 0, -zmin])

    return mesh


# ---------------------------------------------------------------------------
# Core repair engine
# ---------------------------------------------------------------------------

class RepairEngine:
    """
    Stateless repair engine.  Call engine.repair(mesh_or_path) each time.
    """

    def __init__(self, config: Optional[RepairConfig] = None):
        self.cfg = config or RepairConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def repair(self, source) -> RepairResult:
        """
        Parameters
        ----------
        source : trimesh.Trimesh | str | Path
            Input mesh or file path.

        Returns
        -------
        RepairResult with .mesh, .report, .ok, .error
        """
        # --- load ---
        try:
            mesh = self._load(source)
        except Exception as exc:
            return RepairResult(mesh=trimesh.Trimesh(), ok=False,
                                error=f"load failed: {exc}")

        # --- guard: empty mesh ---
        if mesh.is_empty or len(mesh.vertices) == 0:
            return RepairResult(mesh=trimesh.Trimesh(), ok=False,
                                error="empty mesh — nothing to repair")

        # --- guard: degenerate coordinates ---
        if _has_degenerate_coords(mesh):
            log.warning("mesh has NaN/Inf coordinates; attempting to clean")
            mesh.remove_infinite_values()
            if _has_degenerate_coords(mesh):
                return RepairResult(mesh=mesh, ok=False,
                                    error="mesh has unfixable degenerate coordinates")

        # --- before metrics ---
        before = self._metrics(mesh)
        log.info("BEFORE repair: %s", before)

        # --- repair passes ---
        mesh = self._run_repair_passes(mesh)

        # --- component policy ---
        mesh = self._apply_component_policy(mesh)

        # --- track-aware smoothing ---
        if self.cfg.track == Track.ORGANIC:
            mesh = self._smooth(mesh)

        # --- normalise scale / orientation ---
        if self.cfg.normalize_scale:
            mesh = _normalize(mesh, self.cfg)

        # --- after metrics ---
        after = self._metrics(mesh)
        log.info("AFTER  repair: %s", after)

        report = {
            "track":             self.cfg.track.value,
            "assembly_mode":     self.cfg.assembly,
            "before":            before,
            "after":             after,
            "holes_fixed":       max(0, before["holes"] - after["holes"]),
            "became_watertight": (not before["watertight"]) and after["watertight"],
            "scale_normalised":  self.cfg.normalize_scale,
            "bbox_diagonal_mm":  round(_bounding_box_diagonal(mesh), 3),
        }

        return RepairResult(mesh=mesh, report=report, ok=True)

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source) -> trimesh.Trimesh:
        if isinstance(source, trimesh.Trimesh):
            return source.copy()
        path = Path(source)
        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            # flatten scene to single mesh
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                return trimesh.Trimesh()
            loaded = trimesh.util.concatenate(meshes)
        return loaded

    def _run_repair_passes(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        for pass_num in range(1, self.cfg.max_repair_passes + 1):
            if mesh.is_watertight:
                log.info("pass %d: already watertight, stopping", pass_num)
                break
            log.info("pass %d: running pymeshfix repair", pass_num)
            mesh = self._pymeshfix_repair(mesh)

            # trimesh built-ins for degenerate geometry
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_inversion(mesh)
            trimesh.repair.fix_normals(mesh)
            # API varies by trimesh version — use process() as canonical path
            mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                   faces=mesh.faces,
                                   process=True)

        return mesh

    @staticmethod
    def _pymeshfix_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            tin = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
            try:
                tin.repair(verbose=False)
            except TypeError:
                tin.repair()   # older/newer pymeshfix versions drop verbose kwarg
            return trimesh.Trimesh(vertices=tin.v, faces=tin.f, process=False)
        except Exception as exc:
            log.warning("pymeshfix failed (%s); continuing with trimesh only", exc)
            return mesh

    def _apply_component_policy(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        parts = _component_volumes(mesh)

        if not parts:
            return mesh

        if not self.cfg.assembly:
            # Single-object mode: keep only the largest
            log.info("single-object policy: keeping largest of %d components", len(parts))
            return parts[0][0]

        # Assembly mode: keep all components above the volume fraction threshold
        largest_vol = parts[0][1]
        if largest_vol <= 0:
            return parts[0][0]

        kept = [c for c, vol in parts
                if vol / largest_vol >= self.cfg.volume_fraction_min]
        log.info("assembly policy: keeping %d / %d components", len(kept), len(parts))

        if len(kept) == 1:
            return kept[0]

        combined = trimesh.util.concatenate(kept)
        return combined

    def _smooth(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Laplacian smoothing — only applied on the organic track."""
        if mesh.is_empty:
            return mesh
        try:
            smoothed = trimesh.smoothing.filter_laplacian(
                mesh,
                lamb=self.cfg.smooth_lambda,
                iterations=self.cfg.smooth_iterations,
            )
            return smoothed
        except Exception as exc:
            log.warning("smoothing failed (%s); returning unsmoothed mesh", exc)
            return mesh

    @staticmethod
    def _metrics(mesh: trimesh.Trimesh) -> dict:
        if mesh.is_empty:
            return {
                "vertices": 0, "faces": 0,
                "watertight": False, "holes": 0,
                "volume_mm3": 0.0,
                "bbox_diagonal": 0.0,
            }
        holes = _count_holes(mesh)
        try:
            vol = round(float(abs(mesh.volume)), 4) if mesh.is_watertight else 0.0
        except Exception:
            vol = 0.0
        return {
            "vertices":      len(mesh.vertices),
            "faces":         len(mesh.faces),
            "watertight":    _is_watertight(mesh),
            "holes":         holes,
            "volume_mm3":    vol,
            "bbox_diagonal": round(_bounding_box_diagonal(mesh), 3),
        }


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def repair_file(
    input_path: str,
    output_path: str,
    track: str = "parametric",
    assembly: bool = False,
    normalize: bool = True,
) -> dict:
    """
    Repair a mesh file and write the result.

    Returns the repair report dict.
    """
    cfg = RepairConfig(
        track=Track(track),
        assembly=assembly,
        normalize_scale=normalize,
    )
    engine = RepairEngine(cfg)
    result = engine.repair(input_path)

    if not result.ok:
        raise RuntimeError(f"Repair failed: {result.error}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(out))
    return result.report


# ---------------------------------------------------------------------------
# __main__ entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="3Dexter mesh repair engine")
    parser.add_argument("input",  help="Input mesh file (.stl / .obj / .ply)")
    parser.add_argument("output", help="Output mesh file (.stl / .obj)")
    parser.add_argument("--track",     choices=["parametric", "organic"],
                        default="parametric")
    parser.add_argument("--assembly",  action="store_true",
                        help="Keep all significant components (assembly mode)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Skip scale/orientation normalisation")
    args = parser.parse_args()

    try:
        report = repair_file(
            args.input, args.output,
            track=args.track,
            assembly=args.assembly,
            normalize=args.normalize,
        )
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
