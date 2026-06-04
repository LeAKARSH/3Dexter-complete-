from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pymeshfix
import trimesh

log = logging.getLogger("repair")


class Track(str, enum.Enum):
    PARAMETRIC = "parametric"
    ORGANIC = "organic"


@dataclass
class RepairConfig:
    track: Track = Track.PARAMETRIC
    assembly: bool = False
    volume_fraction_min: float = 0.01
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5
    normalize_scale: bool = True
    target_longest_edge_mm: float = 100.0
    up_axis: str = "Z"
    max_repair_passes: int = 3


@dataclass
class RepairResult:
    mesh: trimesh.Trimesh
    report: dict = field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None


def _is_watertight(mesh: trimesh.Trimesh) -> bool:
    return bool(mesh.is_watertight)


def _count_holes(mesh: trimesh.Trimesh) -> int:
    if mesh.is_empty or mesh.is_watertight:
        return 0

    edges = mesh.edges_unique
    edge_faces = mesh.edges_unique_inverse
    face_count = np.bincount(edge_faces, minlength=len(edges))
    boundary_mask = face_count == 1
    boundary_edges = edges[boundary_mask]

    if len(boundary_edges) == 0:
      return 0

    from collections import defaultdict

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    loops = 0
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        loops += 1
        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            stack.extend(adjacency[vertex])
    return loops


def _component_volumes(mesh: trimesh.Trimesh) -> list[tuple[trimesh.Trimesh, float]]:
    parts = []
    for component in mesh.split(only_watertight=False):
        if component.is_empty:
            continue
        try:
            volume = abs(component.volume) if component.is_watertight else abs(component.convex_hull.volume)
        except Exception:
            volume = 0.0
        parts.append((component, volume))
    parts.sort(key=lambda item: item[1], reverse=True)
    return parts


def _has_degenerate_coords(mesh: trimesh.Trimesh) -> bool:
    return bool(np.any(~np.isfinite(mesh.vertices)) or np.any(np.abs(mesh.vertices) > 1e9))


def _bounding_box_diagonal(mesh: trimesh.Trimesh) -> float:
    return float(np.linalg.norm(mesh.bounding_box.extents))


def _normalize(mesh: trimesh.Trimesh, cfg: RepairConfig) -> trimesh.Trimesh:
    if mesh.is_empty:
        return mesh

    normalized = mesh.copy()
    normalized.apply_translation(-normalized.centroid)

    extents = normalized.bounding_box.extents
    longest_axis = int(np.argmax(extents))
    target_axis = {"X": 0, "Y": 1, "Z": 2}.get(cfg.up_axis.upper(), 2)

    if longest_axis != target_axis:
        transform = np.eye(4)
        permutation = list(range(3))
        permutation[longest_axis], permutation[target_axis] = permutation[target_axis], permutation[longest_axis]
        rotation = np.zeros((3, 3))
        for source_idx, target_idx in enumerate(permutation):
            rotation[target_idx, source_idx] = 1.0
        transform[:3, :3] = rotation
        normalized.apply_transform(transform)

    diagonal = _bounding_box_diagonal(normalized)
    if diagonal > 0:
        normalized.apply_scale(cfg.target_longest_edge_mm / diagonal)

    z_min = normalized.bounds[0][2]
    normalized.apply_translation([0, 0, -z_min])
    return normalized


class RepairEngine:
    def __init__(self, config: Optional[RepairConfig] = None):
        self.cfg = config or RepairConfig()

    def repair(self, source) -> RepairResult:
        try:
            mesh = self._load(source)
        except Exception as exc:
            return RepairResult(mesh=trimesh.Trimesh(), ok=False, error=f"load failed: {exc}")

        if mesh.is_empty or len(mesh.vertices) == 0:
            return RepairResult(mesh=trimesh.Trimesh(), ok=False, error="empty mesh; nothing to repair")

        if _has_degenerate_coords(mesh):
            mesh.remove_infinite_values()
            if _has_degenerate_coords(mesh):
                return RepairResult(mesh=mesh, ok=False, error="mesh has unfixable degenerate coordinates")

        before = self._metrics(mesh)
        repaired = self._run_repair_passes(mesh)
        repaired = self._apply_component_policy(repaired)

        if self.cfg.track == Track.ORGANIC:
            repaired = self._smooth(repaired)

        if self.cfg.normalize_scale:
            repaired = _normalize(repaired, self.cfg)

        after = self._metrics(repaired)
        report = {
            "track": self.cfg.track.value,
            "assembly_mode": self.cfg.assembly,
            "before": before,
            "after": after,
            "holes_fixed": max(0, before["holes"] - after["holes"]),
            "became_watertight": (not before["watertight"]) and after["watertight"],
            "scale_normalized": self.cfg.normalize_scale,
            "bbox_diagonal_mm": round(_bounding_box_diagonal(repaired), 3),
        }
        return RepairResult(mesh=repaired, report=report, ok=True)

    @staticmethod
    def _load(source) -> trimesh.Trimesh:
        if isinstance(source, trimesh.Trimesh):
            return source.copy()

        # Support mesh-like objects from model SDKs (e.g., Shape-E tri_mesh)
        # that expose either `vertices/faces` or `verts/faces`.
        if hasattr(source, "vertices") and hasattr(source, "faces"):
            return trimesh.Trimesh(vertices=np.asarray(source.vertices), faces=np.asarray(source.faces), process=False)
        if hasattr(source, "verts") and hasattr(source, "faces"):
            return trimesh.Trimesh(vertices=np.asarray(source.verts), faces=np.asarray(source.faces), process=False)

        path = Path(source)
        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
            if not meshes:
                return trimesh.Trimesh()
            return trimesh.util.concatenate(meshes)
        return loaded

    def _run_repair_passes(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        repaired = mesh.copy()
        for _ in range(self.cfg.max_repair_passes):
            if repaired.is_empty or repaired.is_watertight:
                break
            repaired = self._pymeshfix_repair(repaired)
            trimesh.repair.fix_winding(repaired)
            trimesh.repair.fix_inversion(repaired)
            trimesh.repair.fix_normals(repaired)
            repaired = trimesh.Trimesh(vertices=repaired.vertices, faces=repaired.faces, process=True)
        return repaired

    @staticmethod
    def _pymeshfix_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            fixer = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
            try:
                fixer.repair(verbose=False)
            except TypeError:
                fixer.repair()
            return trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=False)
        except Exception as exc:
            log.warning("pymeshfix failed (%s); keeping current mesh", exc)
            return mesh

    def _apply_component_policy(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        parts = _component_volumes(mesh)
        if not parts:
            return mesh

        if not self.cfg.assembly:
            return parts[0][0]

        largest_volume = parts[0][1]
        if largest_volume <= 0:
            return parts[0][0]

        kept = [component for component, volume in parts if volume / largest_volume >= self.cfg.volume_fraction_min]
        if len(kept) == 1:
            return kept[0]
        return trimesh.util.concatenate(kept)

    def _smooth(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if mesh.is_empty:
            return mesh
        try:
            # trimesh smoothing APIs may mutate in place and return None.
            smoothed = mesh.copy()
            trimesh.smoothing.filter_laplacian(
                smoothed,
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
                "vertices": 0,
                "faces": 0,
                "watertight": False,
                "holes": 0,
                "volume_mm3": 0.0,
                "bbox_diagonal": 0.0,
            }

        try:
            volume = round(float(abs(mesh.volume)), 4) if mesh.is_watertight else 0.0
        except Exception:
            volume = 0.0

        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": _is_watertight(mesh),
            "holes": _count_holes(mesh),
            "volume_mm3": volume,
            "bbox_diagonal": round(_bounding_box_diagonal(mesh), 3),
        }
