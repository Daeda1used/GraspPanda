"""Collision check of a gripper *mesh* against the observed cloud, through trimesh.

:class:`~grasppanda.models.postprocessors.collision.ModelFreeCollisionFilter` tests the
four boxes of the benchmark gripper. A real gripper is not four boxes: this filter
loads the collision mesh of the gripper that will execute the grasp (any format trimesh
reads: STL, OBJ, PLY, GLB), places it at every grasp, and rejects the grasp when scene
points lie inside it. Without a mesh file it builds the open benchmark gripper from
boxes, so it runs out of the box.

trimesh loads the mesh, splits it into bodies, checks that they are closed and builds
convex hulls and the box gripper; the placement of thousands of grasps, the selection of
the points near each gripper and the containment test run here in PyTorch, on the device
of the grasps, so that trimesh's optional ray engines (``rtree``, ``embreex``) are not
needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from grasppanda.models.base import PostProcessor
from grasppanda.registry import POSTPROCESSORS
from grasppanda.structures import GraspGroup, Observation
from grasppanda.utils.imports import require

from .collision import voxel_down_sample
from .scene import CHUNK, as_transform, gripper_frame_points, scene_points

__all__ = ["GripperMeshCollisionFilter"]

#: Direction of the parity rays: no axis, no face diagonal of a typical mesh.
_RAY = (0.5370861555295747, 0.7071067811865476, 0.4605048191608812)


def _inside_closed_mesh(triangles: Tensor, points: Tensor, block: int = 2_000_000) -> Tensor:
    """``(M,)`` whether each point is inside the closed surface ``triangles`` ``(F, 3, 3)``.

    A ray from the point crosses the surface an odd number of times exactly when the
    point is inside (Möller-Trumbore intersection, all rays in one fixed direction).
    Written here so that the check needs neither ``rtree`` nor ``embreex``.
    """
    ray = points.new_tensor(_RAY)
    origin = triangles[:, 0]
    edge1, edge2 = triangles[:, 1] - origin, triangles[:, 2] - origin
    pvec = torch.linalg.cross(ray.expand_as(edge2), edge2)
    det = (edge1 * pvec).sum(dim=1)
    usable = det.abs() > 1e-15
    inverse = torch.where(usable, 1.0 / torch.where(usable, det, torch.ones_like(det)), 0.0)
    inside = []
    step = max(1, block // max(len(triangles), 1))
    for start in range(0, len(points), step):
        tvec = points[start : start + step, None] - origin[None]  # (m, F, 3)
        u = (tvec * pvec).sum(dim=-1) * inverse
        qvec = torch.linalg.cross(tvec, edge1.expand_as(tvec))
        v = (qvec @ ray) * inverse
        t = (qvec * edge2).sum(dim=-1) * inverse
        hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        inside.append(hit.sum(dim=1) % 2 == 1)
    return torch.cat(inside)


@POSTPROCESSORS.register()
class GripperMeshCollisionFilter(PostProcessor):
    """Remove grasps at which a gripper mesh would contain observed points.

    **The tool frame.** The mesh is expressed in the tool frame of a grasp: origin in
    the middle between the fingertips, x along the approach (out of the gripper, into
    the object), y along the closing direction, z along the finger height, metres. For
    a grasp that is ``translation + depth * approach``, with the axes of its rotation.
    A mesh file in another frame (a URDF link frame, millimetres) is brought there with
    ``mesh_transform`` and ``scale``.

    **What is tested.** The gripper as the mesh shows it, usually fully open, at the
    grasp pose. Points between the fingers are outside the mesh and do not count, so
    the object to be grasped is not a collision; a neighbour that an open finger would
    hit is. There is no approach sweep and no per-grasp width: use it beside the box
    filter, not instead of it, when the approach path matters.

    Args:
        mesh: Path of the gripper mesh; ``None`` builds the benchmark gripper, opened to
            ``opening``, from three boxes (two fingers and a palm).
        mesh_transform: ``(4, 4)`` rigid transform from the coordinates of the mesh file
            (after ``scale``) to the tool frame; nested lists or a ``.npy``/text file.
        scale: Factor applied to the mesh coordinates first (``0.001`` for millimetres).
        opening: Opening of the built-in gripper, metres; ``None`` uses the largest
            opening of the gripper of the grasps. Ignored with a mesh file.
        bodies: ``exact`` tests containment in every watertight body of the mesh by ray
            parity (cost grows with points × triangles: use a collision mesh of a few
            hundred to a few thousand triangles, not a visual mesh). ``convex`` replaces every body by its convex hull
            and tests half-spaces in PyTorch: fast, and right when fingers and palm are
            separate bodies; a one-piece hand would fill the gap between its fingers.
        voxel_size: The scene cloud is down-sampled to this edge length first, metres;
            ``0`` uses every point.
        max_points: A grasp with more than this many points inside the mesh is removed.
        cloud: ``full_points`` (before sub-sampling; ``points`` when the observation
            does not keep it) or ``points``.
        finger_length: Finger length of the built-in gripper, metres.
    """

    status = "bridge"
    extras = ("trimesh",)
    source = (
        "Bridge to trimesh (Dawson-Haggerty et al., github.com/mikedh/trimesh, MIT): mesh "
        "loading, body split, watertightness, convex hull and box primitives; the point-cloud-against-gripper-"
        "mesh check of GraspGen (Murali et al., arXiv 2507.13097) as a containment test"
    )
    primary = "mesh"

    def __init__(
        self,
        mesh: str | None = None,
        mesh_transform: Sequence[Sequence[float]] | str | None = None,
        scale: float = 1.0,
        opening: float | None = None,
        bodies: Literal["exact", "convex"] = "exact",
        voxel_size: float = 0.005,
        max_points: int = 0,
        cloud: Literal["full_points", "points"] = "full_points",
        finger_length: float = 0.06,
    ) -> None:
        if bodies not in ("exact", "convex"):
            raise ValueError(f"bodies must be 'exact' or 'convex', got {bodies!r}")
        if scale <= 0 or voxel_size < 0 or max_points < 0 or finger_length <= 0:
            raise ValueError(
                "scale and finger_length must be positive, voxel_size and max_points not negative"
            )
        self.mesh = mesh
        self.mesh_transform = (
            None
            if mesh_transform is None
            else as_transform(mesh_transform, "GripperMeshCollisionFilter")
        )
        self.scale = scale
        self.opening = opening
        self.bodies = bodies
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.cloud = cloud
        self.finger_length = finger_length
        self._parts: dict[Any, list[Any]] = {}
        if mesh is not None:  # fail at build time, not at the first frame
            self._bodies_of_file()

    # ------------------------------------------------------------------ the gripper

    def _trimesh(self) -> Any:
        return require(
            "trimesh", extra="trimesh", purpose="GripperMeshCollisionFilter (gripper mesh)"
        )

    def _bodies_of_file(self) -> list[Any]:
        if "file" in self._parts:
            return self._parts["file"]
        trimesh = self._trimesh()
        loaded = trimesh.load(self.mesh, force="mesh")
        if not len(getattr(loaded, "faces", ())):
            raise ValueError(f"GripperMeshCollisionFilter: {self.mesh} holds no triangles")
        loaded.apply_scale(self.scale)
        if self.mesh_transform is not None:
            loaded.apply_transform(self.mesh_transform.numpy())
        parts = list(loaded.split(only_watertight=False))
        if self.bodies == "convex":
            parts = [part.convex_hull for part in parts if len(part.vertices) >= 4]
        else:
            open_parts = sum(not part.is_watertight for part in parts)
            parts = [part for part in parts if part.is_watertight]
            if open_parts:
                raise ValueError(
                    f"GripperMeshCollisionFilter: {open_parts} of the {open_parts + len(parts)} "
                    f"bodies of {self.mesh} are not watertight, so 'inside' is undefined for "
                    "them. Use a closed collision mesh, or `bodies: convex` to test the convex "
                    "hull of every body."
                )
        if not parts:
            raise ValueError(f"GripperMeshCollisionFilter: {self.mesh} has no usable body")
        extent = np.ptp(np.concatenate([part.vertices for part in parts]), axis=0).max()
        if not 0.02 <= extent <= 1.0:
            raise ValueError(
                f"GripperMeshCollisionFilter: the gripper mesh is {extent:.4g} m across after "
                f"scale={self.scale}. Lengths are metres: a mesh in millimetres needs "
                "scale=0.001."
            )
        self._parts["file"] = parts
        return parts

    def _bodies_of_boxes(self, grasps: GraspGroup) -> list[Any]:
        gripper = grasps.gripper
        opening = gripper.max_width if self.opening is None else self.opening
        key = (opening, gripper.finger_width, gripper.finger_height)
        if key not in self._parts:
            trimesh = self._trimesh()
            width, height, length = gripper.finger_width, gripper.finger_height, self.finger_length

            def box(extents: tuple[float, float, float], centre: tuple[float, float, float]) -> Any:
                pose = np.eye(4)
                pose[:3, 3] = centre
                return trimesh.creation.box(extents=extents, transform=pose)

            side = opening / 2 + width / 2
            self._parts[key] = [
                box((length, width, height), (-length / 2, -side, 0.0)),
                box((length, width, height), (-length / 2, side, 0.0)),
                box((width, opening + 2 * width, height), (-length - width / 2, 0.0, 0.0)),
            ]
        return self._parts[key]

    def gripper_bodies(self, grasps: GraspGroup) -> list[Any]:
        """The bodies of the gripper in the tool frame, as trimesh meshes."""
        return self._bodies_of_file() if self.mesh is not None else self._bodies_of_boxes(grasps)

    # ------------------------------------------------------------------- the check

    def _inside(self, parts: list[Any], points: Tensor) -> Tensor:
        """``(M,)`` whether each tool-frame point lies inside a body."""
        if len(points) == 0:
            return torch.zeros(0, dtype=torch.bool, device=points.device)
        inside = torch.zeros(len(points), dtype=torch.bool, device=points.device)
        if self.bodies == "exact" and self.mesh is not None:
            for part in parts:
                triangles = torch.as_tensor(np.array(part.triangles)).to(points)
                inside |= _inside_closed_mesh(triangles, points)
            return inside
        for part in parts:  # convex: inside iff behind every face
            normals = torch.as_tensor(np.array(part.face_normals)).to(points)
            origins = torch.as_tensor(np.array(part.triangles[:, 0])).to(points)
            signed = points @ normals.T - (normals * origins).sum(dim=1)
            inside |= (signed <= 1e-9).all(dim=1)
        return inside

    @torch.no_grad()
    def collision_counts(self, grasps: GraspGroup, scene: Tensor) -> Tensor:
        """``(G,)`` number of ``scene`` points inside the gripper at every grasp."""
        parts = self.gripper_bodies(grasps)
        corners = np.concatenate([part.bounds for part in parts])
        low = scene.new_tensor(corners.min(axis=0)).double()
        high = scene.new_tensor(corners.max(axis=0)).double()
        scene = scene.double()
        owners, candidates = [], []
        for start in range(0, len(grasps), CHUNK):
            part = grasps[slice(start, start + CHUNK)]
            local = gripper_frame_points(part, scene).double()
            local[..., 0] -= part.depth[:, None].double()  # origin at the fingertips
            near = ((local >= low) & (local <= high)).all(dim=-1)
            grasp_index, point_index = near.nonzero(as_tuple=True)
            owners.append(grasp_index + start)
            candidates.append(local[grasp_index, point_index])
        owner, points = torch.cat(owners), torch.cat(candidates)
        inside = self._inside(parts, points)
        counts = torch.zeros(len(grasps), dtype=torch.long, device=scene.device)
        return counts.index_add_(0, owner[inside], torch.ones_like(owner[inside]))

    def __call__(self, grasps: GraspGroup, observation: Observation) -> GraspGroup:
        if len(grasps) == 0:
            return grasps
        scene = scene_points(
            "GripperMeshCollisionFilter", grasps, observation, self.cloud, fallback=True
        )
        if self.voxel_size > 0:
            scene = voxel_down_sample(scene, self.voxel_size)
        return grasps[self.collision_counts(grasps, scene) <= self.max_points]
