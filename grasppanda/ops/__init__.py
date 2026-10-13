"""Geometric operators.

Every function here has a pure-PyTorch reference implementation that runs on CPU and
defines correct behaviour. Furthest point sampling, the ball and cylinder queries and
kNN can also run on optional compiled kernels, which are used only after reproducing
the reference exactly; :mod:`grasppanda.ops.backends` selects and reports them.
"""

from . import backends
from .interpolate import inverse_distance_weights, three_interpolate
from .nms import grasp_nms
from .query import ball_query, box_query, cylinder_query, gather_points, group_points, knn
from .sampling import furthest_point_sample, generalized_furthest_point_sample
from .serialization import SerializationOrder, hilbert_code, serialization_code, z_order_code
from .views import (
    approach_to_rotation,
    fibonacci_views,
    healpix_views,
    icosphere_views,
    nearest_view,
    rotation_angle,
    two_vector_rotation,
)
from .voxel import VoxelGrid, voxelize

__all__ = [
    "SerializationOrder",
    "VoxelGrid",
    "approach_to_rotation",
    "backends",
    "ball_query",
    "box_query",
    "cylinder_query",
    "fibonacci_views",
    "furthest_point_sample",
    "gather_points",
    "generalized_furthest_point_sample",
    "grasp_nms",
    "group_points",
    "healpix_views",
    "hilbert_code",
    "icosphere_views",
    "inverse_distance_weights",
    "knn",
    "nearest_view",
    "rotation_angle",
    "serialization_code",
    "three_interpolate",
    "two_vector_rotation",
    "voxelize",
    "z_order_code",
]
