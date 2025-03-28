"""
Measurements subpackage for napari-stress.

This subpackage contains functions for measuring curvature, stress, and
temporal correlations on surfaces.
"""

from .curvature import (
    average_mean_curvatures_on_manifold,
    calculate_mean_curvature_on_manifold,
    calculate_patch_fitted_curvature_on_pointcloud,
    calculate_patch_fitted_curvature_on_surface,
    curvature_on_ellipsoid,
    gauss_bonnet_test,
    mean_curvature_differences_radial_cartesian_manifolds,
    mean_curvature_on_ellipse_cardinal_points,
    mean_curvature_on_radial_manifold,
    radial_surface_averaged_mean_curvature,
)
from .deviation_analysis import deviation_from_ellipsoidal_mode
from .geodesics import (
    correlation_on_surface,
    geodesic_distance_matrix,
    geodesic_path,
    local_extrema_analysis,
)
from .intensity import (
    measure_intensity_on_surface,
    sample_intensity_along_vector,
)
from .measurements import distance_to_k_nearest_neighbors
from .stresses import (
    anisotropic_stress,
    calculate_anisotropy,
    maximal_tissue_anisotropy,
    tissue_stress_tensor,
)
from .temporal_correlation import (
    haversine_distances,
    spatio_temporal_autocorrelation,
    temporal_autocorrelation,
)
from .toolbox import comprehensive_analysis

__all__ = [
    "calculate_mean_curvature_on_manifold",
    "mean_curvature_on_radial_manifold",
    "average_mean_curvatures_on_manifold",
    "radial_surface_averaged_mean_curvature",
    "curvature_on_ellipsoid",
    "mean_curvature_on_ellipse_cardinal_points",
    "gauss_bonnet_test",
    "mean_curvature_differences_radial_cartesian_manifolds",
    "calculate_patch_fitted_curvature_on_pointcloud",
    "calculate_patch_fitted_curvature_on_surface",
    "deviation_from_ellipsoidal_mode",
    "temporal_autocorrelation",
    "haversine_distances",
    "spatio_temporal_autocorrelation",
    "geodesic_distance_matrix",
    "geodesic_path",
    "correlation_on_surface",
    "local_extrema_analysis",
    "anisotropic_stress",
    "tissue_stress_tensor",
    "maximal_tissue_anisotropy",
    "calculate_anisotropy",
    "comprehensive_analysis",
    "distance_to_k_nearest_neighbors",
    "sample_intensity_along_vector",
    "measure_intensity_on_surface",
]
