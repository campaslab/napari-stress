import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np
import pandas as pd
from napari.types import ImageData, LayerDataTuple, PointsData
from napari_tools_menu import register_function

from .._utils import frame_by_frame
from .fit_utils import (
    _function_args_to_list,
    edge_functions,
    fit_types,
    interpolation_types,
)

warnings.filterwarnings("ignore")


@register_function(menu="Points > Trace-refine points (n-STRESS)")
@frame_by_frame
def trace_refinement_of_surface(
    intensity_image: ImageData,
    normal_vectors: 'napari.types.VectorsData',
    selected_fit_type: fit_types = fit_types.fancy_edge_fit,
    selected_edge: edge_functions = edge_functions.interior,
    sampling_distance: float = 0.5,
    interpolation_method: interpolation_types = interpolation_types.cubic,
) -> 'napari.types.LayerDataTuple':
    """
    Generate intensity profiles along traces.

    This function receives an intensity image and a pointcloud with points on
    the surface of an object in the intensity image. It assumes that the
    pointcloud corresponds to the vertices on a surface around that object.

    As a first step, the function calculates normals on the surface and
    multiplies the length of this vector with `trace_length`. The intensity of
    the input image is then sampled along this vector perpendicular to the
    surface with a distance of `sampling distance` between each point along the
    normal vector.

    The location of the object's surface is then determined by fitting a
    selected function to the intensity profile along the prolonged normal
    vector.

    Parameters
    ----------
    intensity_image : ImageData
        Intensity image
    normal_vectors: 'napari.types.VectorsData'
        VectorsData containing the normal vectors at the points on the surface.
    selected_fit_type : fit_types, optional
        Type of fit to use for determining the surface location, by default
        fit_types.fancy_edge_fit
    selected_edge : edge_functions, optional
        Type of edge to detect, by default edge_functions.interior
    trace_length : float, optional
        Length of the normal vector, by default 10.0
    sampling_distance : float, optional
        Distance between each point along the normal vector, by default 0.5
    remove_outliers : bool, optional
        Whether to remove outliers from the intensity profile, by default True
    outlier_tolerance : float, optional
        Tolerance for outlier removal, by default 1.5
    interpolation_method : interpolation_types, optional
        Interpolation method to use for sampling the intensity image, by
        default interpolation_types.cubic

    Returns
    -------
    List[LayerDataTuple]
        List of napari layer data tuples
    """
    from .._measurements.intensity import sample_intensity_along_vector
    from .._measurements.measurements import distance_to_k_nearest_neighbors
    from .fit_utils import (
        _fancy_edge_fit,
        _mean_squared_error,
    )

    if isinstance(selected_fit_type, str):
        selected_fit_type = fit_types(selected_fit_type)

    if isinstance(interpolation_method, str):
        interpolation_method = interpolation_types(interpolation_method)

    if isinstance(selected_edge, str):
        edge_detection_function = edge_functions.__members__[
            selected_edge
        ].value[selected_fit_type.value]
    else:
        edge_detection_function = selected_edge.value[selected_fit_type.value]

    # Define start and end points for the surface tracing vectors
    vector_lengths = np.linalg.norm(normal_vectors[:, 1], axis=1)
    n_samples = np.ceil(vector_lengths / sampling_distance).astype(np.int32)
    # Define trace vectors (full length and single step
    start_points = normal_vectors[:, 0]
    vector_step = normal_vectors[:, 1] / n_samples[:, np.newaxis]

    # measure intensity along the vectors
    intensity_along_vector = sample_intensity_along_vector(
        normal_vectors,
        intensity_image,
        sampling_distance=sampling_distance,
        interpolation_method=interpolation_method.value,
    )

    # Allocate arrays for results (location of object border, fit parameters,
    # fit errors, and intensity profiles)
    fit_parameters = _function_args_to_list(edge_detection_function)[1:]
    fit_errors = [p + "_err" for p in fit_parameters]
    columns = ["idx_of_border"] + fit_parameters + fit_errors

    if len(fit_parameters) == 1:
        fit_parameters, fit_errors = [fit_parameters[0]], [fit_errors[0]]

    # create empty dataframe to keep track of results
    fit_data = pd.DataFrame(columns=columns, index=np.arange(len(normal_vectors)))

    # Iterate over all provided target points
    for idx in range(len(normal_vectors)):
        array = np.array(intensity_along_vector.loc[idx].to_numpy())
        # Simple or fancy fit?
        if selected_fit_type == fit_types.quick_edge_fit:
            idx_of_border = edge_detection_function(array)
            perror = 0
            popt = 0
            MSE = np.array([0, 0])

        elif selected_fit_type == fit_types.fancy_edge_fit:
            popt, perror = _fancy_edge_fit(
                array, selected_edge_func=edge_detection_function
            )
            idx_of_border = popt[0]

            # calculate fit errors
            MSE = _mean_squared_error(
                fit_function=edge_detection_function,
                x=np.arange(len(array)),
                y=array,
                fit_params=popt,
            )

        new_point = start_points[idx] + idx_of_border * vector_step[idx]

        fit_data.loc[idx, fit_errors] = perror
        fit_data.loc[idx, fit_parameters] = popt
        fit_data.loc[idx, "idx_of_border"] = idx_of_border
        fit_data.loc[idx, "surface_points_x"] = new_point[0]
        fit_data.loc[idx, "surface_points_y"] = new_point[1]
        fit_data.loc[idx, "surface_points_z"] = new_point[2]
        fit_data.loc[idx, "mean_squared_error"] = MSE[0]
        fit_data.loc[idx, "fraction_variance_unexplained"] = MSE[1]
        fit_data.loc[idx, "fraction_variance_unexplained_log"] = np.log(MSE[1])

    fit_data["start_points"] = list(start_points)

    # get indeces of rows with nans
    no_nan_idx = np.where(~np.isnan(fit_data["surface_points_x"].to_numpy()))[
        0
    ]
    fit_data = fit_data.iloc[no_nan_idx]
    intensity_along_vector = intensity_along_vector.iloc[no_nan_idx]

    # measure distance to nearest neighbor
    fit_data["distance_to_nearest_neighbor"] = distance_to_k_nearest_neighbors(
        fit_data[
            ["surface_points_x", "surface_points_y", "surface_points_z"]
        ].to_numpy(),
        k=15,
    )

    # reformat to layerdatatuple: points
    feature_names = (
        fit_parameters
        + fit_errors
        + [
            "distance_to_nearest_neighbor",
            "mean_squared_error",
            "fraction_variance_unexplained",
            "fraction_variance_unexplained_log",
            "idx_of_border",
        ]
    )
    features = fit_data[feature_names].to_dict("list")
    metadata = {"intensity_profiles": intensity_along_vector}
    properties = {
        "name": "Refined_points",
        "size": 1,
        "features": features,
        "metadata": metadata,
        "face_color": "cyan",
    }
    data = fit_data[
        ["surface_points_x", "surface_points_y", "surface_points_z"]
    ].to_numpy()
    fit_data.drop(
        columns=["surface_points_x", "surface_points_y", "surface_points_z"],
        inplace=True,
    )
    layer_points = (data, properties, "points")

    return layer_points


def remove_outliers(
        points: "napari.layers.Points",
        outlier_tolerance: float = 1.5) -> "napari.layers.Points":
    """
    Remove outliers from points based on the fraction of variance unexplained.

    Parameters
    ----------
    points : napari.layers.Points
        Points layer from which to remove outliers.
    outlier_tolerance : float, optional
        Tolerance factor for outlier detection, by default 1.5
    Returns
    -------
    napari.layers.Points
        New points layer with outliers removed.
    """
    from .fit_utils import _identify_outliers

    good_points = _identify_outliers(
        points.features,
        column_names=["fraction_variance_unexplained_log"],
        which=["above"],
        factor=outlier_tolerance,
        merge="or",
    )
    retained_points = points.data[good_points]
    retained_features = points.features.iloc[good_points]
    
    return napari.layers.Points(
        retained_points,
        features=retained_features,
        scale=points.scale,
        translate=points.translate,
        rotate=points.rotate,
    )




@register_function(menu="Points > Resample spherical pointcloud (n-STRESS)")
@frame_by_frame
def resample_pointcloud(
    points: "napari.types.PointsData", sampling_length: float = 5
) -> "napari.types.PointsData":
    """
    Resampe a spherical-like pointcloud on fibonacci grid.

    Parameters
    ----------
    points : PointsData
    sampling_length : float, optional
        Distance between sampled point locations. The default is 5.

    Returns
    -------
    resampled_points : TYPE

    """
    import vedo
    from scipy.interpolate import griddata

    from .fit_utils import _fibonacci_sampling

    # convert to spherical, relative coordinates
    center = np.mean(points, axis=0)
    points_centered = points - center
    points_spherical = vedo.cart2spher(
        points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
    ).T

    # estimate point number according to passed sampling length
    mean_radius = points_spherical[:, 0].mean()
    surface_area = mean_radius**2 * 4 * np.pi
    n = int(surface_area / sampling_length**2)

    # sample points on unit-sphere according to fibonacci-scheme
    sampled_points = _fibonacci_sampling(n)
    sampled_points = vedo.transformations.cart2spher(
        sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    ).T

    # interpolate cartesian coordinates on (theta, phi) grid
    theta_interpolation = np.concatenate(
        [
            points_spherical[:, 1],
            points_spherical[:, 1],
            points_spherical[:, 1],
        ]
    )
    phi_interpolation = np.concatenate(
        [
            points_spherical[:, 2] + 2 * np.pi,
            points_spherical[:, 2],
            points_spherical[:, 2] - 2 * np.pi,
        ]
    )

    new_x = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 0]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    new_y = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 1]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    new_z = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 2]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    resampled_points = np.stack([new_x, new_y, new_z]).T + center

    no_nan_idx = np.where(~np.isnan(resampled_points[:, 0]))[0]
    return resampled_points[no_nan_idx, :]
