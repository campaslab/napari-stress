from ._utils.frame_by_frame import frame_by_frame
import pandas as pd
import numpy as np

@frame_by_frame
def normal_vectors_on_pointcloud(points: "napari.types.PointsData") -> "napari.types.VectorsData":
    """
    Calculate normal vectors on pointcloud.

    Args:
        pointcloud (napari.types.PointsData): Pointcloud

    Returns:
        napari.types.VectorsData: Normal vectors
    """
    import vedo

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.compute_normals_with_pca(
        orientation_point=pointcloud.centerOfMass()
        )
    
    normals = pointcloud.pointdata['Normals']
    normals = np.stack([points, normals]).transpose((1,0,2))

    return normals

@frame_by_frame
def normal_vectors_on_surface(surface: "napari.types.SurfaceData") -> "napari.types.VectorsData":
    """
    Calculate normal vectors on surface.

    Args:
        surface (napari.types.SurfaceData): Surface mesh

    Returns:
        napari.types.VectorsData: Normal vectors
    """
    import vedo

    surface = vedo.mesh.Mesh((surface[0], surface[1]))
    surface.compute_normals()
    normals = surface.pointdata['Normals']

    normals = np.stack([surface.points(), normals]).transpose((1,0,2))

    return normals

def sample_intensity_along_vector(
    vectors: "napari.types.VectorsData",
    image: "napari.types.ImageData",
    sampling_distance: float = 1.0,
    interpolation_method: str = 'linear') -> pd.DataFrame:
    """
    Sample intensity along vectors of equal length.

    Args:
        vectors (napari.types.VectorsData): Vectors along which to measure intensity
        image (napari.types.ImageData): Image to sample
        sampling_distance (float, optional): Distance between samples. Defaults to 1.0.
        interpolation_method (str, optional): Interpolation method. Defaults to 'linear'.
    """
    from scipy.interpolate import RegularGridInterpolator    

    # Create coords for interpolator
    X1 = np.arange(0, image.shape[0], 1)
    X2 = np.arange(0, image.shape[1], 1)
    X3 = np.arange(0, image.shape[2], 1)
    interpolator = RegularGridInterpolator((X1, X2, X3),
                                           image,
                                           bounds_error=False,
                                           fill_value=np.nan,
                                           method=interpolation_method)
    
    # get start point and unit normal vector
    start_points = vectors[:, 0]
    unit_vector = vectors[:, 1]/np.linalg.norm(vectors[:, 1], axis=1)[:, np.newaxis]

    # calculate number of steps
    steps = np.round(np.linalg.norm(vectors[:, 1], axis=1)[:, np.newaxis] / sampling_distance).mean().astype(int)

    intensity = np.zeros((len(start_points), steps))
    for step in range(steps):
        sampling_coordinates = np.stack([start_points[idx] + step * sampling_distance * unit_vector[idx] for idx in range(len(start_points))])
        intensity[:, step] = interpolator(sampling_coordinates)

    intensity = pd.DataFrame(intensity)
    intensity.columns = [f'step_{idx}' for idx in range(steps)]

    return intensity
