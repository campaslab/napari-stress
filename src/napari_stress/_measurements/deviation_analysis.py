from napari.types import PointsData, VectorsData

def deviation_from_ellipsoidal_mode(points: PointsData):
    from napari_stress import approximation
    
    # calculate errors
    ellipsoid = approximation.least_squares_ellipsoid(points)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, points)
    errors = approximation.pairwise_point_distances(points, ellipsoid_points)

    