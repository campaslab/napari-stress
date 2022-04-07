import numpy as np

def test_point_utils():
    from skimage import filters

    import napari_process_points_and_surfaces as nppas
    from napari_stress._utils import list_of_points_to_points,\
        points_to_list_of_points

    # distribute fibonacci points
    points = np.zeros((100, 4))
    points[:, 0] = np.repeat(np.arange(0, 50, 1), 2)

    list_of_points = points_to_list_of_points(points)
    assert  isinstance(list_of_points, list)

    _points = list_of_points_to_points(list_of_points)
    assert np.array_equal(points, _points)

def test_surf_utils():

    from napari_stress._utils import surface_to_list_of_surfaces,\
    list_of_surfaces_to_surface
    from vedo import Sphere

    list_of_surfaces = [
        (Sphere().points(), np.asarray(Sphere().faces(), dtype=int))
        ] * 10

    surfaces = list_of_surfaces_to_surface(list_of_surfaces)
    _list_of_surfaces = surface_to_list_of_surfaces(surfaces)

    for idx in range(10):
        assert np.array_equal(list_of_surfaces[idx][0], _list_of_surfaces[idx][0])
        assert np.array_equal(list_of_surfaces[idx][1][1], _list_of_surfaces[idx][1][1])



if __name__ == '__main__':
    test_surf_utils()
