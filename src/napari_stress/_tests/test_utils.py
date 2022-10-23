import numpy as np
from napari.types import LayerData, PointsData, SurfaceData, ImageData, VectorsData, LayerDataTuple
from napari.layers import Layer, Points

def test_fit_functions():
    from napari_stress._utils.fit_utils import _sigmoid, _gaussian, _detect_maxima, _detect_max_gradient

    x = np.arange(0, 100, 1)

    y = _sigmoid(x, center=50, amplitude=1.0, slope=0.5, offset=0)
    assert np.max(y) <= 1.0
    assert y[51] > 0.5 and y[49] < 0.5

    argdrop = _detect_max_gradient(y)
    assert 49 <= argdrop <= 51

    y = _gaussian(x, center=50, sigma=2.0, amplitude=2.0)
    assert np.max(y) <= 2.0

    argmax = _detect_maxima(y)
    assert 49 <= argmax <= 51

def test_decorator_points():
    from napari_stress import TimelapseConverter
    from vedo import Sphere

    Converter = TimelapseConverter()

    points_list = [Sphere().points() * k for k in np.arange(1.9, 2.1, 0.1)]
    points_array = Converter.list_of_data_to_data(points_list, PointsData)
    points_list_conv = Converter.data_to_list_of_data(points_array, PointsData)

    for pts, _pts in zip(points_list, points_list_conv):
        assert np.array_equal(points_list, points_list_conv)

def test_decorator_points_layerdatatuple():
    from napari_stress import TimelapseConverter, get_droplet_point_cloud

    Converter = TimelapseConverter()
    list_of_ldtuples = []
    for i in range(10):
        points = list(get_droplet_point_cloud()[0])
        features = {'feature1': np.random.random(len(points[0]))}
        metadata = {'data1': f'test_{i}'}

        points[0] = points[0][:, 1:]
        points[1]['features'] = features
        points[1]['metadata'] = metadata
        list_of_ldtuples.append(points)

    ldtuple_4d = Converter.list_of_data_to_data(list_of_ldtuples, layertype=LayerDataTuple)

    assert 'data1' in ldtuple_4d[1]['metadata'].keys()
    assert ldtuple_4d[0][-1, 0] == 9

def test_decorator_points_layers():

    from napari_stress import TimelapseConverter, get_droplet_point_cloud_4d

    Converter = TimelapseConverter()
    ldt = get_droplet_point_cloud_4d()[0]
    n_frames = int(ldt[0][:, 0].max()) + 1

    # create some dummy metadata
    metadata = {'key1': [f'data_{i}' for i in range(n_frames)]}
    ldt[1]['metadata'] = metadata

    # create some dummy features
    features = np.random.random(len(ldt[0]))
    ldt[1]['features'] = features

    layer_4d = Layer.create(data=ldt[0], meta=ldt[1], layer_type=ldt[2])
    list_of_layers = Converter.data_to_list_of_data(layer_4d, layertype=Points)

def test_decorator_surfaces():
    from napari_stress import TimelapseConverter, frame_by_frame
    from napari_process_points_and_surfaces import sample_points_poisson_disk
    from napari.types import SurfaceData
    from vedo import Sphere

    Converter = TimelapseConverter()

    # create a list of surfaces with points/faces/values
    surface_list = [
        (Sphere().points() * k,
         np.asarray(Sphere().faces()),
         k * np.ones(Sphere().N())) for k in np.arange(1.9, 2.1, 0.1)
        ]
    surface_array = Converter.list_of_data_to_data(surface_list, SurfaceData)
    assert len(surface_array) == 3

    surface_list_conv = Converter.data_to_list_of_data(surface_array, SurfaceData)

    for surf, _surf in zip(surface_list, surface_list_conv):
        assert np.array_equal(surf[0], _surf[0])
        assert np.array_equal(surf[1], _surf[1])

    points = frame_by_frame(sample_points_poisson_disk)(surface_array)
    points = frame_by_frame(sample_points_poisson_disk)(surface_list[0])

def test_decorator_images():

    from napari_stress import TimelapseConverter

    Converter = TimelapseConverter()

    image = np.random.random(size=(50,50,50))
    image_list = [k * image for k in np.arange(0.1, 1, 0.1)]

    image_array = Converter.list_of_data_to_data(image_list, ImageData)

    assert image_array.shape[0] == len(image_list)

    image_list_conv = Converter.list_of_data_to_data(image_array, ImageData)
    for img, _img in zip(image_list, image_list_conv):
        assert np.array_equiv(img, _img)

def test_frame_by_frame_vectors():

    from napari_stress import TimelapseConverter
    Converter = TimelapseConverter()

    # make some points
    points = np.random.random((1000, 4))
    vectors = np.random.random((1000, 4))

    points[:, 0] = np.arange(0, 5, 1).repeat(200)
    vectors[:, 0] = np.arange(0, 5, 1).repeat(200)

    vectors_4d = np.stack((points, vectors)).transpose((1,0,2))

    vectors_list = Converter.data_to_list_of_data(vectors_4d, VectorsData)
    vectors_data = Converter.list_of_data_to_data(vectors_list, VectorsData)

    assert np.array_equal(vectors_data, vectors_4d)


if __name__ =='__main__':
    test_decorator_points_layers()
