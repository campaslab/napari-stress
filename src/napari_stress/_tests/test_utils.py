import numpy as np
from napari.types import (
    PointsData,
    ImageData,
    SurfaceData,
    VectorsData,
    LayerDataTuple,
)
from napari.layers import Layer, Points


def test_fit_functions():
    from napari_stress._reconstruction.fit_utils import (
        _sigmoid,
        _gaussian,
        _detect_maxima,
        _detect_max_gradient,
    )

    x = np.arange(0, 100, 1)

    y = _sigmoid(x, center=50, amplitude=1.0, slope=0.5, offset=0, background_slope=0)
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
    points_array_4d = Converter.list_of_data_to_data(points_list, PointsData)
    points_array_3d = Converter.list_of_data_to_data([points_list[0]], PointsData)
    points_list_conv = Converter.data_to_list_of_data(points_array_4d, PointsData)

    assert np.array_equal(points_array_3d, points_list[0])

    for pts, _pts in zip(points_list, points_list_conv):
        assert np.array_equal(pts, _pts)


def test_decorator_points_layerdatatuple():
    from napari_stress import TimelapseConverter, get_droplet_point_cloud
    import pandas as pd
    from copy import deepcopy

    np.random.seed(42)

    Converter = TimelapseConverter()
    list_of_ldtuples = []
    for i in range(10):
        points = list(get_droplet_point_cloud()[0])
        features = pd.DataFrame({"feature1": np.random.random(len(points[0]))})
        metadata = {"data1": f"test_{i}"}

        points[0] = points[0][:, 1:]
        points[1]["features"] = features
        points[1]["metadata"] = metadata
        list_of_ldtuples.append(points)

    list_of_ldtuples_copy = deepcopy(list_of_ldtuples)

    list_of_ldtuples_copy = deepcopy(list_of_ldtuples)

    ldtuple_4d = Converter.list_of_data_to_data(
        list_of_ldtuples, layertype=LayerDataTuple
    )
    ldtuple_3d = Converter.list_of_data_to_data(
        [list_of_ldtuples[0]], layertype=PointsData
    )

    assert np.array_equal(ldtuple_3d[0][0], list_of_ldtuples[0][0])
    assert "data1" in ldtuple_4d[1]["metadata"].keys()
    assert ldtuple_4d[0][-1, 0] == 9

    list_of_ldtuples_conv = Converter.data_to_list_of_data(
        ldtuple_4d, layertype="napari.types:LayerDataTuple"
    )
    for ldt, _ldt in zip(list_of_ldtuples_copy, list_of_ldtuples_conv):
        assert np.array_equal(ldt[0], _ldt[0])
        assert np.array_equal(
            ldt[1]["features"]["feature1"], _ldt[1]["features"]["feature1"]
        )
        assert np.array_equal(ldt[1]["metadata"]["data1"], _ldt[1]["metadata"]["data1"])


def test_decorator_points_layers():
    from napari_stress import TimelapseConverter, get_droplet_point_cloud_4d

    Converter = TimelapseConverter()
    ldt = get_droplet_point_cloud_4d()[0]
    n_frames = int(ldt[0][:, 0].max()) + 1

    # create some dummy metadata
    metadata = {"key1": [f"data_{i}" for i in range(n_frames)]}
    ldt[1]["metadata"] = metadata

    # create some dummy features
    features = np.random.random(len(ldt[0]))
    ldt[1]["features"] = features

    layer_4d = Layer.create(data=ldt[0], meta=ldt[1], layer_type=ldt[2])
    list_of_layers = Converter.data_to_list_of_data(layer_4d, layertype=Points)

    assert len(list_of_layers) == n_frames


def test_decorator_surfaces():
    from napari_stress import TimelapseConverter, frame_by_frame
    from napari_process_points_and_surfaces import sample_points_from_surface
    from vedo import Sphere

    Converter = TimelapseConverter()

    surface_list = [
        (
            Sphere().points() * k,
            np.asarray(Sphere().cells),
            k * np.ones(Sphere().npoints),
        )
        for k in np.arange(1.9, 2.1, 0.1)
    ]
    n_frames = len(surface_list)
    surface_array_4d = Converter.list_of_data_to_data(surface_list, SurfaceData)
    surface_array_3d = Converter.list_of_data_to_data([surface_list[0]], SurfaceData)

    assert np.array_equal(surface_array_3d[0], surface_list[0][0])
    assert len(surface_array_4d) == 3

    surface_list_conv = Converter.data_to_list_of_data(surface_array_4d, SurfaceData)

    for surf, _surf in zip(surface_list, surface_list_conv):
        assert np.array_equal(surf[0], _surf[0])
        assert np.array_equal(surf[1], _surf[1])

    points_4d = frame_by_frame(sample_points_from_surface)(surface_array_4d)
    points_3d = frame_by_frame(sample_points_from_surface)(surface_list[0])

    assert np.array_equal(points_3d, points_4d[points_4d[:, 0] == 0][:, 1:])
    assert np.array_equal(points_4d.shape, (1010 * n_frames, 4))


def test_decorator_images():
    from napari_stress import TimelapseConverter

    Converter = TimelapseConverter()

    image = np.random.random(size=(50, 50, 50))
    image_list = [k * image for k in np.arange(0.1, 1, 0.1)]

    image_array_4d = Converter.list_of_data_to_data(image_list, ImageData)
    image_array_3d = Converter.list_of_data_to_data([image_list[0]], ImageData)

    assert np.array_equal(image_array_3d.squeeze(), image_list[0])
    assert image_array_4d.shape[0] == len(image_list)

    image_list_conv = Converter.list_of_data_to_data(image_array_4d, ImageData)
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

    vectors_4d = np.stack((points, vectors)).transpose((1, 0, 2))

    vectors_list = Converter.data_to_list_of_data(vectors_4d, VectorsData)
    vectors_data_4d = Converter.list_of_data_to_data(vectors_list, VectorsData)
    vectors_data_3d = Converter.list_of_data_to_data([vectors_list[0]], VectorsData)

    assert np.array_equal(vectors_data_3d, vectors_list[0])
    assert np.array_equal(vectors_data_4d, vectors_4d)


def test_frame_by_frame_dataframes():
    from napari_stress import TimelapseConverter
    import pandas as pd

    Converter = TimelapseConverter()

    # make some dataframes
    df1 = pd.DataFrame({"data": np.random.random(100)})
    df2 = pd.DataFrame({"data": np.random.random(100)})
    df3 = pd.DataFrame({"data": np.random.random(100)})

    single_df = Converter.list_of_data_to_data([df1, df2, df3], layertype=pd.DataFrame)
    list_of_dfs = Converter.data_to_list_of_data(single_df, layertype=pd.DataFrame)

    assert np.array_equal(list_of_dfs[0]["data"], df1["data"])
    assert np.array_equal(list_of_dfs[1]["data"], df2["data"])
    assert np.array_equal(list_of_dfs[2]["data"], df3["data"])
