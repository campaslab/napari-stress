# -*- coding: utf-8 -*-

def test_compatibility_decorator():
    import inspect
    import numpy as np
    import napari_stress

    def my_function(manifold: napari_stress._stress.manifold_SPB.manifold, sigma: float = 1.0) -> (dict, dict):
        some_data = np.random.random((10,3))
        metadata = {'attribute': 1}
        metadata['manifold'] = manifold
        features = {'attribute2': np.random.random(10)}
        return features, metadata

    function = napari_stress.measurements.utils.naparify_measurement(my_function)
    sig = inspect.signature(function)

    assert sig.parameters['manifold'].annotation == 'napari.layers.Points'
