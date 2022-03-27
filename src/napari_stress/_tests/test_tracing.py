def test_tracing(make_napari_viewer):
    from skimage import filters
    import numpy as np
    import napari_process_points_and_surfaces as nppas
    from napari_stress._refine_surfaces import trace_refinement_of_surface

    viewer = make_napari_viewer()

    # Make a blurry sphere
    s = 100
    data = np.zeros((s, s, s), dtype=float)
    x0 = 50
    radius = 15

    for x in range(s):
        for y in range(s):
            for z in range(s):
                if np.sqrt((x-x0)**2 + (y-x0)**2 + (z-x0)**2) < radius:
                    data[x, y, z] = 1.0

    image = filters.gaussian(data, sigma=1.5)
    viewer.add_image(image)

    binary = image > filters.threshold_otsu(image)
    viewer.add_labels(binary)

    surface = nppas.label_to_surface(binary.astype(int))
    viewer.add_surface(surface)

    new_points = trace_refinement_of_surface(image,
                                             surface,
                                             fit_method=1,
                                             trace_length=5)

if __name__ == '__main__':
    import napari
    test_tracing(napari.Viewer)
