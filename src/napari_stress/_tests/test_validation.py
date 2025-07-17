def test_validation(make_napari_viewer):
    from napari_stress import measurements, sample_data, utils
    import os
    import pandas as pd
    import numpy as np
    
    # Load sample data
    viewer = make_napari_viewer()
    layer = viewer.open_sample('napari-stress', sample='PC_2')[0]
    n_frames = int(layer.data[:, 0].max()) + 1
    time_step = 3

    results_stress_analysis = measurements.comprehensive_analysis(
        layer.data, max_degree=20, n_quadrature_points=434,
        use_dask=True
    )

    # Compile data
    df_over_time, df_nearest_pairs, df_all_pairs, df_autocorrelations, ellipse_contrib = utils.compile_data_from_layers(
        results_stress_analysis, n_frames=n_frames, time_step=time_step)
    
    df_STRESS = pd.DataFrame()
    for file in os.listdir('./results_STRESS/'):
        # read from file with data stored in row direction
        single_result = pd.read_csv(f'./results_STRESS/{file}', header=None).T

        column_name = file.split('.')[0]
        df_STRESS[column_name] = single_result.values.flatten()

    to_compare = ['stress_total_anisotropy', 'stress_cell_anisotropy', 'stress_tissue_anisotropy']

    for item in to_compare:
        difference_absolute = df_STRESS[item] - df_over_time[item]
        difference_relative = np.abs(difference_absolute) / df_over_time[item]

        df_over_time[f'{item}_difference_absolute'] = difference_absolute
        df_over_time[f'{item}_difference_relative'] = difference_relative

        assert difference_relative.mean() < 0.1

