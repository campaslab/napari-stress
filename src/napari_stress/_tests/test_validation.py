def test_validation(make_napari_viewer):
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from napari_stress import measurements, utils

    # directory of this file
    current_dir = Path(__file__).parent

    # Load sample data
    viewer = make_napari_viewer()
    layer = viewer.open_sample("napari-stress", sample="PC_2")[0]
    n_frames = int(layer.data[:, 0].max()) + 1
    time_step = 3

    results_stress_analysis = measurements.comprehensive_analysis(
        layer.data,
        max_degree=20,
        n_quadrature_points=434,
        use_dask=True,
        gamma=3.3,
    )

    # Compile data
    df_over_time, _, _, _, _ = utils.compile_data_from_layers(
        results_stress_analysis, n_frames=n_frames, time_step=time_step
    )

    results_dir = os.path.join(current_dir, "results_STRESS")
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv")
    ]

    df_STRESS = pd.DataFrame()
    for file in files:
        # read from file with data stored in row direction
        single_result = pd.read_csv(
            os.path.join(results_dir, file), header=None
        ).T

        df_STRESS[Path(file).stem] = single_result.values.flatten()

    to_compare = [
        "stress_total_anisotropy",
        "stress_cell_anisotropy",
        "stress_tissue_anisotropy",
    ]

    for item in to_compare:
        difference_absolute = df_STRESS[item] - df_over_time[item]
        difference_relative = np.abs(difference_absolute) / df_over_time[item]

        df_over_time[f"{item}_difference_absolute"] = difference_absolute
        df_over_time[f"{item}_difference_relative"] = difference_relative

        assert difference_relative.mean() < 0.1
