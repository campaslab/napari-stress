import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def draw_chronological_kde_plot(
    df: pd.DataFrame,
    x: str,
    hue: str = "time",
    ax: plt.Axes = None,
    colormap: str = "flare",
    legend: bool = True,
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a KDE plot of the data in df, with x on the x-axis and hue as the
    color. The data is assumed to be chronological, so the hue is used to
    color the data in the order it was collected.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    x : str
        Column name to plot on the x-axis
    hue : str, optional
        Column name to use for coloring the data, by default 'time'
    ax : plt.Axes, optional
        Axes to plot on, by default None
    colormap : str, optional
        Colormap to use for coloring the data, by default 'flare'

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    ax : plt.Axes
        Axes containing the plot
    """

    if ax is None:
        fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=x, hue=hue, ax=ax, palette=colormap, legend=False)

    if legend:
        ax.legend()

    if grid:
        ax.grid(which="both", linestyle="--", alpha=0.5, color="grey")
    else:
        plt.grid(False)

    return ax.figure, ax


def draw_chronological_lineplot_with_errors(
    df: pd.DataFrame,
    y: str,
    hue: str = None,
    x: str = "time",
    ax: plt.Axes = None,
    grid: bool = True,
    error: str = "se",
    estimator=np.mean,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a line plot of the data in df, with x on the x-axis and hue as the
    color. The data is assumed to be chronological, so the hue is used to
    color the data in the order it was collected.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    y : str
        Column name to plot on the y-axis
    hue : str, optional
        Column name to use for coloring the data, by default None
    x : str, optional
        Value to plot on the x-axis, by default 'time'
    ax : plt.Axes, optional
        Axes to plot on, by default None
    grid : bool, optional
        Whether to show the grid, by default True
    error : str, optional
        Type of error to show, by default 'se'
    estimator : function, optional
        Function to use for estimating the error, by default np.mean
    **kwargs : dict
        Additional arguments to pass to sns.lineplot

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    ax : plt.Axes
        Axes containing the plot
    """

    if ax is None:
        fig, ax = plt.subplots()

    sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        estimator=estimator,
        errorbar=error,
        marker="o",
        **kwargs,
    )

    if grid:
        ax.grid(which="both", linestyle="--", alpha=0.5, color="grey")
    else:
        plt.grid(False)

    return ax.figure, ax


def create_all_stress_plots(
    results_stress_analysis: list, time_step: float, n_frames: int
) -> dict:
    import matplotlib as mpl
    from .._utils._aggregate_measurements import find_metadata_in_layers
    from .. import types
    from .._utils._aggregate_measurements import compile_data_from_layers

    # Compile data
    (
        df_over_time,
        df_nearest_pairs,
        df_all_pairs,
        df_autocorrelations,
    ) = compile_data_from_layers(
        results_stress_analysis, time_step=time_step, n_frames=n_frames
    )

    # PLOTS
    mpl.style.use("default")
    # Fit residue
    df = find_metadata_in_layers(
        results_stress_analysis, types._METADATAKEY_FIT_RESIDUE
    )
    df["time"] = df["frame"] * time_step
    fig_residue, axes = plt.subplots(ncols=2, figsize=(10, 5))
    draw_chronological_kde_plot(df, x="fit_residue", hue="time", ax=axes[0])
    draw_chronological_lineplot_with_errors(df, y="fit_residue", ax=axes[1], error="sd")

    # Fit quality
    fig_GaussBonnet_error, axes = plt.subplots(ncols=2, figsize=(10, 5))
    draw_chronological_lineplot_with_errors(
        df_over_time, y=types._METADATAKEY_GAUSS_BONNET_REL, ax=axes[0]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, y=types._METADATAKEY_GAUSS_BONNET_ABS, ax=axes[1]
    )

    # Curvature
    df = find_metadata_in_layers(
        [results_stress_analysis[4]], types._METADATAKEY_MEAN_CURVATURE
    )
    df["time"] = df["frame"] * time_step
    fig_mean_curvature, axes = plt.subplots(ncols=2, figsize=(10, 5))
    draw_chronological_kde_plot(
        df,
        x=types._METADATAKEY_MEAN_CURVATURE,
        hue="time",
        ax=axes[0],
        colormap="flare",
    )
    draw_chronological_lineplot_with_errors(
        df, y=types._METADATAKEY_MEAN_CURVATURE, x="time", ax=axes[1], error="sd"
    )

    # Total stress
    df = find_metadata_in_layers(
        results_stress_analysis, types._METADATAKEY_STRESS_TOTAL
    )
    df["time"] = df["frame"] * time_step
    fig_total_stress, axes = plt.subplots(ncols=3, figsize=(13, 5))
    axes = axes.flatten()
    draw_chronological_kde_plot(
        df=df, x=types._METADATAKEY_STRESS_TOTAL, ax=axes[0], legend=False
    )
    draw_chronological_lineplot_with_errors(
        df=df, x="time", y=types._METADATAKEY_STRESS_TOTAL, ax=axes[1], error="sd"
    )
    draw_chronological_lineplot_with_errors(
        df=df_over_time,
        x="time",
        y=types._METADATAKEY_STRESS_TOTAL_ANISO,
        ax=axes[2],
        error="sd",
    )

    # Cell-scale
    df = find_metadata_in_layers(
        results_stress_analysis, types._METADATAKEY_STRESS_CELL
    )
    df["time"] = df["frame"] * time_step
    fig_cell_stress, axes = plt.subplots(ncols=3, figsize=(13, 5))
    axes = axes.flatten()
    draw_chronological_kde_plot(
        df=df, x=types._METADATAKEY_STRESS_CELL, ax=axes[0], legend=False
    )
    draw_chronological_lineplot_with_errors(
        df=df, x="time", y=types._METADATAKEY_STRESS_CELL, ax=axes[1], error="sd"
    )
    draw_chronological_lineplot_with_errors(
        df=df_over_time,
        x="time",
        y=types._METADATAKEY_STRESS_CELL_ANISO,
        ax=axes[2],
        error="sd",
    )

    # Tissue-scale (only anisotropy)
    fig_tissue_stress, ax = plt.subplots()
    draw_chronological_lineplot_with_errors(
        df=df_over_time,
        x="time",
        y=types._METADATAKEY_STRESS_TISSUE_ANISO,
        ax=ax,
        error="sd",
    )

    # ellipsoidal stress tensor
    fig_stress_tensor, axes = plt.subplots(ncols=3, figsize=(13, 5))
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_TENSOR_ELLI_E1, ax=axes[0]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_TENSOR_ELLI_E2, ax=axes[0]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_TENSOR_ELLI_E3, ax=axes[0]
    )

    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_ELLIPSOID_ANISO_E12, ax=axes[1]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_ELLIPSOID_ANISO_E13, ax=axes[1]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_STRESS_ELLIPSOID_ANISO_E23, ax=axes[1]
    )

    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_ANGLE_ELLIPSOID_CART_E1, ax=axes[2]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_ANGLE_ELLIPSOID_CART_E2, ax=axes[2]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_ANGLE_ELLIPSOID_CART_E3, ax=axes[2]
    )

    # All pairs
    fig_all_pairs, axes = plt.subplots(ncols=4, figsize=(20, 5))
    draw_chronological_kde_plot(
        df_all_pairs,
        x=types._METADATAKEY_STRESS_CELL_ALL_PAIR_DIST,
        ax=axes[0],
        legend=False,
    )
    draw_chronological_kde_plot(
        df_all_pairs,
        types._METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO,
        ax=axes[1],
        legend=False,
    )
    draw_chronological_kde_plot(
        df_nearest_pairs,
        types._METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST,
        ax=axes[2],
        legend=False,
    )
    draw_chronological_kde_plot(
        df_nearest_pairs, types._METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO, ax=axes[3]
    )

    # Spatial autocorrelations
    fig_spatial_autocorrelation, axes = plt.subplots(ncols=3, figsize=(15, 5))
    draw_chronological_lineplot_with_errors(
        df=df_autocorrelations,
        x="distances",
        y=types._METADATAKEY_AUTOCORR_SPATIAL_TOTAL,
        hue="time",
        ax=axes[0],
        markersize=0,
    )
    draw_chronological_lineplot_with_errors(
        df=df_autocorrelations,
        x="distances",
        y=types._METADATAKEY_AUTOCORR_SPATIAL_CELL,
        hue="time",
        ax=axes[1],
        markersize=0,
    )
    draw_chronological_lineplot_with_errors(
        df=df_autocorrelations,
        x="distances",
        y=types._METADATAKEY_AUTOCORR_SPATIAL_TISSUE,
        hue="time",
        ax=axes[2],
        markersize=0,
    )

    # Temoral correlations
    fig_temporal_autocorrelation, axes = plt.subplots(ncols=3, figsize=(15, 5))
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_AUTOCORR_TEMPORAL_TOTAL, ax=axes[0]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_AUTOCORR_TEMPORAL_CELL, ax=axes[1]
    )
    draw_chronological_lineplot_with_errors(
        df_over_time, types._METADATAKEY_AUTOCORR_TEMPORAL_TISSUE, ax=axes[2]
    )

    # Ellipsoid contribution
    data = df_over_time[types._METADATAKEY_ELIPSOID_DEVIATION_CONTRIB].values
    fig_ellipsoid_contribution, axes = plt.subplots(
        ncols=4, nrows=n_frames // 4 + 1, figsize=(12, n_frames)
    )
    for t, ax in enumerate(axes.flatten()):
        if t >= n_frames:
            ax.axis("off")
            continue
        ax.imshow(np.triu(data[t]), cmap="inferno")
        ax.tick_params(
            labelbottom=False, labeltop=True, labelleft=False, labelright=True
        )
        ax.set_xlabel("Degree")
        ax.set_ylabel("Order")
        ax.set_title(f"Time-step: {t}")

    figures = {
        "Figure_reside": {"figure": fig_residue, "path": "fit_residues.png"},
        "fig_GaussBonnet_error": {
            "figure": fig_GaussBonnet_error,
            "path": "gauss_bonnet_errors.png",
        },
        "fig_mean_curvature": {
            "figure": fig_mean_curvature,
            "path": "mean_curvatures.png",
        },
        "fig_total_stress": {"figure": fig_total_stress, "path": "Stresses_total.png"},
        "fig_cell_stress": {"figure": fig_cell_stress, "path": "Stresses_cell.png"},
        "fig_tissue_stress": {
            "figure": fig_tissue_stress,
            "path": "Stresses_tissue.png",
        },
        "fig_stress_tensor": {"figure": fig_stress_tensor, "path": "Stress_tensor.png"},
        "fig_all_pairs": {
            "figure": fig_all_pairs,
            "path": "Autocorrelations_spatial_all_pairs.png",
        },
        "fig_spatial_autocorrelation": {
            "figure": fig_spatial_autocorrelation,
            "path": "Autocorrelations_spatial_nearest_pairs.png",
        },
        "fig_temporal_autocorrelation": {
            "figure": fig_temporal_autocorrelation,
            "path": "Autocorrelations_temporal.png",
        },
        "fig_ellipsoid_contribution": {
            "figure": fig_ellipsoid_contribution,
            "path": "Ellipsoid_contribution.png",
        },
    }

    return figures
