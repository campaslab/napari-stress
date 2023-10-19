import pandas as pd
import numpy as np
from typing import List
from napari.types import LayerDataTuple


def compile_data_from_layers(
    results_stress_analysis: list, n_frames: int, time_step: float
) -> list:
    """
    Compile data from the results of the stress analysis into a list of
    dataframes.

    Parameters
    ----------
    results_stress_analysis : list
        List of tuples containing the results of the stress analysis
    n_frames : int
        Number of frames in the data
    time_step : float
        Time step between frames

    Returns
    -------
    df_over_time : pd.DataFrame
        Dataframe containing the singular values results, e.g. results
        from the stress analysis that refer to a single value per frame.
        Columns:
            time : float
                Time of the frame
            etc.
    df_nearest_pairs : pd.DataFrame
        Dataframe containing the nearest pair extrema results.
        Columns:
            time : float
                Time of the frame
            nearest_pair_distance : float
                Distance between the nearest pairs of extrema
            nearest_pair_anisotropy : float
                Stress Anisotropy of the nearest pair
    df_all_pairs : pd.DataFrame
        Dataframe containing the all pair extrema results.
        Columns:
            time : float
                Time of the frame
            all_pair_distance : float
                Distance between all pairs of extrema
            all_pair_anisotropy : float
                Stress Anisotropy of all pairs
    df_autocorrelations : pd.DataFrame
        Dataframe containing the spatial autocorrelation results.
        Columns:
            time : float
                Time of the frame
            distances : float
                Distances at which the autocorrelation was calculated
            autocorrelation_total : float
                Autocorrelation of the total stress
            autocorrelation_cell : float
                Autocorrelation of the cell stress
            autocorrelation_tissue : float
                Autocorrelation of the tissue stress
    """
    from .. import measurements
    from .. import types

    df_over_time = aggregate_singular_values(
        results_stress_analysis,
        n_frames=n_frames,
        time_step=time_step,
    )
    df_nearest_pairs, df_all_pairs = aggregate_extrema_results(
        results_stress_analysis, n_frames=n_frames, time_step=time_step
    )
    df_autocorrelations = aggregate_spatial_autocorrelations_results(
        results_stress_analysis, n_frames=n_frames, time_step=time_step
    )

    # Calculate anisotropies: Total
    df = find_metadata_in_layers(
        results_stress_analysis, types._METADATAKEY_STRESS_TOTAL
    )
    result = measurements.calculate_anisotropy(
        df, types._METADATAKEY_STRESS_TOTAL, alpha=0.05, group_column="frame"
    )
    df_over_time[types._METADATAKEY_STRESS_TOTAL_ANISO] = result[
        types._METADATAKEY_STRESS_TOTAL + "_anisotropy"
    ].values

    # Calculate anisotropies: Cell
    df = find_metadata_in_layers(
        results_stress_analysis, types._METADATAKEY_STRESS_CELL
    )
    result = measurements.calculate_anisotropy(
        df, types._METADATAKEY_STRESS_CELL, alpha=0.05, group_column="frame"
    )
    df_over_time[types._METADATAKEY_STRESS_CELL_ANISO] = result[
        types._METADATAKEY_STRESS_CELL + "_anisotropy"
    ].values

    return df_over_time, df_nearest_pairs, df_all_pairs, df_autocorrelations


def find_metadata_in_layers(layers: list, name: str) -> "napari.layers.Layer":
    """
    Find the layer in the viewer that contains specific metadata or feature.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer
    name : str
        Name of the metadata or feature to find

    Returns
    -------
    layer : napari.types.LayerDataTuple
        Layer containing the metadata or feature
    """
    import pandas as pd

    for layer in layers:
        if "metadata" in layer[1].keys():
            if name in layer[1]["metadata"].keys():
                return pd.DataFrame(layer[1]["metadata"])
        if "features" in layer[1].keys():
            if name in pd.DataFrame(layer[1]["features"]).columns:
                return pd.DataFrame(layer[1]["features"])


def aggregate_singular_values(
    results_stress_analysis: List[LayerDataTuple], n_frames: int, time_step: float
) -> pd.DataFrame:
    """
    Aggregate singular values results into a single dataframe

    Parameters
    ----------
    results_stress_analysis : List[LayerDataTuple]
        List of tuples containing the results of the stress analysis
    n_frames : int
        Number of frames in the data
    time_step : float
        Time step between frames

    Returns
    -------
    df_singular_values : pd.DataFrame
        Dataframe containing the singular values results, e.g. results
        from the stress analysis that refer to a single value per frame.
        Columns:
            time : float
                Time of the frame
            etc.
    """
    from ..types import (
        _METADATAKEY_STRESS_TOTAL,
        _METADATAKEY_STRESS_TISSUE,
        _METADATAKEY_AUTOCORR_TEMPORAL_TOTAL,
        _METADATAKEY_AUTOCORR_TEMPORAL_CELL,
        _METADATAKEY_AUTOCORR_TEMPORAL_TISSUE,
    )

    from .._measurements.temporal_correlation import temporal_autocorrelation

    def flatten_dictionary(input_dict, parent_key_prefix="", separator="_"):
        """Flatten a nested dictionary and convert singleton values to lists."""
        flat_dict = {}
        for key, value in input_dict.items():
            current_key = (
                f"{parent_key_prefix}{separator}{key}" if parent_key_prefix else key
            )

            if isinstance(value, dict):
                flat_dict.update(
                    flatten_dictionary(value, current_key, separator=separator)
                )
            else:
                # Convert scalar values to lists
                if not isinstance(value, list):
                    value = [value]
                flat_dict[current_key] = value

        return flat_dict

    # Single values over time
    _metadata = [
        layer[1]["metadata"]
        for layer in results_stress_analysis
        if "metadata" in layer[1].keys()
    ]
    _metadata = [flatten_dictionary(d) for d in _metadata]
    df_over_time = pd.concat([pd.DataFrame(x) for x in _metadata], axis=1)
    df_over_time = df_over_time.loc[:, ~df_over_time.columns.duplicated()].copy()

    # Find layer with stress_tissue in features
    for layer in results_stress_analysis:
        if "features" not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_TOTAL in layer[1]["features"].keys():
            df_total_stress = pd.DataFrame(layer[1]["features"])
            df_total_stress["time"] = layer[0][:, 0] * time_step

        if _METADATAKEY_STRESS_TISSUE in layer[1]["features"].keys():
            df_tissue_stress = pd.DataFrame(layer[1]["features"])
            df_tissue_stress["time"] = layer[0][:, 0] * time_step

    df_over_time["time"] = df_over_time["frame"] * time_step
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TOTAL] = temporal_autocorrelation(
        df_total_stress, "stress_total_radial", frame_column_name="frame"
    )
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_CELL] = temporal_autocorrelation(
        df_total_stress, "stress_cell", frame_column_name="frame"
    )
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TISSUE] = temporal_autocorrelation(
        df_tissue_stress, "stress_tissue", frame_column_name="frame"
    )

    return df_over_time


def aggregate_extrema_results(
    results_stress_analysis: List[LayerDataTuple], n_frames: int, time_step: float
) -> tuple:
    """
    Aggregate extrema results into a single dataframe

    Parameters
    ----------
    results_stress_analysis : List[LayerDataTuple]
        List of tuples containing the results of the stress analysis
    n_frames : int
        Number of frames in the data
    time_step : float
        Time step between frames

    Returns
    -------
    df_nearest_pair: pd.DataFrame
        Dataframe containing the nearest pair extrema results.
        Columns:
            time : float
                Time of the frame
            nearest_pair_distance : float
                Distance between the nearest pairs of extrema
            nearest_pair_anisotropy : float
                Cell Stress Anisotropy of the nearest pair
    df_all_pair: pd.DataFrame
        Dataframe containing the all pair extrema results.
        Columns:
            time : float
                Time of the frame
            all_pair_distance : float
                Distance between all pairs of extrema
            all_pair_anisotropy : float
                Cell Stress Anisotropy of all pairs
    """
    from ..types import (
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST,
    )

    # Find layer with NEAREST EXTREMA data
    for layer in results_stress_analysis:
        if "metadata" not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO in layer[1]["metadata"].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]["metadata"]
    if n_frames > 1:
        frames = (
            np.concatenate(
                [
                    [i] * len(metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO][i])
                    for i in range(n_frames)
                ]
            )
            * time_step
        )
        min_max_pair_distances = np.concatenate(
            metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST]
        )
        min_max_pair_anisotropies = np.concatenate(
            metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO]
        )
    else:
        frames = np.zeros(len(metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST]))
        min_max_pair_distances = metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST]
        min_max_pair_anisotropies = metadata[
            _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO
        ]

    df_nearest_pair = pd.DataFrame(
        {
            "frame": frames,
            "time": frames * time_step,
            _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST: min_max_pair_distances,
            _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO: min_max_pair_anisotropies,
        }
    )

    # Find layer with ALL PAIR EXTREMA data
    for layer in results_stress_analysis:
        if "metadata" not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO in layer[1]["metadata"].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]["metadata"]
    if n_frames > 1:
        frames = (
            np.concatenate(
                [
                    [i] * len(metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO][i])
                    for i in range(n_frames)
                ]
            )
            * time_step
        )
        all_pair_distances = np.concatenate(
            metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_DIST]
        )
        all_pair_anisotropies = np.concatenate(
            metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO]
        )
    else:
        frames = np.zeros(len(metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO]))
        all_pair_distances = metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_DIST]
        all_pair_anisotropies = metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO]

    df_all_pair = pd.DataFrame(
        {
            "frame": frames,
            "time": frames * time_step,
            _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST: all_pair_distances,
            _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO: all_pair_anisotropies,
        }
    )

    return df_nearest_pair, df_all_pair


def aggregate_spatial_autocorrelations_results(
    results_stress_analysis: List[LayerDataTuple], n_frames: int, time_step: float
) -> tuple:
    """
    Aggregate spatial autocorrelation results into a single dataframe

    Parameters
    ----------
    results_stress_analysis : List[LayerDataTuple]
        List of tuples containing the results of the stress analysis
    n_frames : int
        Number of frames in the data
    time_step : float
        Time step between frames

    Returns
    -------
    df_autocorrelations : pd.DataFrame
        Dataframe containing the spatial autocorrelation results.
        Columns:
            time : float
                Time of the frame
            distances : float
                Distances at which the autocorrelation was calculated
            autocorrelation_total : float
                Autocorrelation of the total stress
            autocorrelation_cell : float
                Autocorrelation of the cell stress
            autocorrelation_tissue : float
                Autocorrelation of the tissue stress
    """
    from ..types import (
        _METADATAKEY_AUTOCORR_SPATIAL_CELL,
        _METADATAKEY_AUTOCORR_SPATIAL_TISSUE,
        _METADATAKEY_AUTOCORR_SPATIAL_TOTAL,
    )

    # Find layer with SPATIAL AUTOCORRELATIONS
    for layer in results_stress_analysis:
        if "metadata" not in layer[1].keys():
            continue
        if _METADATAKEY_AUTOCORR_SPATIAL_CELL in layer[1]["metadata"].keys():
            break

    # TOTAL STRESS
    metadata = layer[1]["metadata"][_METADATAKEY_AUTOCORR_SPATIAL_TOTAL]
    if n_frames > 1:
        distances = [
            metadata[t]["auto_correlations_distances"] for t in range(n_frames)
        ]
        normalized_autocorrelation_total = [
            metadata[t]["auto_correlations_averaged_normalized"]
            for t in range(n_frames)
        ]
        frames = [
            [t] * len(metadata[t]["auto_correlations_averaged_normalized"])
            for t in range(n_frames)
        ]
    else:
        distances = [metadata["auto_correlations_distances"]]
        normalized_autocorrelation_total = [
            metadata["auto_correlations_averaged_normalized"]
        ]
        frames = [[0] * len(metadata["auto_correlations_averaged_normalized"])]

    df_autocorrelations_total = pd.DataFrame(
        {
            "time": np.concatenate(frames).squeeze() * time_step,
            "distances": np.concatenate(distances).squeeze(),
            _METADATAKEY_AUTOCORR_SPATIAL_TOTAL: np.concatenate(
                normalized_autocorrelation_total
            ).squeeze(),
        }
    )

    # CELL STRESS
    metadata = layer[1]["metadata"][_METADATAKEY_AUTOCORR_SPATIAL_CELL]
    if n_frames > 1:
        distances = [
            metadata[t]["auto_correlations_distances"] for t in range(n_frames)
        ]
        normalized_autocorrelation_cell = [
            metadata[t]["auto_correlations_averaged_normalized"]
            for t in range(n_frames)
        ]
        frames = [
            [t] * len(metadata[t]["auto_correlations_averaged_normalized"])
            for t in range(n_frames)
        ]
    else:
        distances = [metadata["auto_correlations_distances"]]
        normalized_autocorrelation_cell = [
            metadata["auto_correlations_averaged_normalized"]
        ]
        frames = [[0] * len(metadata["auto_correlations_averaged_normalized"])]

    df_autocorrelations_cell = pd.DataFrame(
        {
            "time": np.concatenate(frames).squeeze() * time_step,
            "distances": np.concatenate(distances).squeeze(),
            _METADATAKEY_AUTOCORR_SPATIAL_CELL: np.concatenate(
                normalized_autocorrelation_cell
            ).squeeze(),
        }
    )

    # TISSUE STRESS
    metadata = layer[1]["metadata"][_METADATAKEY_AUTOCORR_SPATIAL_TISSUE]
    if n_frames > 1:
        distances = [
            metadata[t]["auto_correlations_distances"] for t in range(n_frames)
        ]
        normalized_autocorrelation_tissue = [
            metadata[t]["auto_correlations_averaged_normalized"]
            for t in range(n_frames)
        ]
        frames = [
            [t] * len(metadata[t]["auto_correlations_averaged_normalized"])
            for t in range(n_frames)
        ]
    else:
        distances = [metadata["auto_correlations_distances"]]
        normalized_autocorrelation_tissue = [
            metadata["auto_correlations_averaged_normalized"]
        ]
        frames = [[0] * len(metadata["auto_correlations_averaged_normalized"])]

    df_autocorrelations_tissue = pd.DataFrame(
        {
            "time": np.concatenate(frames).squeeze() * time_step,
            "distances": np.concatenate(distances).squeeze(),
            _METADATAKEY_AUTOCORR_SPATIAL_TISSUE: np.concatenate(
                normalized_autocorrelation_tissue
            ).squeeze(),
        }
    )

    df_autocorrelations = pd.merge(
        df_autocorrelations_total,
        df_autocorrelations_tissue,
        "left",
        on=["time", "distances"],
    )
    df_autocorrelations = pd.merge(
        df_autocorrelations, df_autocorrelations_cell, "left", on=["time", "distances"]
    )

    return df_autocorrelations
