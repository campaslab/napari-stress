import pandas as pd
import numpy as np
from typing import List
from napari.types import LayerDataTuple


def aggregate_singular_values(results_stress_analysis: List[LayerDataTuple],
                              n_frames: int,
                              time_step: float) -> pd.DataFrame:
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
        _METADATAKEY_STRESS_CELL,
        _METADATAKEY_AUTOCORR_TEMPORAL_TOTAL,
        _METADATAKEY_AUTOCORR_TEMPORAL_CELL,
        _METADATAKEY_AUTOCORR_TEMPORAL_TISSUE
    )

    from .._measurements.temporal_correlation import temporal_autocorrelation

    # Single values over time
    _metadata = [layer[1]['metadata'] for layer in results_stress_analysis if 'metadata' in layer[1].keys()]
    df_over_time = {}
    df_over_time['frame'] = np.arange(n_frames)
    for meta in _metadata:
        for key in meta.keys():
            if type(meta[key][0]) == dict:
                for key2 in meta[key][0].keys():
                    v = [dic[key2] for dic in meta[key]]
                    df_over_time[key2] = v
            else:
                df_over_time[key] = meta[key]
    df_over_time = pd.DataFrame(df_over_time)
    df_over_time['time'] = df_over_time['frame'].values * time_step

    # Find layer with stress_tissue in features
    for layer in results_stress_analysis:
        if 'features' not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_TOTAL in layer[1]['features'].keys():
            df_total_stress = pd.DataFrame(layer[1]['features'])
            df_total_stress['time'] = layer[0][:, 0] * time_step

        if _METADATAKEY_STRESS_TISSUE in layer[1]['features'].keys():
            df_tissue_stress = pd.DataFrame(layer[1]['features'])
            df_tissue_stress['time'] = layer[0][:, 0] * time_step

    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TOTAL] = temporal_autocorrelation(
        df_total_stress, 'stress_total_radial', frame_column_name='time')
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_CELL] = temporal_autocorrelation(
        df_total_stress, 'stress_cell', frame_column_name='time')
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TISSUE] = temporal_autocorrelation(
        df_tissue_stress, 'stress_tissue', frame_column_name='time')

    return df_over_time


def aggregate_extrema_results(results_stress_analysis: List[LayerDataTuple],
                              n_frames: int,
                              time_step: float) -> tuple:
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
        _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST
    )

    # Find layer with NEAREST EXTREMA data
    for layer in results_stress_analysis:
        if 'metadata' not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO in layer[1]['metadata'].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]['metadata']
    frames = np.concatenate(
        [[i] * len(metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO][i]
                   ) for i in range(n_frames)]
        ) * time_step
    min_max_pair_distances = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST])
    min_max_pair_anisotropies = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO])

    df_nearest_pair = pd.DataFrame(
        {'frame': frames,
         'time': frames * time_step,
         'nearest_pair_distance': min_max_pair_distances,
         'nearest_pair_anisotropy': min_max_pair_anisotropies})

    # Find layer with ALL PAIR EXTREMA data
    for layer in results_stress_analysis:
        if not 'metadata' in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO in layer[1]['metadata'].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]['metadata']
    frames = np.concatenate(
        [[i] * len(metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO][i]
                     ) for i in range(n_frames)]
        ) * time_step
    all_pair_distances = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_DIST])
    all_pair_anisotropies = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO])

    df_all_pair = pd.DataFrame(
        {'frame': frames,
         'time': frames * time_step,
         'all_pair_distance': all_pair_distances,
         'all_pair_anisotropy': all_pair_anisotropies})

    return df_nearest_pair, df_all_pair


def aggregate_spatial_autocorrelations_results(results_stress_analysis: List[LayerDataTuple],
                                               n_frames: int,
                                               time_step: float) -> tuple:
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
        _METADATAKEY_AUTOCORR_SPATIAL_TOTAL
    )

    # Find layer with SPATIAL AUTOCORRELATIONS
    for layer in results_stress_analysis:
        if 'metadata' not in layer[1].keys():
            continue
        if _METADATAKEY_AUTOCORR_SPATIAL_CELL in layer[1]['metadata'].keys():
            break

    # TOTAL STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_TOTAL]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_total = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_total = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_total': np.concatenate(normalized_autocorrelation_total).squeeze()
         })

    # CELL STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_CELL]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_cell = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_cell = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_cell': np.concatenate(normalized_autocorrelation_cell).squeeze()
         })

    # TISSUE STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_TISSUE]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_tissue = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_tissue = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_tissue': np.concatenate(normalized_autocorrelation_tissue).squeeze()
         })

    df_autocorrelations = pd.merge(
        df_autocorrelations_total,
        df_autocorrelations_tissue,
        'left',
        on=['time', 'distances']
        )
    df_autocorrelations = pd.merge(
        df_autocorrelations,
        df_autocorrelations_cell, 
        'left',
        on=['time', 'distances']
        )

    return df_autocorrelations
