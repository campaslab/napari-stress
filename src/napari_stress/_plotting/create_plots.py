import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def draw_chronological_kde_plot(df: pd.DataFrame,
                                x: str,
                                hue: str = 'time',
                                ax: plt.Axes = None,
                                colormap: str = 'flare',
                                grid: bool = True) -> Tuple[plt.Figure, plt.Axes]:
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
    sns.kdeplot(data=df, x=x, hue=hue, ax=ax, palette=colormap)

    if grid:
        ax.grid(which='both', linestyle='--', alpha=0.5, color='grey')
    else:
        plt.grid(False)

    return ax.figure, ax


def draw_chronological_lineplot_with_errors(
        df: pd.DataFrame,
        y: str,
        hue: str = None,
        x: str = 'time',
        ax: plt.Axes = None,
        grid: bool = True,
        error: str = 'se',
        estimator=np.mean,
        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
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

    sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, estimator=estimator,
                 errorbar=error, **kwargs)

    if grid:
        ax.grid(which='both', linestyle='--', alpha=0.5, color='grey')
    else:
        plt.grid(False)

    return ax.figure, ax