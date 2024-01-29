# %% [markdown]
# (toolboxes:analyze_everything)=
# # Analyze everything
#
# This notebook demonstrates how to run a complete STRESS analysis and produce all relevant output graphs. If you want to download this notebook and execute it locally on your machine, download this file as a `ipynb` Jupyter notebook file and run it in your local python environment using the download button at the top of this page.

# %%
import napari_stress
import napari
import numpy as np
from napari_stress import (
    reconstruction,
    measurements,
    TimelapseConverter,
    utils,
    stress_backend,
    plotting,
)
import os
import datetime

from skimage import io

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import yaml

# %%
reconstruction_parameters = None
measurement_parameters = None

# %% [markdown]
# ## Load the data
#
# Replace the following code with the commented out part (and remove the rest) below to load your own data for analysis:

# %%
image = napari_stress.get_droplet_4d()[0][0][:1]
image.shape
filename = None

## Replace this code with a command to import your data. Example:
# filename = 'path/to/data.tif'
# image = io.imread(filename)

# %% [markdown]
# ### Data dimensions
# You need to set a few parameters pertaining to your data:

# %%
voxel_size_x = 2.078  # microns
voxel_size_y = 2.078  # microns
voxel_size_z = 3.998  # microns
target_voxel_size = 2.078  # microns
time_step = 3  # minutes

# %% [markdown]
# ### Analysis parameters
#
# In case you ran the reconstruction previously interactively from the napari viewer (as explained [here](toolboxes:droplet_reconstruction:interactive)) and exported the settings, you can import the settings here, too. To do so, simply uncomment the line below (remove the `#`) and provide the path to the saved settings file:

# %%
# reconstruction_parameters = utils.import_settings(file_name='path/of/reconstruction/settings.yaml')
# measurement_parameters = utils.import_settings(file_name='path/of/measurement/settings.yaml')

# %% [markdown]
# If you used a parameter file, you can skip the next step. Otherwise, use this cell to provide the necessary parameters for the reconstruction and the measurement. The parameters are explained here:
# - [Reconstruction](toolboxes:droplet_reconstruction:interactive)
# - [Measurement](toolboxes:stress_toolbox:stress_toolbox_interactive)
#
# If you used the previous cell to import some parameters, skip the next cell or delete it.

# %%
reconstruction_parameters = {
    "voxelsize": np.asarray([voxel_size_z, voxel_size_y, voxel_size_x]),
    "target_voxelsize": target_voxel_size,
    "smoothing_sigma": 1,
    "n_smoothing_iterations": 15,
    "n_points": 256,
    "n_tracing_iterations": 2,
    "resampling_length": 1,
    "fit_type": "fancy",  # can be 'fancy' or 'quick'
    "edge_type": "interior",  # can be 'interior' or 'surface'
    "trace_length": 20,
    "sampling_distance": 1,
    "interpolation_method": "linear",  # can be 'linear' 'cubic' or 'nearest'
    "outlier_tolerance": 1.5,
    "remove_outliers": True,
    "return_intermediate_results": True,
}

measurement_parameters = {
    "max_degree": 20,  # spherical harmonics degree
    "n_quadrature_points": 590,  # number of quadrature points to measure on (maximum is 5180)
    "gamma": 3.3,
}  # interfacial tension of droplet
alpha = 0.05  # lower and upper boundary in cumulative distribution function which should be used to calculate the stress anisotropy

# %% [markdown]
# *Hint:* If you are working with timelapse data, it is recommended to use parallel computation to speed up the analysis.

# %%
parallelize = False

# %% [markdown]
# # Analysis

# %%
viewer = napari.Viewer(ndisplay=3)

# %%
viewer.add_image(image)

# %%
n_frames = image.shape[0]

# %% [markdown]
# We run the reconstruction and the stress analysis:

# %%
results_reconstruction = reconstruction.reconstruct_droplet(
    image, **reconstruction_parameters, use_dask=parallelize
)

for res in results_reconstruction:
    layer = napari.layers.Layer.create(res[0], res[1], res[2])
    viewer.add_layer(layer)

# %%
_ = stress_backend.lbdv_info(
    Max_SPH_Deg=measurement_parameters["max_degree"],
    Num_Quad_Pts=measurement_parameters["n_quadrature_points"],
)

input_data = viewer.layers["points_patch_fitted"].data
results_stress_analysis = measurements.comprehensive_analysis(
    results_reconstruction[2][0], **measurement_parameters, use_dask=parallelize
)

for res in results_stress_analysis:
    layer = napari.layers.Layer.create(res[0], res[1], res[2])
    viewer.add_layer(layer)

# %% [markdown]
# To get an idea about the returned outputs and which is stored in which layer, let's print them:

# %%
for res in results_stress_analysis:
    print("-->", res[1]["name"])
    if "metadata" in res[1].keys():
        for key in res[1]["metadata"].keys():
            print("\t Metadata: ", key)
    if "features" in res[1].keys():
        for key in res[1]["features"].keys():
            print("\t Features: ", key)

# %% [markdown]
# To make handling further down easier, we store all data and metadata in a few simple dataframes

# %%
# Compile data
(
    df_over_time,
    df_nearest_pairs,
    df_all_pairs,
    df_autocorrelations,
) = utils.compile_data_from_layers(
    results_stress_analysis, n_frames=n_frames, time_step=time_step
)

# %% [markdown]
# # Visualization
#
# In this section, we will plot some interesting results and save the data to disk. The file location will be at the

# %%
# %%capture
figures_dict = plotting.create_all_stress_plots(
    results_stress_analysis, time_step=time_step, n_frames=n_frames
)

# %%
mpl.style.use("default")
colormap_time = "flare"
if filename is not None:
    filename_without_ending = os.path.basename(filename).split(".")[0]
    save_directory = os.path.join(
        os.path.dirname(filename),
        filename_without_ending
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
else:
    save_directory = os.path.join(
        os.getcwd(), "results_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# %% [markdown]
# ## Fit errors
#
# We first show all the errors that were calculated during the pointcloud refinement:

# %%
fit_error_df = pd.DataFrame(results_reconstruction[3][1]["features"].reset_index())
fit_error_df

# %%
fig, axes = plt.subplots(
    ncols=4, nrows=len(fit_error_df.columns) // 4 + 1, figsize=(20, 10), sharey=True
)
axes = axes.flatten()
for idx, column in enumerate(fit_error_df.columns):
    ax = axes[idx]

    sns.histplot(data=fit_error_df, x=column, ax=ax, bins=100)
    ax.set_xlabel(column, fontsize=16)
    ax.set_ylabel("Counts [#]", fontsize=16)

if save_directory is not None:
    fig.savefig(os.path.join(save_directory, "fit_error_reconstruction.png"), dpi=300)

# %% [markdown]
# ## Spherical harmonics
#
# ### Fit residue
#
# We now show the errors made when approximating the reconstructed pointcloud with the spherical harmonics:

# %%
figure = figures_dict["Figure_reside"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ### Fit quality
#
# We can quantify the quality of the extracted pointcloud by using the absolute and relative Gauss-Bonnet errors:

# %%
figure = figures_dict["fig_GaussBonnet_error"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Curvature
#
# We next show mean curvature histograms and averages over time:

# %%
figure = figures_dict["fig_mean_curvature"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"].axes[0].set_xlabel("Mean curvature [$mm^{-1}$]", fontsize=16)

figure["figure"].axes[1].set_ylabel("Mean curvature [$mm^{-1}$]", fontsize=16)
figure["figure"].axes[1].set_xlabel("Time [min]", fontsize=16)

figure["figure"]

# %%
figure = figures_dict["fig_total_stress"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %%
figure = figures_dict["fig_cell_stress"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Tissue-scale stresses

# %%
figure = figures_dict["fig_tissue_stress"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Stress along axes

# %%
figure = figures_dict["fig_stress_tensor"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Extrema analysis

# %%
figure = figures_dict["fig_all_pairs"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ##  Autocorrelations: Spatial

# %%
figure = figures_dict["fig_spatial_autocorrelation"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Autocorrelations: Temporal

# %%
figure = figures_dict["fig_temporal_autocorrelation"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Ellipsoid deviation

# %%
figure = figures_dict["fig_ellipsoid_contribution"]
if save_directory is not None:
    figure["figure"].savefig(os.path.join(save_directory, figure["path"]), dpi=300)

figure["figure"]

# %% [markdown]
# ## Droplet movement
#
# This analyzes how much the center of the droplet moves over time.

# %%
Converter = TimelapseConverter()
list_of_points = Converter.data_to_list_of_data(
    results_reconstruction[3][0], layertype=napari.types.PointsData
)
center = [np.mean(points, axis=0) for points in list_of_points]
center_displacement = np.asarray(
    [np.linalg.norm(center[t] - center[0]) for t in range(n_frames)]
)
df_over_time["droplet_center_displacement"] = center_displacement * target_voxel_size

sns.lineplot(data=df_over_time, x="time", y="droplet_center_displacement", marker="o")

# %% [markdown]
# ## Export data
#
# We first agregate the data from the spatial autocorrelations in a separate dataframe. This dataframe has a column for autocorrelations of total, cell and tissue-scale stresses.

# %%
df_to_export = pd.DataFrame()
for col in df_over_time.columns:
    if isinstance(df_over_time[col].iloc[0], np.ndarray):
        continue
    if np.stack(df_over_time[col].to_numpy()).shape == (n_frames,):
        df_to_export[col] = df_over_time[col].to_numpy()

df_to_export.to_csv(os.path.join(save_directory, "results_over_time.csv"), index=False)

# %% [markdown]
# We also export the used settings for the analysis  into a `.yml` file:

# %%
utils.export_settings(
    reconstruction_parameters,
    file_name=os.path.join(save_directory, "reconstruction_settings.yaml"),
)
utils.export_settings(
    measurement_parameters,
    file_name=os.path.join(save_directory, "measurement_settings.yaml"),
)

# %%
