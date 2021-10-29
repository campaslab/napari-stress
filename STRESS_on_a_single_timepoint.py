#!/usr/bin/env python
# coding: utf-8

# # Stress implementation

# In[1]:


import os
import tifffile as tf

import matplotlib.pyplot as plt
import napari

import pandas as pd
import numpy as np

from stress import stress, utils, tracing, surface

get_ipython().run_line_magic('matplotlib', 'notebook')


# Load data

# ### Config

# In[2]:


fluorescence = 'interior'  # type of fluorescence
n_samples = 2**8  # number of points on initial estimate
trace_fit_method = 'quick_edge'
patch_radius = 2  # radius within which a point is counted as neighbour of another point
n_refinements = 2  # number of refinement steps for surface resampling
surface_sampling_density = 1  # designated density of points on the drop surface


# ### Processing

# In[3]:


root = r'D:\Documents\Promotion\Projects\2021_STRESS_translation'
f_image = os.path.join(root, 'data', 'ExampleTifSequence-InteriorLabel-vsx_2.076um-vsz_3.998um-TimeInterval_3.00min-21timesteps.tif')    

series = tf.imread(f_image)


# Pick one timepoint and visualize

# In[4]:


image = series[0]

fig, axes = plt.subplots(ncols=3)
axes[0].imshow(image[image.shape[0]//2])
axes[1].imshow(image[:, image.shape[1]//2, :])
axes[2].imshow(image[:, :, image.shape[2]//2])


# ### Preprocessing
# Do some preprocessing:
# - Resampling

# In[5]:


image = stress.preprocessing(image, vsx=2.076, vsy=2.076, vsz=3.998)

fig, axes = plt.subplots(ncols=3)
axes[0].imshow(image[15])
axes[1].imshow(image[:, 15, :])
axes[2].imshow(image[:, :, 15])


# ###  Initial ellipse fit
# To get an initial estimate on the shape and surface of the oil drop, fit an ellipse to the dropplet:

# In[6]:


_points = stress.fit_ellipse(image, fluorescence=fluorescence, n_samples=n_samples)


# To keep track of the data, create a pandas dataframe for our data and store the detected points in the dataframe:

# In[7]:


_points = np.asarray(_points).transpose()  # convert to 3xN array
points = pd.DataFrame(columns=['Z', 'Y', 'X'])
points['Z'] = _points[0, :]
points['Y'] = _points[1]
points['X'] = _points[2]
center = np.asarray([points.Z.mean(), points.Y.mean(), points.X.mean()])


# Visualize in napari

# In[8]:


# viewer = napari.Viewer(ndisplay=3)
# viewer.add_image(image, colormap='gray')
# ellipse_layer = viewer.add_points(utils.df2ZYX(points), size=1, face_color ='orange', edge_width=0.1, opacity=0.7, name='Ellipse fit')
# viewer.add_points(center, size=1, face_color ='red', edge_width=0.1, opacity=1.0, name='Center')
# napari.utils.nbscreenshot(viewer, canvas_only=False)


# ### Ray-tracing
# Shoot rays from center towards all currently known surface points. Image intensity is measured along the way.
# If fluorescence = 'interior', the surface is considered to be the location where intensity drops to 0.5 x maxIntensity.
# If fluorescence = 'edge', the surface is considered to be the location of maximal intesity along the ray.

# In[9]:


new_points, errors, fitparams = tracing.get_traces(image, start_pts=center, target_pts=utils.df2ZYX(points),
                                                    detection=trace_fit_method, fluorescence=fluorescence)
points['Z'] = new_points[:, 0]
points['Y'] = new_points[:, 1]
points['X'] = new_points[:, 2]
points['errors'] = errors
points['fitparams'] = fitparams


# In[10]:


# points_layer_1 = viewer.add_points(utils.df2ZYX(points), size=0.3, face_color ='blue', edge_width=0.1, opacity=0.7, name='Raytracing - initial')
# ellipse_layer.opacity=0.2
# napari.utils.nbscreenshot(viewer, canvas_only=False)


# ### Surface filtering
# 
# Next, we want to reject "bad" points from the list of identified points on the surface. For this, we first identify the neighbours of each point and add the index and number of neighbours to the dataframe. The ```neighbours``` attribute contains the row index of the neighbouring point and ```n_neighbours``` contains the number of each point's neighbours for convenience.

# In[11]:


neighbours, n_neighbours = surface.get_neighbours(utils.df2ZYX(points), patch_radius=patch_radius)
points['neighbours'] = neighbours
points['n_neighbours'] = n_neighbours
points


# We now use the ```filter_dataframe()``` function which calculates quantiles and interquartile distances for every point property and rejects points classified as outliers from the dataframe. In this case, we want to remove points with hardly any neighbours (e.g. that are somehow far away from the surface) or an excessive number of neighbours (for instance due to local edges in the surface)

# In[12]:


points = utils.filter_dataframe(points, columns=['n_neighbours'], criteria=['within'], inplace=True, verbose=True)


# ### Surface resampling
# It is necessary to resample the points on the surface as the initial fit procedure may yield inaccurate results. For demonstration: 

# In[13]:


_points = surface.resample_points(utils.df2ZYX(points), surface_sampling_density=surface_sampling_density)
# points_layer_1.opacity = 0.2
# points_layer_2 = viewer.add_points(_points, size=0.3, face_color ='cyan', edge_width=0.1, opacity=0.7)


# This procedure is now repeated for a number of ```n_refinements``` (default = 2). During each iteration, the following steps are executed:
# - Surface is resampled to predefined density
# - Local normals are calculated
# - Raytracing is repeated along the calculated normal
# - Coordinates are filtered

# In[ ]:


#for i in range(n_refinements):
#print(f'Iteration #{i+1}:')

# resample points on surface to desired density
_points = surface.resample_points(utils.df2ZYX(points), surface_sampling_density=surface_sampling_density)


# In[ ]:


# Calculate local normals
normals = surface.get_local_normals(_points)


# In[ ]:


# Calculate starting points for advanced tracing and run tracing
start_pts = _points - 2*normals
_points, errors, fitparams = tracing.get_traces(image, start_pts=start_pts, target_pts=_points,
                                                detection=trace_fit_method, fluorescence=fluorescence)


# In[ ]:


# Find neighbours
neighbours, n_neighbours = surface.get_neighbours(_points, patch_radius=patch_radius)


# In[ ]:


# Overwrite dataframe
points = pd.DataFrame(columns=points.columns)
points['Z'] = _points[:, 0]
points['Y'] = _points[:, 1]
points['X'] = _points[:, 2]
points['errors'] = errors
points['fitparams'] = fitparams
points['neighbours'] = neighbours
points['n_neighbours'] = n_neighbours


# In[ ]:


# Repeat point filtering as before
points = utils.filter_dataframe(points, columns=['n_neighbours'], criteria=['within'], inplace=True, verbose=True)


# In[ ]:




