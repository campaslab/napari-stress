# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:18:26 2021

@author: johan
"""

import tifffile as tf

from stress import surface, tracing, curvature
# from stress.functions import visualize
# import surface, tracing, curvature
from skimage import filters, measure
from scipy import ndimage

import pandas as pd
import numpy as np
import os
import tqdm

class analysis:
    def __init__(self):
        self.filename = ''
        
        # Data containers
        self.image = None
        self.mask = None
        self.points = pd.DataFrame(columns=['XYZ', 'FitErrors',
                                            'FitParams', 'Neighbours',
                                            'N_Neighbours', 'Normals'])
                
        # Processing configuration
        self.sigma_gaussian = 1.0  # smoothing factor for image masking
        self.surface_sampling_density = 1.0  # spacing between points on surface
        self.patch_radius = 2.0  # range within which a point will be counted as neighbour
        self.point_filter_scale = 1.5  # Interquartile distance will be multiplied with this factor to identify outliers
        
        # Input config
        self.fluorescence = 'interior'
        self.trace_fit_method = 'quick_edge'
        
        # Image flags for visualization
        self.has_image = False
        self.has_points = False
        self.has_normals = False
        self.has_curv = False
        self.has_error = False
        self.has_mask = False
        
    def get_x(self):
        XYZ = np.vstack(self.points.XYZ)
        return XYZ[:, 0]
    
    def get_y(self):
        XYZ = np.vstack(self.points.XYZ)
        return XYZ[:, 1]
    
    def get_z(self):
        XYZ = np.vstack(self.points.XYZ)
        return XYZ[:, 2]
    
    def get_center(self):
        x = self.get_x().mean()
        y = self.get_y().mean()
        z = self.get_z().mean()
        
        return np.asarray([x,y,z])
    
    def load(self, filename, timestep=0):
        """
        Loads a tifffile from filename
        """
        self.image = tf.imread(filename)[timestep]  # Ax order: TZYX, take only first timestep for now
        self.has_image = True
    
def preprocessing(image, vsx=2.076, vsy=2.076, vsz=3.998):
    """
    Preprocesses an input 3D image for further processing. Preprocessing includes
    resampling to isotropic voxels.

    Parameters
    ----------
    image : MxNxK array
        3D image array with intensity values in each pixel
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 2.076
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 2.076
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 3.998

    Returns
    -------
    image : MxNxK array
        preprocessed image

    """
    
    # Filtering
    output = surface.resample(image, vsz, vsx, vsy)
    
    return output
    
def threshold(image, threshold = 0.2, **kwargs):
    
    sigma = kwargs.get('sigma', 1)
    
    # Masking
    image = filters.gaussian(image, sigma=sigma)
    mask = measure.label(image > threshold*image.max())
    
    return mask
    

def fit_ellipse(image, **kwargs):
    """
    Fits an ellipse to binarized data
    """
    
    fluorescence = kwargs.get('fluorescence', 'interior')
    n_samples = kwargs.get('n_samples', 256)
    
    mask = threshold(image, **kwargs)
    
    ## Approximate object with ellipsoid
    props = measure.regionprops(mask)
    assert len(props) == 1  # to do: escape the detection of multiple objects, should this occur
    
    CoM = np.asarray(ndimage.center_of_mass(mask * image))
    
    
    # Create coordinate grid:
    ZZ, YY, XX = np.meshgrid(np.arange(image.shape[1]),
                             np.arange(image.shape[0]),
                             np.arange(image.shape[2]))
    
    # Substract center of mass and mask
    ZZ = (ZZ.astype(float).flatten() - CoM[0]) * mask.flatten()
    YY = (YY.astype(float).flatten() - CoM[1]) * mask.flatten()
    XX = (XX.astype(float).flatten() - CoM[2]) * mask.flatten()
    
    # Concatenate to single (Nx3) coordinate vector
    XYZ = np.vstack([XX, YY, ZZ]).transpose((1,0))
    
    # Calculate orientation matrix
    S = 1/np.sum(mask) * np.dot(XYZ.conjugate().T, XYZ)
    D, RotMat = np.linalg.eig(S)
    
    # This part comes straight from the matlab script, no idea how this works.
    if fluorescence == 'interior':
        semiAxesLengths = np.sqrt(5.0 * D)
    elif fluorescence == 'surface':
        semiAxesLengths = np.sqrt(3.0 * D);
        
    # Now create points on the surface of an ellipse
    pts = surface.fibonacci_sphere(semiAxesLengths, RotMat, CoM, samples=n_samples)
    pts = [np.asarray([x[0], x[1], x[2]]) for x in pts]
    
    return pts
    
def resample_surface(image, points, center, n_refinements=2, **kwargs):
    """
    Resamples points on the dropplet surface to higher density
    """
    
    trace_fit_method = kwargs.get('trace_fit_method', 'quick_edge')
    n_refinements = kwargs.get('n_refinements', 2)
    fluorescence = kwargs.get('fluorescence', 'interior')
    
    # Do tracing
    points = tracing.get_traces(image, start_pts=center, target_pts=points,
                                detection=trace_fit_method, fluorescence=fluorescence)
    
    # Clean up points based on neighborhood, etc
    self.points = surface.clean_coordinates(self)
    print('\n---- Refinement-----')
    
    for i in range(n_refinements):
        print(f'Iteration #{i+1}:')
        
        self.points = surface.resample_surface(self)
        
        # Calculate new center of mass
        XYZ = np.vstack(self.points.XYZ)
        self.CoM = np.asarray([XYZ[:,0].mean(), XYZ[:,1].mean(), XYZ[:,2].mean()])
        
        self.points = surface.get_local_normals(self.points)
        
        # Calculate starting points for advanced tracing
        start_pts = self.points.XYZ - 2*self.points.Normals
        
        self.points = tracing.get_traces(self.image,
                                         self.points,
                                         start_pts=start_pts,
                                         target_pts=self.points.XYZ,
                                         detection=self.trace_fit_method,
                                         fluorescence=self.fluorescence)
        
        self.points = surface.clean_coordinates(self)
    
    # Raise flags for provided data
    self.has_normals = True
        
def fit_curvature():
    """
    Find curvature for every point
    """
    
    print('\n---- Curvature-----')
    curv = []
    for idx, point in tqdm.tqdm(self.points.iterrows(), desc='Measuring mean curvature', total=len(self.points)):
        sXYZ, sXq = surface.get_patch(self.points, idx, self.CoM)
        curv.append(curvature.surf_fit(sXYZ, sXq))
        
    self.points['Curvature'] = curv
    self.points = surface.clean_coordinates(self)
    
    # Raise flags for provided data
    self.has_curv = True