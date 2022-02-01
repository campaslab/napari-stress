# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:18:26 2021

@author: johan
"""

import tifffile as tf

from stress import surface, tracing, curvature
from stress.functions import visualize
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
    
    def preprocessing(self, vsx=2.076, vsy=2.076, vsz=3.998):
        """
        Filters the image with a gaussian kernel and then creates binary mask
        """
        
        # Filtering
        self.image = surface.resample(self.image, vsz, vsx, vsy)
        
        # Masking
        self.image = filters.gaussian(self.image, sigma=self.sigma_gaussian)
        self.mask = measure.label(self.image > 0.2*self.image.max())
        self.has_mask = True
        
    def fit_ellipse(self):
        """
        Fits an ellipse to binarized data
        """
        ## Approximate object with ellipsoid
        props = measure.regionprops(self.mask)
        assert len(props) == 1  # to do: escape the detection of multiple objects, should this occur
        
        self.CoM = np.asarray(ndimage.center_of_mass(self.mask * self.image))
        
        
        # Create coordinate grid:
        ZZ, YY, XX = np.meshgrid(np.arange(self.image.shape[1]),
                                 np.arange(self.image.shape[0]),
                                 np.arange(self.image.shape[2]))
        
        # Substract center of mass and mask
        ZZ = (ZZ.astype(float).flatten() - self.CoM[0]) * self.mask.flatten()
        YY = (YY.astype(float).flatten() - self.CoM[1]) * self.mask.flatten()
        XX = (XX.astype(float).flatten() - self.CoM[2]) * self.mask.flatten()
        
        # Concatenate to single (Nx3) coordinate vector
        XYZ = np.vstack([XX, YY, ZZ]).transpose((1,0))
        
        # Calculate orientation matrix
        S = 1/np.sum(self.mask) * np.dot(XYZ.conjugate().T, XYZ)
        D, RotMat = np.linalg.eig(S)
        
        # This part comes straight from the matlab script, no idea how this works.
        if self.fluorescence == 'interior':
            self.semiAxesLengths = np.sqrt(5.0 * D)
        elif self.fluorescence == 'surface':
            self.semiAxesLengths = np.sqrt(3.0 * D);
            
        # Now create points on the surface of an ellipse
        pts = surface.fibonacci_sphere(self.semiAxesLengths, RotMat, self.CoM)
        pts = [np.asarray([x[0], x[1], x[2]]) for x in pts]
        self.points['XYZ'] = pts
        
        self.has_points = True
        
    def resample_surface(self, n_refinements=2):
        """
        Resamples points on the dropplet surface to higher density
        """
        
        # Do tracing
        self.points = tracing.get_traces(self.image,
                                         self.points,
                                         start_pts=self.CoM,
                                         target_pts=self.points.XYZ,
                                         detection=self.trace_fit_method,
                                         fluorescence = self.fluorescence)
        
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
            
    def fit_curvature(self):
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

    def visualize(self):
        visualize(self)