# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:09:51 2021

@author: johan
"""

import napari
import numpy as np

def visualize(STRESS, viewer=None, colorcode='Curvature'):
    """
    Create napari visualization for stress surface points
    """
    
    viewer = napari.Viewer()
    
    if STRESS.has_image:
        image_layer = viewer.add_image(STRESS.image, colormap='gray',
                                       name='Image')
        
    if STRESS.has_mask:
        label_layer = viewer.add_labels(STRESS.mask, name='Binary mask')
        
    if STRESS.has_points:
        points = STRESS.points
        point_properties = {
                'Curvature': np.vstack(points['Curvature']).squeeze(),
                'FitErrors': np.vstack(points['FitErrors']).squeeze()
            }   

        # Check if data exists        
        if colorcode == 'Curvature' and STRESS.has_curv:
            face_color = 'Curvature'
            name = 'Sample points (curvature)'

        elif colorcode == 'FitError' and STRESS.has_error:
            face_color = 'FitError'
            name = 'Sample points (Fit Error)'

        # Add points
        points_layer = viewer.add_points(
            np.vstack(points.XYZ),
            properties=point_properties,
            face_color=face_color,
            face_colormap='viridis',
            edge_width=0.05,
            size=0.35,
            name=name
        )
        
        
    if STRESS.has_normals:
        data = np.zeros((len(points.Normals), 2, 3))
        data[:, 0] = np.vstack(points.XYZ)
        data[:, 1] =  np.vstack(points.Normals)
        vector_layer = viewer.add_vectors(
                data, length=1, edge_width=0.15,
                name='surface normals',
                edge_color='orange')
        
        