# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:13:13 2021

@author: johan
"""

import os
from stress.stress import analysis
from stress.functions.visualize import visualize

if __name__ == "__main__":
    root = r'D:\Documents\Promotion\Projects\2021_STRESS_translation'
    f_image = os.path.join(root, 'data', 'ExampleTifSequence-InteriorLabel-vsx_2.076um-vsz_3.998um-TimeInterval_3.00min-21timesteps.tif')    
    
    Job = analysis()
    Job.load(f_image)
     
    # Point fitting on surface       
    Job.preprocessing(vsx=2.076, vsy=2.076, vsz=3.998)
    Job.fit_ellipse()
    Job.resample_surface()
    
    # Curvature measurement
    Job.fit_curvature()
    
    visualize(Job)