# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:31:29 2021

@author: johan
"""

import numpy as np
import vedo

def pointcloud_to_vertices4D(surfs: list) -> np.ndarray:
    
    n_vertices = sum([surf.N() for surf in surfs])
    vertices_4d = np.zeros([n_vertices, 4])
    
    for idx, surf in enumerate(surfs):
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 1:] = surf.points()
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 0] = idx
        
    return vertices_4d

def vertices4d_to_pointcloud(vertices: np.ndarray) -> list:
    
    assert vertices.shape[1] == 4
    
    frames = np.unique(vertices[:, 0])
    
    surfs =  []
    for idx in frames:
        frame = vertices(np.where(vertices[:, 0] == idx))
        surfs.append(vedo.pointcloud.Points(frame))
        
    return surfs
        
