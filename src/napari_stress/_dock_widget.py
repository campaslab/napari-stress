"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from magicgui import magic_factory

import napari

import numpy as np
import tifffile as tf

from napari_stress._surface import reconstruct_surface, calculate_curvatures, surface2layerdata
from napari_stress._preprocess import preprocessing, fit_ellipse
from napari_stress._tracing import get_traces

@magic_factory
def stress_widget(viewer: napari.Viewer,
                  img_layer: "napari.layers.Image",
                  vsx: float = 2.076,
                  vsy: float = 2.076,
                  vsz: float = 3.99,
                  vt:float = 3,
                  N_points: np.uint16 = 256,
                  curvature_radius: float = 2):

    image = img_layer.data
    
    if image.shape ==3:
        img_layer.data = img_layer.data[None, :]  # add empty time dimension if it doesn't exist

    # Preprocessing
    img_layer.scale = [vt, vsz, vsy, vsx]
    image_resampled, mask_resampled = preprocessing(img_layer.data,
                                                    vsx=vsx, vsy=vsy, vsz=vsz)
    # scale = [vt] + [np.min([vsx, vsy, vsz])] * 3
    n_frames = img_layer.data.shape[0]

    image_resampled = viewer.add_image(image_resampled, name='Resampled image')
    mask_resampled = viewer.add_labels(mask_resampled, name='Resampled mask')

    # Fit ellipse
    pts = np.zeros([N_points * n_frames, 4])
    for t in range(image_resampled.data.shape[0]):
        _pts = fit_ellipse(binary_image=mask_resampled.data[t])
        pts[t * N_points : (t + 1) * N_points, 1:] = _pts
        pts[t * N_points : (t + 1) * N_points, 0] = t

    # Do first tracing
    # Calculate the center of mass of determined points for every frame
    pts_surf = np.zeros([N_points * n_frames, 4])
    for t in range(n_frames):
        _pts = pts[t * N_points : (t + 1) * N_points, :]
        CoM = _pts.mean(axis=0)

        _pts_surf, _err, _FitP = get_traces(image = image_resampled.data[t],
                                            start_pts=CoM[1:], target_pts=_pts[:, 1:])

        pts_surf[t * N_points : (t + 1) * N_points, 1:] = _pts_surf
        pts_surf[t * N_points : (t + 1) * N_points, 0] = t

    # Reconstruct the surface and calculate curvatures
    surfs = reconstruct_surface(pts_surf, dims=image_resampled.data[0].shape)
    surfs = calculate_curvatures(surfs, radius=curvature_radius)

    # Add to viewer
    surf_data = surface2layerdata(surfs)
    viewer.add_surface(surf_data,
                       colormap='viridis',
                       contrast_limits=(np.quantile(surf_data[2], 0.2),
                                        np.quantile(surf_data[2], 0.8))
                       )

    

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [stress_widget]


# if __name__ == "__main__":

#     filename = r'D:\Documents\Biapol\Shared\BiAPoLprojects\STRESS\1_first data\ExampleTifSequence-InteriorLabel-vsx_2.076um-vsz_3.998um-TimeInterval_3.00min-21timesteps.tif'
#     image = tf.imread(filename)

#     if len(image.shape) == 3:
#         image = image[None, :]

#     viewer = napari.Viewer()
#     viewer.add_image(image)

#     vsx: float = 2.076
#     vsy: float = 2.076
#     vsz: float = 3.99
#     vt: float = 3
#     N_points: np.uint16 = 256
    
#     curv_method = 'Gauss_Curvature'

#     img_layer = viewer.layers[0]

#     # Preprocessing
#     img_layer.scale = [vt, vsz, vsy, vsx]
#     image_resampled, mask_resampled = preprocessing(img_layer.data,
#                                                     vsx=vsx, vsy=vsy, vsz=vsz)
#     scale = [vt] + [np.min([vsx, vsy, vsz])] * 3
#     n_frames = img_layer.data.shape[0]

#     image_resampled = viewer.add_image(image_resampled, name='Resampled image')
#     mask_resampled = viewer.add_labels(mask_resampled, name='Resampled mask')

#     # Fit ellipse
#     pts = np.zeros([N_points * n_frames, 4])
#     for t in range(image_resampled.data.shape[0]):
#         _pts = fit_ellipse(binary_image=mask_resampled.data[t])
#         pts[t * N_points : (t + 1) * N_points, 1:] = _pts
#         pts[t * N_points : (t + 1) * N_points, 0] = t

#     # Do first tracing
#     # Calculate the center of mass of determined points for every frame
#     pts_surf = np.zeros([N_points * n_frames, 4])
#     for t in range(n_frames):
#         _pts = pts[t * N_points : (t + 1) * N_points, :]
#         CoM = _pts.mean(axis=0)

#         _pts_surf, _err, _FitP = get_traces(image = image_resampled.data[t],
#                                             start_pts=CoM[1:], target_pts=_pts[:, 1:])

#         pts_surf[t * N_points : (t + 1) * N_points, 1:] = _pts_surf
#         pts_surf[t * N_points : (t + 1) * N_points, 0] = t
    
#     surfs = reconstruct_surface(pts_surf, dims=image_resampled.data[0].shape,
#                                 curv_method=curv_method)
#     surfs = calculate_curvature(surfs, radius=2)
    
#     # add surfaces to viewer
#     vertices = []
#     faces = []
#     values = []
#     n_verts = 0
#     for idx, surf in enumerate(surfs):
#         # Add time dimension to points coordinate array
#         t = np.ones((surf.points().shape[0], 1)) * idx
#         vertices.append(np.hstack([t, surf.points()]))
        
#         # Offset indices in faces list by previous amount of points
#         faces.append(n_verts + np.array(surf.faces()))
#         values.append(surf.pointdata['Spherefit_curvature'])
        
#         # Add number of vertices in current surface to n_verts
#         n_verts += surf.N()
        
#     props = {'curvature': np.concatenate(values)}
#     viewer.add_points(np.vstack(vertices),
#                       properties=props,
#                       face_color='curvature',
#                       face_colormap='viridis',
#                       edge_width=0.1, size=0.6)
    
#     viewer.add_surface((
#         np.vstack(vertices),
#         np.vstack(faces),
#         np.concatenate(values)
#         ))
