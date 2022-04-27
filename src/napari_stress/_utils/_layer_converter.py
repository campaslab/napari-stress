# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData, SurfaceData, ImageData, LabelsData, LayerDataTuple

from typing import List

class Converter:
    def __init__(self):

        self.funcs_data_to_list = {
            PointsData: self._points_to_list_of_points,
            SurfaceData: self._surface_to_list_of_surfaces,
            ImageData: self._image_to_list_of_images,
            LabelsData: self._image_to_list_of_images
            }

        self.funcs_list_to_data = {
            PointsData: self._list_of_points_to_points,
            SurfaceData: self._list_of_surfaces_to_surface,
            ImageData: self._list_of_images_to_image,
            LabelsData: self._list_of_images_to_image,
            List[LayerDataTuple]: self._list_of_layerdatatuple_to_layerdatatuple
            }

        # This list of aliases allows to map LayerDataTuples to the correct napari.types
        self.tuple_aliases = {
            'points': PointsData,
            'surface': SurfaceData,
            'image': ImageData,
            'labels': LabelsData,
            }

    def data_to_list_of_data(self, data, layertype) -> list:
        """
        Function to convert 4D data into a list of 3D frames

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        layertype : TYPE
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        conversion_function = self.funcs_data_to_list(layertype)
        return conversion_function(data)

    def list_of_data_to_data(self, data, layertype):
        conversion_function = self.funcs_list_to_data(layertype)
        return conversion_function(data)


    # =============================================================================
    # LayerDataTuple
    # =============================================================================
    def _list_of_layerdatatuple_to_layerdatatuple(self,
                                                 tuple_data: list
                                                 ) -> LayerDataTuple:
        """
        Convert a list of 3D layerdatatuple objects to a single 4D LayerDataTuple
        """
        layertype = self.tuple_aliases[tuple_data[-1]]

        # Convert data to array with dimensions [result, frame, data]
        data = list(np.asarray(tuple_data).transpose((1, 0, -1)))

        # Reminder: Each list entry is tuple (data, properties, type)
        results = [None] * len(data)  # allocate list for results
        for idx, res in enumerate(data):
            dtype = res[0, -1]
            _result = [None] * 3
            _result[0] = self.funcs_list_to_data[layertype](res[:, 0])
            _result[1] = res[0, 1]  # smarter way to combine properties?
            _result[2] = dtype
            results[idx] = _result

        return results

    # =============================================================================
    # Images
    # =============================================================================

    def _image_to_list_of_images(self, image: ImageData) -> list:
        """Convert 4D image to list of images"""
        #TODO: Check if it actually is 4D
        return list(image)

    def _list_of_images_to_image(self, images: list) -> ImageData:
        """Convert a list of 3D image data to single 4D image data."""
        return np.stack(images)


    # =============================================================================
    # Surfaces
    # =============================================================================

    def _surface_to_list_of_surfaces(self, surface: SurfaceData) -> list:
        """Convert a 4D surface to list of 3D surfaces"""
        #TODO: Check if it actually is 4D
        points = surface[0]
        faces = np.asarray(surface[1], dtype=int)

        n_frames = len(np.unique(points[:, 0]))
        points_per_frame = [sum(points[:, 0] == t) for t in range(n_frames)]

        idx_face_new_frame = []
        t = 0
        for idx, face in enumerate(faces):
            if points[face[0], 0] == t:
              idx_face_new_frame.append(idx)
              t += 1
        idx_face_new_frame.append(len(faces))

        surfaces = [None] * n_frames
        for t in range(n_frames):
            _points = points[points[:, 0] == t, 1:]
            _faces = faces[idx_face_new_frame[t] : idx_face_new_frame[t+1]-1] - sum(points_per_frame[:t])
            surfaces[t] = (_points, _faces)

        return surfaces

    def _list_of_surfaces_to_surface(self, surfs: list) -> tuple:
        """
        Convert list of 3D surfaces to single 4D surface.
        """

        vertices = [surf[0] for surf in surfs]
        faces = [surf[1] for surf in surfs]
        values = None
        if len(surfs[0]) == 3:
            values = np.concatenate([surf[2] for surf in surfs])

        vertices = list_of_points_to_points(vertices)

        n_verts = 0
        for idx, surf in enumerate(surfs):

            # Offset indices in faces list by previous amount of points
            faces[idx] = n_verts + np.array(faces[idx])

            # Add number of vertices in current surface to n_verts
            n_verts += surf[0].shape[0]

        faces = np.vstack(faces)

        if values is None:
            return (vertices, faces)
        else:
            return (vertices, faces, values)


    # =============================================================================
    # Points
    # =============================================================================

    def _list_of_points_to_points(self, points: list) -> np.ndarray:
        """Convert list of 3D point data to single 4D point data."""

        n_points = sum([len(frame) for frame in points])
        t = np.concatenate([[idx] * len(frame) for idx, frame in enumerate(points)])

        points_out = np.zeros((n_points, 4))
        points_out[:, 1:] = np.vstack(points)
        points_out[:, 0] = t

        return points_out


    def _points_to_list_of_points(self, points: PointsData) -> list:
        """Convert a 4D point array to list of 3D points"""
        #TODO: Check if it actually is 4D
        n_frames = len(np.unique(points[:, 0]))

        points_out = [None] * n_frames
        for t in range(n_frames):
            points_out[t] = points[points[:, 0] == t, 1:]

        return points_out
