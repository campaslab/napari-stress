# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData, SurfaceData, ImageData, LabelsData, LayerDataTuple, VectorsData
from napari.layers import Points, Layer

from typing import List

from functools import wraps
import inspect

import pandas as pd
import tqdm

def frame_by_frame(function, progress_bar: bool = False):

    @wraps(function)
    def wrapper(*args, **kwargs):

        sig = inspect.signature(function)
        annotations = [
            sig.parameters[key].annotation for key in sig.parameters.keys()
            ]

        converter = TimelapseConverter()

        args = list(args)
        n_frames = None

        # Convert 4D data to list(s) of 3D data for every supported argument
        # and store the list in the same place as the original 4D data
        index_of_converted_arg = []  # remember which arguments were converted

        for idx, arg in enumerate(args):
            if annotations[idx] in converter.supported_data:
                args[idx] = converter.data_to_list_of_data(arg, annotations[idx])
                index_of_converted_arg.append(idx)
                n_frames = len(args[idx])

        # apply function frame by frame
        #TODO: Put this in a thread by default?
        results = [None] * n_frames
        frames = tqdm.tqdm(range(n_frames)) if progress_bar else range(n_frames)
        for t in frames:
            _args = args.copy()

            # Replace 4D argument by single frame (arg[t])
            for idx in index_of_converted_arg:
                _args[idx] = _args[idx][t]

            results[t] = function(*_args, **kwargs)

        return converter.list_of_data_to_data(results, sig.return_annotation)
    return wrapper

class TimelapseConverter:
    """
    This class allows converting napari 4D layer data between different formats.
    """
    def __init__(self):

        # Supported LayerData types
        self.data_to_list_conversion_functions = {
            PointsData: self._points_to_list_of_points,
            SurfaceData: self._surface_to_list_of_surfaces,
            ImageData: self._image_to_list_of_images,
            LabelsData: self._image_to_list_of_images,
            VectorsData: self._vectors_to_list_of_vectors,
            Points: self._layer_to_list_of_layers
            }

    # Supported list data types
        self.list_to_data_conversion_functions = {
            Layer: self._list_of_layers_to_layer,
            PointsData: self._list_of_points_to_points,
            SurfaceData: self._list_of_surfaces_to_surface,
            ImageData: self._list_of_images_to_image,
            LabelsData: self._list_of_images_to_image,
            LayerDataTuple: self._list_of_layerdatatuple_to_layerdatatuple,
            List[LayerDataTuple]: self._list_of_multiple_ldtuples_to_multiple_ldt_tuples,
            VectorsData: self._list_of_vectors_to_vectors,
            Points: self._list_of_layers_to_layer
            }

        # This list of aliases allows to map LayerDataTuples to the correct napari.types
        self.tuple_aliases = {
            'points': PointsData,
            'surface': SurfaceData,
            'image': ImageData,
            'labels': LabelsData,
            'vectors': VectorsData
            }

        self.supported_data = list(self.list_to_data_conversion_functions.keys())

    def data_to_list_of_data(self, data, layertype: type) -> list:
        """
        Convert 4D data into a list of 3D data frames

        Parameters
        ----------
        data : 4D data to be converted
        layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
        `ImageData`, `LabelsData` or `List[LayerDataTuple]`

        Raises
        ------
        TypeError
            Error to indicate that the converter does not support the passed
            layertype

        Returns
        -------
        list: List of 3D objects of type `layertype`

        """
        if not layertype in list(self.data_to_list_conversion_functions.keys()):
            raise TypeError(f'{layertype} data to list conversion currently not supported.')

        conversion_function = self.data_to_list_conversion_functions[layertype]
        return conversion_function(data)

    def list_of_data_to_data(self, data, layertype: type):
        """
        Function to convert a list of 3D frames into 4D data.

        Parameters
        ----------
        data : list of 3D data (time)frames
        layertype : layerdata type. Can be any of 'PointsData', `SurfaceData`,
        `ImageData`, `LabelsData` or `List[LayerDataTuple]`

        Raises
        ------
        TypeError
            Error to indicate that the converter does not support the passed
            layertype

        Returns
        -------
        4D data of type `layertype`

        """
        if not layertype in self.supported_data:
            raise TypeError(f'{layertype} list to data conversion currently not supported.')
        conversion_function = self.list_to_data_conversion_functions[layertype]
        return conversion_function(data)

    # =============================================================================
    # Layers
    # =============================================================================

    def _list_of_layers_to_layer(self, layer_list: list) -> Layer:
        """Convert a list of layers to single layer."""
        list_of_layerdatatuples = [layer.as_layerdatatuple() for layer in layer_list]
        layerdatatuple = self.list_of_layerdatatuple_to_layerdatatuple(list_of_layerdatatuples)

        converted_layer = Layer.create(layerdatatuple[0],
                                       meta = layerdatatuple[1],
                                       layer_type = layerdatatuple[2])

        return converted_layer

    def _layer_to_list_of_layer(self, layer: Layer) -> list:
        """Convert layer to list of layers."""
        ldtuple = layer.as_layer_data_tuple()
        list_of_ldtuples = self._ldtuple_to_list_of_ldtuple(ldtuple)

        list_of_layers = [Layer.create(ldt[0], ldt[1], ldt[2]) for ldt in list_of_ldtuples]
        return list_of_layers

    # =============================================================================
    # LayerDataTuple(s)
    # =============================================================================

    def _ldtuple_to_list_of_ldtuple(self, tuple_data: list) -> LayerDataTuple:
        """Convert single 4D layerdatatuple to list of layerdatatuples."""
        layertype = self.tuple_aliases[tuple_data[-1]]

        list_of_data = self.data_to_list_of_data(tuple_data[0],
                                                 layertype=layertype)

        if len(list_of_data) == 1:
            list_of_features = [tuple_data[1]['features']]
            list_of_metadata = [tuple_data[1]['metadata']]

        else:
            # unstack features
            if 'features' in tuple_data[1].keys():
                # group features by time-stamp
                features = tuple_data[1]['features']
                list_of_features = [
                    x for _, x in features.groupby(tuple_data[0][:, 0])
                    ]

            # unstack metadata
            if 'metadata' in tuple_data[1].keys():
                metadata = tuple_data[1]['metadata']
                list_of_metadata = [
                    {key : value[i] for key, value in metadata.items()}
                    for i in range(len(list_of_data))
                    ]

        list_of_props = [{'features': features, 'metadata': metadata}
                         for features, metadata in zip(
                                 list_of_features, list_of_metadata
                                 )]

        list_of_ldtuples = [(data, props, layertype)
                            for data, props in zip(list_of_data, list_of_props)]

        return list_of_ldtuples


    def _list_of_multiple_ldtuples_to_multiple_ldt_tuples(self,
                                                          tuple_data: list,
                                                          ) -> List[LayerDataTuple]:
        """If a function returns a list of LayerDataTuple"""

        # Convert data to array with dimensions [frame, results, data]
        data = np.stack(tuple_data)

        if len(data) == 1:
            layertypes = data[:,..., -1].squeeze()
        else:
            layertypes = data[:,..., -1].squeeze()[0]

        converted_tuples = []
        for idx, res_type in enumerate(layertypes):
            tuples_to_convert = data[:, idx]
            converted_tuples.append(
                self._list_of_layerdatatuple_to_layerdatatuple(list(tuples_to_convert))
                )

        return converted_tuples


    def _list_of_layerdatatuple_to_layerdatatuple(self,
                                                 tuple_data: list
                                                 ) -> LayerDataTuple:
        """
        Convert a list of 3D layerdatatuple objects to a single 4D LayerDataTuple
        """
        layertype = self.tuple_aliases[tuple_data[-1][-1]]

        # Convert data to array with dimensions [frame, data]
        data = np.stack(tuple_data)
        properties = data[:, 1]

        # If data was only 3D
        _properties = {}
        if len(data) == 1:
            if 'features' in properties[0].keys():
                _properties['features'] = data[0][1]['features']
            if 'metadata' in properties[0].keys():
                _properties['metadata'] = data[0][1]['metadata']
        else:
            # Stack features
            if 'features' in properties[0].keys():
                features = self._list_of_dictionaries_to_dictionary([frame['features'] for frame in properties])
                _properties['features'] = features
                [frame.pop('features') for frame in properties]

            # Stack metadata
            if 'metadata' in properties[0].keys():
                metadata_list = [frame['metadata'] for frame in properties]
                new_metadata = {}
                for key in metadata_list[0].keys():
                    new_metadata[key] = [frame[key] for frame in metadata_list]

                _properties['metadata'] = new_metadata
                [frame.pop('metadata') for frame in properties]

        # Stack the other properties
        layer_props = self._list_of_dictionaries_to_dictionary(properties)

        # exclude 'scale' from stacked metadata
        if 'scale' in layer_props.keys() and len(data) != 1:
            layer_props['scale'] = properties[0]['scale']

        for key in layer_props.keys():
            _properties[key] = layer_props[key]

        # Reminder: Each list entry is tuple (data, properties, type)
        results = [None] * len(data)  # allocate list for results

        dtype = data[0, -1]
        result = [None] * 3
        result[0] = self.list_to_data_conversion_functions[layertype]([x for x in data[:, 0]])
        result[1] = _properties
        result[2] = dtype

        return tuple(result)

    def _list_of_dictionaries_to_dictionary(self, dictionaries: list) -> dict:
        _dictionary = {}
        for key in dictionaries[-1].keys():
            if isinstance(dictionaries[-1][key], dict):
                _dictionary[key] = self._list_of_dictionaries_to_dictionary(
                    [frame[key] for frame in dictionaries])
                continue
            elif isinstance(dictionaries[-1][key], str):
                _dictionary[key] = dictionaries[-1][key]
                continue

            if hasattr(dictionaries[-1][key], '__len__'):
                _dictionary[key] = np.concatenate([frame[key] for frame in dictionaries]).squeeze()
            else:
                _dictionary[key] = dictionaries[-1][key]
        return _dictionary

    # =========================================================================
    # Layers
    # =========================================================================

    def _layer_to_list_of_layers(self, layer: Layer) -> list:
        ldtuple = layer.as_layer_data_tuple()
        list_of_layerdatatuples = self._ldtuple_to_list_of_ldtuple(ldtuple)

        layers = []

        for ldt in list_of_layerdatatuples:
            layers.append(Layer.create(data=ldt[0], meta=ldt[1],
                                       layer_type=ldtuple[-1]))
        return layers

    # =============================================================================
    # Images
    # =============================================================================

    def _image_to_list_of_images(self, image: ImageData) -> list:
        """Convert 4D image to list of images"""
        while len(image.shape) < 4:
            image = image[np.newaxis, :]
        return list(image)

    def _list_of_images_to_image(self, images: list) -> ImageData:
        """Convert a list of 3D image data to single 4D image data."""
        return np.stack(images)

    # =============================================================================
    # Vectors
    # =============================================================================

    def _vectors_to_list_of_vectors(self, vectors: VectorsData) -> list:
        base_points = vectors[:, 0]
        vectors = vectors[:, 1]

        # the vectors and points should abide to the same dimensions
        point_list = self._points_to_list_of_points(base_points)
        vector_list = self._points_to_list_of_points(vectors)

        output_vectors = [
            np.stack([pt, vec]).transpose((1, 0, 2)) for pt, vec in zip (point_list, vector_list)
            ]
        return output_vectors

    def _list_of_vectors_to_vectors(self, vectors: list) -> VectorsData:
        base_points = [v[:, 0] for v in vectors]
        directions = [v[:, 1] for v in vectors]

        base_points = self._list_of_points_to_points(base_points)
        directions = self._list_of_points_to_points(directions)

        vectors = np.stack([base_points, directions]).transpose((1,0,2))
        return vectors


    # =============================================================================
    # Surfaces
    # =============================================================================

    def _surface_to_list_of_surfaces(self, surface: SurfaceData) -> list:
        """Convert a 4D surface to list of 3D surfaces"""

        points = surface[0]
        faces = np.asarray(surface[1], dtype=int)

        # Check if values were assigned to the surface
        has_values = False
        if len(surface) == 3:
            has_values = True
            values = surface[2]

        while points.shape[1] < 4:
            t = np.zeros(len(points), dtype=points.dtype)
            points = np.insert(points, 0, t, axis=1)

        n_frames = len(np.unique(points[:, 0]))
        points_per_frame = [sum(points[:, 0] == t) for t in range(n_frames)]

        # find out at which index in the point array a new timeframe begins
        frame_of_face = [points[face[0], 0] for face in faces]
        idx_face_new_frame = list(np.argwhere(np.diff(frame_of_face) != 0).flatten() + 1)
        idx_face_new_frame = [0] + idx_face_new_frame + [len(faces)]

        # Fill list of frames with correct points and corresponding faces
        # as previously determined
        surfaces = [None] * n_frames
        for t in range(n_frames):

            # Find points with correct frame index
            _points = points[points[:, 0] == t, 1:]

            # Get parts of faces array that correspond to this frame
            _faces = faces[idx_face_new_frame[t] : idx_face_new_frame[t+1]] - sum(points_per_frame[:t])

            # Get values that correspond to this frame
            if has_values:
                _values = values[points[:, 0] == t]
            else:
                _values = np.ones(len(_points))

            surfaces[t] = (_points, _faces, _values)

        return surfaces

    def _list_of_surfaces_to_surface(self, surfaces: list) -> tuple:
        """
        Convert list of 3D surfaces to single 4D surface.
        """
        # Put vertices, faces and values into separate lists
        # The original array is tuple (vertices, faces, values)
        vertices = [surface[0] for surface in surfaces]  # retrieve vertices
        faces = [surface[1] for surface in surfaces]  # retrieve faces

        # Surfaces do not necessarily have values - check if this is the case
        if len(surfaces[0]) == 3:
            values = np.concatenate([surface[2] for surface in surfaces])  # retrieve values if existant
        else:
            values = None

        vertices = self._list_of_points_to_points(vertices)

        n_vertices = 0
        for idx, surface in enumerate(surfaces):

            # Offset indices in faces list by previous amount of points
            faces[idx] = n_vertices + np.array(faces[idx])

            # Add number of vertices in current surface to n_vertices
            n_vertices += surface[0].shape[0]

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
        n_frames = len(points)
        n_points = sum([len(frame) for frame in points])
        if n_frames > 1: # actually a timelapse
            t = np.concatenate([[idx] * len(frame) for idx, frame in enumerate(points)])

            points_out = np.zeros((n_points, 4))
            points_out[:, 1:] = np.vstack(points)
            points_out[:, 0] = t
        else:
            points_out = np.vstack(points)

        return points_out


    def _points_to_list_of_points(self, points: PointsData) -> list:
        """Convert a 4D point array to list of 3D points"""

        while points.shape[1] < 4:
            t = np.zeros(len(points), dtype=points.dtype)
            points = np.insert(points, 0, t, axis=1)

        n_frames = len(np.unique(points[:, 0]))

        # Allocate empty list
        points_out = [None] * n_frames

        # Fill the respective entries in the list with coordinates in the
        # original data where time-coordinate matches the current frame
        for t in range(n_frames):

            # Find points with correct frame index
            points_out[t] = points[points[:, 0] == t, 1:]

        return points_out
