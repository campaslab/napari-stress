# -*- coding: utf-8 -*-
from functools import wraps
import inspect
import napari
import pandas as pd

def naparify_measurement(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        sig = inspect.signature(function)
        annotations = [
            sig.parameters[key].annotation for key in sig.parameters.keys()
            ]

        # If a layer was passed instead of the necessary parameters:
        # if napari.layers.Layer in annotations and :
        if 'viewer' in kwargs:
            new_args = []
            metadata = args[0].metadata
            for annotation in annotations:
                for key in metadata.keys():
                    if isinstance(metadata[key], annotation):
                        new_args.append(metadata[key])

            features, metadata = function(*new_args, **kwargs)

            for key in features.keys():
                args[0].features[key] = features[key]

            for key in metadata.keys():
                args[0].metadata[key] = metadata[key]

            return args[0]

        else:
            return function(*args, **kwargs)


        # If a viewer exists...?
    return wrapper
