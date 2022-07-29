# -*- coding: utf-8 -*-
from functools import wraps
import inspect
import napari

def convert(value, new_annotation):
    """Check if an object is of a specific type and convert it if it isn't."""
    from .._stress.manifold_SPB import manifold
    from napari_stress import _METADATAKEY_MANIFOLD

    if isinstance(value, new_annotation):
        return value

    if isinstance(value, napari.layers.Layer) and new_annotation is manifold:
        return value.metadata[_METADATAKEY_MANIFOLD]
    raise ValueError(f'Unsupported conversion {type(value)} -> {new_annotation}')



def naparify_measurement(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        wrapper_sig = inspect.signature(wrapper)

        annotation_was_converted = False

        for key in sig.parameters.keys():
            value = bound.arguments[key]
            original_annotation = sig.parameters[key].annotation
            new_annotation = wrapper_sig.parameters[key].annotation

            print('Old:', original_annotation, 'New:', new_annotation)
            if original_annotation is not new_annotation:
                converted_value = convert(value, original_annotation)
                bound.arguments[key] = converted_value

                if not value is converted_value:
                    annotation_was_converted = True

        result = function(*bound.args, **bound.kwargs)

        # depending on what was received (layer or manifold),
        # append result to input layer or return raw return value
        if not annotation_was_converted:
            return result
        else:
            if isinstance(value, napari.layers.Layer):
                features, metadata = result[1], result[2]
                for key in features.keys():
                    value.features[key] = features[key]
                for key in metadata.keys():
                    value.metadata[key] = metadata[key]
            return None



    # If we find a manifold parameter in the passed data, we replace its
    # annotation with napari.layers.Points
    import inspect
    from .._stress.manifold_SPB import manifold
    sig = inspect.signature(wrapper)
    parameters = []
    for name, value in sig.parameters.items():

        # replace manifold parameter annotation with napari.layers.Points
        if value.annotation is manifold:
            parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="napari.layers.Points"))
        else:
            parameters.append(value)

    wrapper.__signature__ = inspect.Signature(parameters)
    return wrapper
