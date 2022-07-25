# -*- coding: utf-8 -*-
from functools import wraps
import inspect

def naparify_measurement(function):

    @wraps(function)
    def wrapper(layer, *args, **kwargs):
        if 'metadata' in kwargs:
            layer.metadata = kwargs['metadata']

        return layer
    return wrapper
