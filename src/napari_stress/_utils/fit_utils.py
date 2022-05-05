# -*- coding: utf-8 -*-
import numpy as np
import inspect

def _sigmoid(x, center, amplitude, slope, offset):
    "https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python"
    return amplitude / (1 + np.exp(-slope*(x-center))) + offset

def _gaussian(x, center, sigma, amplitude):
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(x - center)**2 / (2*sigma**2))

def _detect_maxima(profile, center: float = None):
    return np.argmax(profile)

def _detect_drop(profile, center: float = None):
    return np.argmax(np.diff(profile))

def _func_args_to_list(func: callable) -> list:

    sig = inspect.signature(func)
    return list(sig.parameters.keys())
