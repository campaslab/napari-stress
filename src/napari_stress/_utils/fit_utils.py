import numpy as np
import inspect

def _sigmoid(x, center, amplitude, slope, offset):
    """
    Sigmoidal fit function
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    """
    return amplitude / (1 + np.exp(-slope*(x-center))) + offset

def _gaussian(x, center:float, sigma:float, amplitude:float):
    """
    Gaussian normal fit function
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(x - center)**2 / (2*sigma**2))

def _detect_maxima(profile, center: float = None):
    """
    Simple function to find the maximum's index of data with a single peak.
    """
    return np.argmax(profile)

def _detect_drop(profile, center: float = None):
    """
    Simple function to find the location of the steepest drop for sigmoidal data
    """
    return np.argmax(np.abs(np.diff(profile)))

def _func_args_to_list(func: callable) -> list:
    """
    Simple function to return a functions' keywords as a list of strings
    """

    sig = inspect.signature(func)
    return list(sig.parameters.keys())
