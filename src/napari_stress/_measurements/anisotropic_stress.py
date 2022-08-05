# -*- coding: utf-8 -*-
from ..types import Curvature, H0_surface_integral
from ..types import _METADATAKEY_ANISOTROPIC_STRESS
from .utils import naparify_measurement

@naparify_measurement
def anisotropic_stress(mean_curvature: Curvature,
                       H0_surface_curvature: H0_surface_integral,
                       gamma: float) -> (None, dict, None):
    """
    Convert mean curvature to anisotropic stress.

    Parameters
    ----------
    mean_curvature : Points
    gamma : float
        Interfacial tension value for droplet

    Returns
    -------
    None

    """
    anisotropic_stress = 2 * gamma * (mean_curvature - H0_surface_curvature)

    features = {_METADATAKEY_ANISOTROPIC_STRESS: anisotropic_stress}

    return (None, features, None)
