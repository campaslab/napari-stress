# -*- coding: utf-8 -*-
"""
Definition of napari-stress specific types.

This eases passing data between napari layers and functions from napari stress
that return or receive metadata or features from a layer.
"""

import numpy as np
from typing import (
    NewType
)

from ._stress.manifold_SPB import manifold

_METADATAKEY_MANIFOLD = 'manifold'
_METADATAKEY_MEAN_CURVATURE = 'Mean_curvature'
_METADATAKEY_H0_SURFACE_INTEGRAL = 'H0_surface_integral'
_METADATAKEY_H0_ARITHMETIC_AVERAGE = 'H0_arithmetic_average'
_METADATAKEY_ANISOTROPIC_STRESS = 'anisotropic_stress'


Manifold = NewType(_METADATAKEY_MANIFOLD, manifold)
Curvature = NewType(_METADATAKEY_MEAN_CURVATURE, np.ndarray)
H0_surface_integral = NewType(_METADATAKEY_H0_SURFACE_INTEGRAL, float)
