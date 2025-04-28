(glossary:stresses:cell_stresses)=
## Quantifying Higher Order Stresses

To quantify the stresses that arise from higher-order deformation modes (beyond the ellipsoidal mode), which are related to smaller length scales closer to the size of a cell, we calculate how much the droplet's shape deviates from the ellipsoidal shape. For any given timepoint, we subtract the calculated total anisotropic stress value $ \sigma_A $ from the tissue-scale stresses associated with the ellipsoidal deformation, namely $ \sigma_T $, to obtain the cell-scale stresses.

Specifically, to compute the cell-scale stress anisotropy between points $ p $ and $ q $ on the surface, we use the following equation:

$$
\sigma_C^A = \gamma \left[ \left(H(p) - H(q)\right) - \left(H_e(p) - H_e(q)\right) \right]
$$

Here, we choose the points $ p $ and $ q $ on the ellipsoid that correspond to the same ellipsoidal coordinates as $ p $ and $ q $ on the droplet.

### Finding the Extrema on the Deformed Surface

To identify the extreme points (maxima and minima) on the deformed surface, we use a method called surface triangulation on the 5810 Lebedev nodes that represent the surface. Two points are considered neighbors if they are connected by an edge.

- **Local maxima** are points where the value of cellular stress is greater than or equal to the values of all their neighboring points.
- **Local minima** are points where the value of cellular stress is less than or equal to the values of all their neighboring points.

The extrema stresses (or adjacent extrema stresses) are obtained by calculating the difference in cell-scale stresses between maxima and minima (or between adjacent maxima and minima).

Calculating these stresses is part of the [comprehensive stress quantification workflow](toolboxes:analyze_everything)=
