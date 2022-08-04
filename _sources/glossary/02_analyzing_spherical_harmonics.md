(spherical_harmonics:measurements)=
# Analyzing spherical harmonics

Performing a spherical harmonics expansion on a given pointcloud yields a number of properties which can be used to analyze the obtained expansion (and thus, the original pointcloud) quantitiatively. This section describes the nature of the available measurements in napari stress.

(spherical_harmonics:measurements:fit_residue)=
## Spherical harmonics expansion fit residue
In practice, a spherical harmonics expansion only uses contributions up to a defined degree $l$ (the order $m$ is constrained by $-l < m < l$). This can have various reasons: For once, using high degrees for the expansion can lead to high computational expense whereas using only the first few orders may already lead to acceptable results. On the other hand, using high orders can lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting) - using an infinitely high degree will lead to an expansion that describes each point perfectly, but does not capture the overall shape well anymore.

Depending on the highest used degree for the expansion, the expansion will lead to a residual error $\Delta$:

$\Delta = \vec{x}_i - f(\theta_i, \phi_i)$

where $f(\theta_i, \phi_i)$ is a superposition of multiple spherical harmonics base functions (see section about [mathematical basics](spherical_harmonics:mathematical_basics) for details).

This example shows the result of a spherical harmonics expansion of low degree $l$ for the approximation of a pointcloud:

![](../imgs/viewer_screenshots/fit_spherical_harmonics3.png)

The orange points show the input pointcloud consisting of points $\vec{x}_i$, the color-coded points show the results of the spherical harmonics expansion with $l<=1$ - which corresponds to only the basic modes of spherical harmonics being used - which in turn corresponds to a sphere. The color scale of the resulting sphere corresponds to the fit residue of the expansion for every point $\Delta_i$.