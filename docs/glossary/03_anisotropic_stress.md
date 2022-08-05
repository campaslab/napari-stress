(spherical_harmonics:stress)=
# Analyzing stresses

The [spherical harmonics expansion](spherical_harmonics:mathematical_basics) and the [mean curvature](spherical_harmonics:measurements:mean_curvature) can be used to calculate the quantitative mechanical stress at work in or on analayzed objects, such as oild droplets. This section explains the related quantities and calculations.

(spherical_harmonics:stress:general)=
## Anisotropic stress

The anisotropic stress at a point $\vec{x}_i$ is expressed as $\sigma_i$. It can be calculated by subtracting the surface-integrated curvature $H_{0, surf~int}$ from the curvature $H_i$ and multiply it with the droplet's [interfacial tension](https://en.wikipedia.org/wiki/Surface_tension) $\gamma$:

\begin{align}
\sigma = 2\gamma  \left(H_i - $H_{0, surf~int}$\right)
\end{align}

The value of $\gamma$ is specific to the used material.