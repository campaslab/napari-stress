(glossary:stresses:tissue_stresses)=
# Tissue-Scale Stresses

## Understanding the Shape of a Droplet

To study the shape of a droplet, we fit it to an ellipsoid, which is like a stretched or squashed sphere. We do this by using a method called "least-squares fitting" on the droplet's 3D shape data, which is represented by a collection of points (a point cloud). The equation we use to describe this ellipsoid looks like this:

$[ Ax^2 + Bxy + Cy^2 + Dxz + Eyz + Fz^2 + Gx + Hy + Iz = 1 ]$

Here, A, B, C, D, E, F, G, H, and I are coefficients we calculate to best fit the shape.

For more information on how to fit an ellipsoid to a point cloud from code see [this example](glossary:ellipse_fitting:code) - or [this tutorial](glossary:ellipse_fitting:interactive) on how to do it interactively from the viewer.

## Finding the Curvature of the Ellipsoid

To better understand the ellipsoid, we change its coordinate system so that the new axes (x1, x2, x3) align with its main directions: the major axis (longest), medial axis (middle), and minor axis (shortest). These new coordinates are centered at the middle of the ellipsoid, and we describe them using two angles, u and v:

$$\begin{align*}
[ x_1 &= a \cdot \cos(u) \cdot \sin(v) ]\\
[ x_2 &= b \cdot \sin(u) \cdot \sin(v) ]\\
[ x_3 &= c \cdot \cos(v) ]
\end{align*}
$$

Here, **a**, **b**, and **c** represent the lengths of the ellipsoid's axes, and **u** and **v** help us describe any point on the surface.

## Calculating the Mean Curvature

Using the equations above, we can calculate something called the **mean curvature** (denoted as $H_e$) at any point on the ellipsoid's surface. Curvature tells us how much the surface bends at a point.

## Relating Curvature to Tissue Stresses

The maximum and minimum mean curvatures of the ellipsoid, which happen at specific points on its surface, are:

- **Maximum Mean Curvature** $ H_{e,M} = \frac{a}{2c^2} + \frac{a}{2b^2}$
- **Minimum Mean Curvature** $ H_{e,N} = \frac{c}{2b^2} + \frac{c}{2a^2}$

The difference between these curvatures helps us understand the anisotropic tissue stresses, which describe how different directions in the tissue experience different amounts of stress. The formula to calculate this stress is:

$$
 \sigma_T = 2 \gamma (H_{e,M} - H_{e,N}) = \gamma \left(\frac{a}{c^2} + \frac{a-c}{b^2} - \frac{c}{a^2}\right)
$$
Here, $ \gamma $ represents the surface tension of the droplet, which is a measure of how tightly the surface molecules are held together.

For more information on ellipsoids, you can check out this [link](https://mathworld.wolfram.com/Ellipsoid.html). Measureing tissue stresses is part of [this workflow](toolboxes:analyze_everything)
