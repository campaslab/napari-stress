(spherical_harmonics:mathematical_basics)=
(glossary:spherical_harmonics:mathematical_basics)=
# Spherical harmonics

In this notebook you will learn about [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics), how they work and how they can be used with a spherical harmonics expansion to represent biological data.

## Polar coordinates

Generally speaking, spherical harmonics provide a mathematical framework to tackle problems that are best described in polar coordinates. When we use polar coordinates, we describe the location of a point in space in terms of

* latitude $\theta$
* longitude $\phi$
* radius $r$

....rather than x, y and z, which we refer to as *cartesian coordinates*. There are many instances of problems that are better described in polar coordinates:

- [Quantum mechanics](https://en.wikipedia.org/wiki/Atomic_orbital): The probability of encountering an electron at a particular position in the orbit around the nucleus of atoms follows a particular distribution which can be described well with spherical harmonics.
- [Earth magnetism](https://en.wikipedia.org/wiki/Earth%27s_magnetic_field): Magnetic forces on the surface of the earth can easily be described with respect to latitude and longtitude
- The shape of cell nuclei: This is what we will be doing with napari-stress :)

## Spherical harmonics

What the above listed probels from quantum mechanics and magentism have in common, is that the problems that are described have a periodical nature: For instance, if youwere to circumvent the surface of the earth along a line of constant longitude, you would experience a periodical rise and fall of magnetic field strength as you get closer and further to the poles. Naturally, such phenomena are well described by trigonometric functions (*cosine* & *sine*). Spherical harmonics generalize these functions to describe phenomena on the surface of spheres.

Mathematically, spherical harmonic functions $Y_l^m$ of degree $l$ and order $m$ follow the following formulation:

$Y_l^m(\theta, \phi) = N e^{im\phi} P_l^m(\theta, \phi)$

where $N$ and $P_l^m$ are a problem-specific normalization constant and the [associated Legendre polynomials](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials). You may notice, that the factor $e^{im\phi}$ introduces an imaginary and a real part of this function:

$
Im(Y_l^m) = \sin(\left|m\right| \phi) \cdot P_l^{\left|m\right|}\\
Re(Y_l^m) = \cos(m \phi) \cdot P_l^{m}
$

(spherical_harmonics:mathematical_basics:degree)=
## Spherical harmonics expansion

This framework can be used to represent any function $f(\theta, \phi)$ on the surface of a sphere as a linear superposition of $Y_l^m$ of different degree $l$ and order $m$:

$f(\theta, \phi) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_l^m Y_l^m(\theta, \phi)$

where $f_l^m$ are the coefficients that determine how much each degree and order of the spherical harmonics functions $Y_l^m(\theta, \phi)$ contribute to the sum. In other words, you can use this superposition to approximate the value of any function that depends on latitude and longitude as inputs. It is important to understand that this definition is assuming spheres of radius $r=1$ - the result of $f(\theta, \phi)$ can represent any property on the surface of this sphere. The Wikipedia page features a nice [visualization](https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Rotating_spherical_harmonics.gif) for the contributions of spherical harmonics of specific degree and order on the surface of a sphere.

Using this framework has several desirable characteristics:

- Any real function on the surface of a sphere can be approximated with a spherical harmonics expansion
- An obtained spherical harmonics expansion can be described entirely by the coefficients $f_l^m$

## Spherical harmonics shape description

In the context of napari-stress, a spherical harmonics expansion is used to obtain spherical harmonics coefficients $f_l^m$ that describe the position of a vertex on the surface of an object (e.g., an oil droplet):

![](../../imgs/viewer_screenshots/open_sample_droplet1.png)

In this case, the property that should be described in terms of latitude and longitude, is the *position* of the vertices on the surface. In other words, we would like to obtain a relation such as follows:

$f(\vec{x}_i) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_l^m Y_l^m(\theta_i, \phi_i)$

There are different ways of achieving this:

### i ) Calculate point radius as function of latitude and longitude

In the simplest approach, we can convert all points $\vec{x}_i$ on a surface to polar coordinates with respect to their center of mass:

$\vec{x}_i = [\theta_i, \phi_i, r_i]$

We can then use a sperhical harmonics expansion to describe the radius $r$ as a function of latitude of longitude ($r = r(\theta, \phi)$:

$
r(\theta, \phi) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_l^m Y_l^m(\theta_i, \phi_i)
$

The expression for the ccordinate of a point $\vec{x}_i$ then becomes:

$
\vec{x}_i = [\theta_i, \phi_i, r(\theta, \phi)]
$

This works well for point clouds that retain sphere-like geometries , but may lead to an erroneous expansion if the to-be-approximated pointcloud is highly curved and non-convex.

### ii ) Calculate cartesian coordinate as a function of latitude and longitude

To address the problem of approximating highly non-sphere-like pointclouds, it is of advantage to approximate the different components of a point coordinate separately. This can be done by not only expressing the radius of a point as a function of latitude and longitude, but each of its cartesian components:

$\vec{x}_i = [x_i, y_i, z_i]$

Each of these can be approximated by a separate spherical harmonics expansion with a separate set of spherical harmonics coefficients $f_l^m$:

$
x(\theta, \phi) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_{l, x}^m Y_l^m(\theta_i, \phi_i)\\
y(\theta, \phi) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_{l, y}^m Y_l^m(\theta_i, \phi_i)\\
z(\theta, \phi) = \sum_{l=0}^{\inf} \sum_{m=-l}^{l} f_{l, z}^m Y_l^m(\theta_i, \phi_i)
$

The position of a point $\vec{x}_i$ can then be described by the following expression

$\vec{x}_i = [x(\theta_i, \phi_i), y(\theta_i, \phi_i), z(\theta_i, \phi_i)]$

where the coefficients $f_x$, $f_y$, $f_z$ refer to the spherical harmonics coefficients of the three distinct expansions.

### iii ) Calculate other coordinates as a function of latitude and longitude

In principle, the previously described coordinate parametrizations can be expanded to any set of coordinate parametrization. Other coordinate formulations that can be used for this sort of coordinate parametrization are [cylindrical coordinates ](https://en.wikipedia.org/wiki/Cylindrical_coordinate_system) or [ellipsoidal coordinates](https://en.wikipedia.org/wiki/Ellipsoidal_coordinates).
