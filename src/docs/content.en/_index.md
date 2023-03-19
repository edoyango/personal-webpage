---
title: Landing
type: docs
---

# GraSPH Documentation


## This Page
Provides more detailed documentation than in the GitHub README. It will help you create your own geometry. In the future, I aim to add documentation that explains the code structure and how to modify the code. Topics are on the menu on the left. If you have feedback (for the docs or the code), don't hestiate to [open an issue on the GitHub](https://github.com/edoyango/GraSPH/issues/new), or [email me](mailto://edward_yang_125@hotmail.com). But keep in mind I do this in my free time.

{{< button href="/" >}}Go Back to Main Site{{< /button >}}

## About GraSPH
Gra(nular)SPH is a [Smoothed Particle Hydrodynamics (SPH)](https://doi.org/10.1093%2Fmnras%2F181.3.375) code written in Fortran. It uses the SPH method to simulate fluid dynamics using the "weakly" compressible approach described in [Monaghan 1994](https://doi.org/10.1006/jcph.1994.1034). Currently, it can only simulate fluids, but will be developed in the future to incorporate simple granular models such as that described in [Bui et al. (2008)](https://doi.org/10.1002/nag.688). 

GraSPH can be run in serial, or in parallel using Coarray Fortran (which is based on the MPI implementation described in [Yang et al. (2021)](https://doi.org/10.1016/j.compgeo.2020.103474)). It has further been developed to be able to run on an NVIDIA GPU.

It uses HDF5 as the IO library, so output files can be read from Matlab or Python (using h5py). Future work include writing the code in a HDF5 file format compatible with ParaView.

GraSPH is a hobby project of mine that builds on the work I did with SPH in my masters. It is not supported by MCGLAB, which its origins are traced from. You can download the code from [my GitHub](https://github.com/edoyango/GraSPH). See the README for brief instructions on how to get started, or the [Getting Started](/docs/docs/start) page.

## About the Author
See my [about page](https://ed-yang.com/about). I worked on SPH as a part of my time with MCGLAB as an undergrad + masters student between 2017-2020. But now I work in health science research at WEHI as a Research Computing Engineer.