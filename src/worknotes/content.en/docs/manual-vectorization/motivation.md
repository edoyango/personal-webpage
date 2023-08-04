---
title: Motivation
weight: 1
---

SPH is a continuum particle method that is often used for simulations. Typically, the most time consuming part of codes
that aim to perform SPH simulations, is finding the pairs of SPH particles that are within a fixed-cutoff of each other
(the pair-search step from herein), and calculating the contribution to particles' motion, due to its corresponding
pair (the force calculation sweep step from herein). These steps can be combined together when organising the code,
but it's useful to seperate them when needing to re-use the pair list.

The pattern that the force calculation sweep looks like, can be illustrated with code (in C++) to calculate the
rate-of-change of density due to the continuity equation (`drho/dt = div(v)`):

```c {style=tango,linenos=false}
for (int k = 0; k < num_pairs; ++k) {
    i = pair_i[k];
    j = pair_j[k];

    dvx = vx[i] - vx[j];
    dvy = vy[i] - vy[j];
    dvz = vz[i] - vz[j];

    vcc = mass*(dvx*dwdx[k] + dvy*dwdy[k] + dvz*dwdz[k]);

    drhodt[i] += vcc;
    drhodt[j] += vcc;
}
```

where `v_` stores the velocity of each particle in `x`, `y`, and `z` directions; `drho_dt` stores the rate of change of 
density; and `dwd_x` stores the spatial gradient of the kernel function in the `x`, `y`, and `z` directions.

This pattern is not automatically vectorized by compilers, due to the non-sequential memory access pattern in the
algorithm. Consequently, I wanted to invetigate manually vectorizing this loop, to see if I could reduce run times.

x86 CPUs have a long list of assembly instructions that can be accessed in C/C++ via [Intel's SIMD Intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).

## Platform

Everything on this page is run on the following platform:


* CPU: Xeon Gold 6342 CPU (Icelake)
* OS: CentOS 7
* Compiler: `g++` v10.3.0

Code is benchmarked with the [Google microbenchmark framework](https://github.com/google/benchmark/)