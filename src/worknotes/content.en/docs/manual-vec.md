---
title: Exploring Manual Vectorization For SPH
weight: 1
---

# Exploring Manual Vectorization For SPH

SPH is a continuum particle method that is often used for simulations. Typically, the most time consuming part of codes
that aim to perform SPH simulations, is finding the pairs of SPH particles that are within a fixed-cutoff of each other
(the pair-search step from herein), and calculating the contribution to particles' motion, due to its corresponding
pair (the force calculation sweep step from herein). These steps can be combined together when organising the code,
but it's useful to seperate them when needing to re-use the pair list.

The pattern that the force calculation sweep looks like, can be illustrated with code (in C++) to calculate the
rate-of-change of density due to the continuity equation (`drho/dt = div(v)`):

```c
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

## Hardware

Everything on this page is run on the following platform:

* Cluster with
    * Xeon Gold 6342 CPU (Icelake)
    * CentOS 7

The `g++` compiler is used here (version 9.4.0).

## Basic Vectorization

x86 CPUs have a long list of assembly instructions that can be accessed in C/C++ via [Intel's SIMD Intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html).
A common place to start, is to manually vectorize the addition of two arrays, and storing the result in a third array
(from herein, ABC addition):

```cpp
void abc_base_sgl(const int nelem, float* a, float* b, float*) {
    for (int i = 0; i < nelem; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

### Vectorizing the ABC function with 128-bit vectors (SSE instructions)

This function, manually vectorized, looks like:

```cpp
// 128bit width (SSE)
void abc_128_sgl(const int nelem, float* a, float* b, float* c) {
    for (int i = 0; i < nelem; i += 4) {
        __m128 v1 = _mm_loadu_ps(&a[i]);
        __m128 v2 = _mm_loadu_ps(&b[i]);
        __m128 res = _mm_add_ps(v1, v2);
        _mm_storeu_ps(&c[i], res);
    }
}
```

Note that this requires `#include <immintrin.h>` to make use of the types and functions.

The `_mm_loadu_ps` function name can be broken down as:

* `mm`: MultiMedia (Intel instructions intended for media). `mm` alone, usually indicates 128-bit width vectors.
* `loadu`: Load data into SIMD vectors. This is specifically for "unaligned" data. More on this later.
* `ps`: Packed Single precision data.

Hence, this function loads single precision floats from a `float` pointer/array, to a packed SIMD single precision 
vector. This load operation takes the supplied pointer, and loads the next 4 elements following the address. It knows to
load 4 elements, as single precision data is 32 bits. Loading 4 elements at a time, is the reason why the loop counter
is incremented by 4 every iteration.

`_mm_add_ps` is part of the same MultiMedia extensions, and works with single precision floats, but performs an
add operation on two 128-bit vectors. The add operation will operate on each element on each vector.

`_mm_storeu_ps` takes the `res` vector, and places back at the position specified by the supplied pointer (in
this case, `&c[i]`).

`__m128` is the 128-bit wide vector for single precision floating point numbers.

The double precision version looks similar:

```cpp
// 128bit width (SSE)
void abc_128_dbl(const int nelem, double* a, double* b, double* c) {
    for (int i = 0; i < nelem; i += 2) {
        __m128d v1 = _mm_loadu_pd(&a[i]);
        __m128d v2 = _mm_loadu_pd(&b[i]);
        __m128d res = _mm_add_pd(v1, v2);
        _mm_storeu_pd(&c[i], res);
    }
}
```

There are 3 main differences in this version. Two of them are that `ps` is replaced by `pd` everywhere, and `d` is added
to the end of the `__m128` types. These ensure that the equivalent functions and types for double precision floating
point numbers are used. The 3rd change is that the loop counter is incremented by two instead of 4 every iteration. This
is because double precision floating point data is stored using 64 bits, so only two of them can fit into an 128-bit
SIMD vector.

### Vectorizing the ABC function with 256-bit vectors (AVX instructions)

The AVX instruction set adds new instructions to the collection of SIMD instructions, as well as allowing for 256-bit
vectors. For this example, the code looks pretty similar. The single precision code looks like:

```cpp
// 256bit width (AVX)
void abc_256_sgl(const int nelem, float* a, float* b, float* c) {
    for (int i = 0; i < nelem; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&a[i]);
        __m256 v2 = _mm256_loadu_ps(&b[i]);
        __m256 res = _mm256_add_ps(v1, v2);
        _mm256_storeu_ps(&c[i], res);
    }
}
```

Where `mm` has been changed to `mm256`, to indicate these instructions are part of the 256-bit wide vector instructions,
and the incrememnt has been doubled from 4 to 8, due to the doubling in width of the vectors. The same changes are true
for the double precision version:

```cpp
// 256bit width (AVX)
void abc_256_dbl(const int nelem, double* a, double* b, double* c) {
    for (int i = 0; i < nelem; i += 4) {
        __m256d v1 = _mm256_loadu_pd(&a[i]);
        __m256d v2 = _mm256_loadu_pd(&b[i]);
        __m256d res = _mm256_add_pd(v1, v2);
        _mm256_storeu_pd(&c[i], res);
    }
}
```

### Performance of the SINGLE PRECISION manually vectorized ABC function with 128-bit vectors (SSE instructions)

Here, I make use of [Google's benchmark tool](https://github.com/google/benchmark). Each array, `a`, `b`, and `c`, are
between 4,096 and 16,777,216 elements long. The program compilation command:

```bash
g++ -o abc.x abc.cpp -mavx2 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
```
First, the single precision data and using the benchmark comparison tool:

```
Comparing abc_base_sgl (from ./abc.x) to abc_128_sgl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_128_sgl]/4096        -0.5372       -0.5372         12619          5841         12590          5827
[abc_base_sgl vs. abc_128_sgl]/32768       -0.5164       -0.5164        100548         48628        100314         48514
[abc_base_sgl vs. abc_128_sgl]/262144      -0.5091       -0.5093        802916        394148        801042        393099
[abc_base_sgl vs. abc_128_sgl]/2097152     -0.5027       -0.5027       6484863       3225174       6469666       3217643
[abc_base_sgl vs. abc_128_sgl]/16777216    -0.5172       -0.5172      54029226      26084896      53902545      26023999
OVERALL_GEOMEAN                            -0.5166       -0.5167             0             0             0             0
```

In this results, the "Old" is the non-manually vectorized loop, and the "New" is the vectorized loop. The comparison
shows at least (1-0.5166)^-1 = 2x speedup by manually vectorization for all array sizes tested here! To compare the manually
vectorized code to compiler's automatic vectorization, we can compile the code again with:

```bash
g++ -o abc.x abc.cpp -msse -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
```

which
* replaces `-mavx2` with `-msse`, enabling only 128-bit wide vectors and SSE instructions,
* add `-O2`, which enables level 2 compiler optimisations, and
* add `-ftree-vectorize`, which enables automatic vectorization of loops.

These options will make the code faster through compiler optimisations, which should benefit both the original base code
and the manually vectorized code, but they will also enable automatic vectorization on the original loop only.
Comparing the results performance of the now automatically vectorized function (`abc_base_sgl`), with the manually
vectorized function (`abc_128_sgl`), I get:

```
Comparing abc_base_sgl (from ./abc.x) to abc_128_sgl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_128_sgl]/4096        -0.0001       +0.0002           630           630           629           629
[abc_base_sgl vs. abc_128_sgl]/32768       -0.0021       -0.0021          6345          6332          6330          6317
[abc_base_sgl vs. abc_128_sgl]/262144      +0.0162       +0.0163        104701        106392        104437        106144
[abc_base_sgl vs. abc_128_sgl]/2097152     +0.0023       +0.0027        908528        910590        906019        908466
[abc_base_sgl vs. abc_128_sgl]/16777216    +0.0225       +0.0230      14303786      14625654      14263831      14591516
OVERALL_GEOMEAN                            +0.0077       +0.0080             0             0             0             0
```

which shows that at worst, the manually vectorized code is approx. 2% slower than the automatically vectorized loop.

### Performance of the SINGLE PRECISION manually vectorized ABC function with 256-bit vectors (AVX instructions)

Using the same test method as the 128-bit version, using no compiler optimisations, I get:

```
Comparing abc_base_sgl (from ./abc.x) to abc_256_sgl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------ -----------------------------------------------------------------------------
[abc_base_sgl vs. abc_256_sgl]/4096        -0.7394       -0.7395         12390          3229         12361          3221
[abc_base_sgl vs. abc_256_sgl]/32768       -0.7276       -0.7277         99812         27192         99579         27118
[abc_base_sgl vs. abc_256_sgl]/262144      -0.7242       -0.7243        802667        221387        800794        220789
[abc_base_sgl vs. abc_256_sgl]/2097152     -0.7151       -0.7152       6437792       1834148       6422769       1829137
[abc_base_sgl vs. abc_256_sgl]/16777216    -0.6711       -0.6713      53525278      17602131      53399838      17553344
OVERALL_GEOMEAN                            -0.7164       -0.7165             0             0             0             0
```

Which shows a further speedup over the 128-bit version. Over the base version, the speedup is approximately 
(1-0.7164)^-1 = 3.5x. You may notice that the speedup obtained decreases slightly with the size of test, which I suspect
is due to the size of cache being used on the CPU. The 4,096 element test (4096 elements * 4 bytes * 2 arrays = ~33kB), 
is enough to sit within L1 cache. The next test moves to L2 cache as 32,768 elements * 4 bytes * 2 arrays = ~262kB moves
to L2 unified cache. The next test (2,097,152 elements * 4 bytes * 2 arrays = ~16.8MB) must move to L3 unified cache, 
and the final test (16, 777, 216 * 4 bytes * 2 arrays = ~134MB) must go to RAM. This behaviour isn't observed with 
128-bit width registers, as the smaller vectors means less data needing to be fetched per transaction.

The minimum speedup is ~3x, and the best is ~3.8x - neither of which is 2x that observed with the 128-bit vectors.

| Memory type | Capacity | Capacity (single precision elements) |
| ------------- | -------- | --- |
| L1 cache   | 48KiB | 12,288 |
| L2 cache (unified) | 1280KiB | 327,680 |
| L3 cache (unified) | 36864KiB | 9,437,184 |
| RAM | 500GiB | lots |

Confirming similarity of manual vectorization to automatic vectorization:

```bash
g++ -o abc.x abc.cpp -mavx -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2 -ftree-vectorize
```

```
Comparing abc_base_sgl (from ./abc.x) to abc_256_sgl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_256_sgl]/4096        +0.0010       +0.0012           633           633           631           632
[abc_base_sgl vs. abc_256_sgl]/32768       +0.0012       +0.0016          6097          6104          6080          6090
[abc_base_sgl vs. abc_256_sgl]/262144      +0.0009       +0.0013        105703        105800        105417        105551
[abc_base_sgl vs. abc_256_sgl]/2097152     -0.0020       -0.0016        910727        908903        908205        906771
[abc_base_sgl vs. abc_256_sgl]/16777216    +0.0281       +0.0286      14356772      14760474      14316874      14725833
OVERALL_GEOMEAN                            +0.0058       +0.0062             0             0             0             0
```

I see a similarly small variation in performance between the the automatically vectorized base function and the
manually vectorized version.

The code shown here achieves far less than ideal speedup e.g., ~2x achieved for 128-bit vectors, when 4x is ideal; and
~3.5x achieved for 256-bit vectors, when 8x is ideal. I have yet to understand the details of the mechanism behind this,
but I think it is due to memory latency and bandwidth.

### Performance of the DOUBLE PRECISION manually vectorized ABC function with 128-bit vectors (SSE instructions)

Using the compilation command with no compiler options:

```bash
g++ -o abc.x abc.cpp -msse -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
```

We get the results between the non-vectorized base function and the manually vectorized function:

```
Comparing abc_base_dbl (from ./abc.x) to abc_128_dbl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_dbl vs. abc_128_dbl]/4096        -0.0514       -0.0516         12544         11899         12515         11869
[abc_base_dbl vs. abc_128_dbl]/32768       -0.0230       -0.0230        101078         98749        100842         98518
[abc_base_dbl vs. abc_128_dbl]/262144      -0.0429       -0.0431        812827        777929        810929        775969
[abc_base_dbl vs. abc_128_dbl]/2097152     -0.0112       -0.0116       6743898       6668566       6728128       6650022
[abc_base_dbl vs. abc_128_dbl]/16777216    -0.0746       -0.0748      56644964      52418793      56512720      52284069
OVERALL_GEOMEAN                            -0.0409       -0.0411             0             0             0             0
```

Which shows very modest improvement of approx 4%. Furthermore, only the largest test is determined by the comparison
tool to be statistically significant. This is not surprising, given the results from the single precision manually
vectorized code. The single precision code obtained a speedup of approximately 2x, despite being able to combine 4
add operations into one. However, the double precision code can only combine 2 add operations, which is apparently
insufficient to offset the memory latency/bandwidth.

Checking the manually vectorized code against automatic vectorization:

```bash
g++ -o abc.x abc.cpp -msse -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2 -ftree-vectorize
```

```
Comparing abc_base_dbl (from ./abc.x) to abc_128_dbl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_dbl vs. abc_128_dbl]/4096        +0.0233       +0.0237          1532          1568          1528          1565
[abc_base_dbl vs. abc_128_dbl]/32768       -0.0313       -0.0309         13050         12641         13014         12612
[abc_base_dbl vs. abc_128_dbl]/262144      +0.0126       +0.0128        212077        214751        211534        214245
[abc_base_dbl vs. abc_128_dbl]/2097152     -0.0357       -0.0353       2723765       2626471       2716242       2620327
[abc_base_dbl vs. abc_128_dbl]/16777216    -0.0097       -0.0095      30242092      29949972      30165349      29880070
OVERALL_GEOMEAN                            -0.0084       -0.0081             0             0             0             0
```

Which again, shows minimal difference between the manually and automatically vectorized code.

### Performance of the DOUBLE PRECISION manually vectorized ABC function with 256-bit vectors (AVX instructions)

Comparison results:

```
Comparing abc_base_dbl (from ./abc.x) to abc_256_dbl (from ./abc.x)
Benchmark                                     Time           CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------
[abc_base_dbl vs. abc_256_dbl]/4096        -0.4700       -0.4699         12537          6644         12506          6629
[abc_base_dbl vs. abc_256_dbl]/32768       -0.4609       -0.4606        101533         54741        101257         54613
[abc_base_dbl vs. abc_256_dbl]/262144      -0.4548       -0.4546        814199        443934        811995        442898
[abc_base_dbl vs. abc_256_dbl]/2097152     -0.3759       -0.3758       6699499       4180956       6682306       4171199
[abc_base_dbl vs. abc_256_dbl]/16777216    -0.3386       -0.3385      56663574      37476558      56520373      37388836
OVERALL_GEOMEAN                            -0.4224       -0.4222             0             0             0             0
```

Like the single precision 256-bit results, we can see a decreasing speedup as the tests increase in size. The best
speedup achieved is ~1.9x and slowest is ~1.5x. 

