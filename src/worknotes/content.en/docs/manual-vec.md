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

```cpp {style=tango,linenos=false}
void abc_base_sgl(const int nelem, float* a, float* b, float*) {
    for (int i = 0; i < nelem; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

In the benchmarking code, the arrays are created with `new`.

### Vectorizing the ABC function with 128-bit vectors (SSE instructions)

This function, manually vectorized, looks like:

```cpp {style=tango,linenos=false}
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

```cpp {style=tango,linenos=false}
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

```cpp {style=tango,linenos=false}
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

```cpp {style=tango,linenos=false}
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

```bash {style=tango,linenos=false}
g++ -o abc.x abc.cpp -mavx2 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
```
First, the single precision data and using the benchmark comparison tool:

``` {style=tango,linenos=false}
Comparing abc_base_sgl (from ./abc.x) to abc_128_sgl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_128_sgl]/4096        -0.5372    -0.5372       12619        5841       12590        5827
[abc_base_sgl vs. abc_128_sgl]/32768       -0.5164    -0.5164      100548       48628      100314       48514
[abc_base_sgl vs. abc_128_sgl]/262144      -0.5091    -0.5093      802916      394148      801042      393099
[abc_base_sgl vs. abc_128_sgl]/2097152     -0.5027    -0.5027     6484863     3225174     6469666     3217643
[abc_base_sgl vs. abc_128_sgl]/16777216    -0.5172    -0.5172    54029226    26084896    53902545    26023999
OVERALL_GEOMEAN                            -0.5166    -0.5167           0           0           0           0
```

In this results, the "Old" is the non-manually vectorized loop, and the "New" is the vectorized loop. The comparison
shows at least (1-0.5166)^-1 = 2x speedup by manually vectorization for all array sizes tested here! To compare the manually
vectorized code to compiler's automatic vectorization, we can compile the code again with:

```bash {style=tango,linenos=false}
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

``` {style=tango,linenos=false}
Comparing abc_base_sgl (from ./abc.x) to abc_128_sgl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_128_sgl]/4096        -0.0001    +0.0002         630         630         629         629
[abc_base_sgl vs. abc_128_sgl]/32768       -0.0021    -0.0021        6345        6332        6330        6317
[abc_base_sgl vs. abc_128_sgl]/262144      +0.0162    +0.0163      104701      106392      104437      106144
[abc_base_sgl vs. abc_128_sgl]/2097152     +0.0023    +0.0027      908528      910590      906019      908466
[abc_base_sgl vs. abc_128_sgl]/16777216    +0.0225    +0.0230    14303786    14625654    14263831    14591516
OVERALL_GEOMEAN                            +0.0077    +0.0080           0           0           0           0
```

which shows that at worst, the manually vectorized code is approx. 2% slower than the automatically vectorized loop and
none of the differences are statistically significant.

### Performance of the SINGLE PRECISION manually vectorized ABC function with 256-bit vectors (AVX instructions)

Using the same test method as the 128-bit version, using no compiler optimisations, I get:

``` {style=tango,linenos=false}
Comparing abc_base_sgl (from ./abc.x) to abc_256_sgl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
------------------------------------------ ------------------------------------------------------------------
[abc_base_sgl vs. abc_256_sgl]/4096        -0.7394    -0.7395       12390        3229       12361        3221
[abc_base_sgl vs. abc_256_sgl]/32768       -0.7276    -0.7277       99812       27192       99579       27118
[abc_base_sgl vs. abc_256_sgl]/262144      -0.7242    -0.7243      802667      221387      800794      220789
[abc_base_sgl vs. abc_256_sgl]/2097152     -0.7151    -0.7152     6437792     1834148     6422769     1829137
[abc_base_sgl vs. abc_256_sgl]/16777216    -0.6711    -0.6713    53525278    17602131    53399838    17553344
OVERALL_GEOMEAN                            -0.7164    -0.7165           0           0           0           0
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

```bash {style=tango,linenos=false}
g++ -o abc.x abc.cpp -mavx -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2 -ftree-vectorize
```

``` {style=tango,linenos=false}
Comparing abc_base_sgl (from ./abc.x) to abc_256_sgl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------
[abc_base_sgl vs. abc_256_sgl]/4096        +0.0010    +0.0012         633         633         631         632
[abc_base_sgl vs. abc_256_sgl]/32768       +0.0012    +0.0016        6097        6104        6080        6090
[abc_base_sgl vs. abc_256_sgl]/262144      +0.0009    +0.0013      105703      105800      105417      105551
[abc_base_sgl vs. abc_256_sgl]/2097152     -0.0020    -0.0016      910727      908903      908205      906771
[abc_base_sgl vs. abc_256_sgl]/16777216    +0.0281    +0.0286    14356772    14760474    14316874    14725833
OVERALL_GEOMEAN                            +0.0058    +0.0062           0           0           0           0
```

I see a similarly small variation in performance between the the automatically vectorized base function and the
manually vectorized version.

The code shown here achieves far less than ideal speedup e.g., ~2x achieved for 128-bit vectors, when 4x is ideal; and
~3.5x achieved for 256-bit vectors, when 8x is ideal. I have yet to understand the details of the mechanism behind this,
but I think there are many factors that go into this, like the ordering of instructions or whether the data is
prefetched.

### Performance of the DOUBLE PRECISION manually vectorized ABC function with 128-bit vectors (SSE instructions)

Using the compilation command with no compiler options:

```bash {style=tango,linenos=false}
g++ -o abc.x abc.cpp -msse -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
```

We get the results between the non-vectorized base function and the manually vectorized function:

``` {style=tango,linenos=false}
Comparing abc_base_dbl (from ./abc.x) to abc_128_dbl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
---------------------------------------------------------------------------------------------------- --------
[abc_base_dbl vs. abc_128_dbl]/4096        -0.0514    -0.0516       12544       11899       12515       11869
[abc_base_dbl vs. abc_128_dbl]/32768       -0.0230    -0.0230      101078       98749      100842       98518
[abc_base_dbl vs. abc_128_dbl]/262144      -0.0429    -0.0431      812827      777929      810929      775969
[abc_base_dbl vs. abc_128_dbl]/2097152     -0.0112    -0.0116     6743898     6668566     6728128     6650022
[abc_base_dbl vs. abc_128_dbl]/16777216    -0.0746    -0.0748    56644964    52418793    56512720    52284069
OVERALL_GEOMEAN                            -0.0409    -0.0411           0           0           0           0
```

Which shows very modest improvement of approx 4%. Furthermore, only the largest test is determined by the comparison
tool to be statistically significant. This is not surprising, given the results from the single precision manually
vectorized code. The single precision code obtained a speedup of approximately 2x, despite being able to combine 4
add operations into one. However, the double precision code can only combine 2 add operations, which is apparently
insufficient to offset the memory latency/bandwidth.

Checking the manually vectorized code against automatic vectorization:

```bash {style=tango,linenos=false}
g++ -o abc.x abc.cpp -msse -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2 -ftree-vectorize
```

``` {style=tango,linenos=false}
Comparing abc_base_dbl (from ./abc.x) to abc_128_dbl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------
[abc_base_dbl vs. abc_128_dbl]/4096        +0.0233    +0.0237        1532        1568        1528        1565
[abc_base_dbl vs. abc_128_dbl]/32768       -0.0313    -0.0309       13050       12641       13014       12612
[abc_base_dbl vs. abc_128_dbl]/262144      +0.0126    +0.0128      212077      214751      211534      214245
[abc_base_dbl vs. abc_128_dbl]/2097152     -0.0357    -0.0353     2723765     2626471     2716242     2620327
[abc_base_dbl vs. abc_128_dbl]/16777216    -0.0097    -0.0095    30242092    29949972    30165349    29880070
OVERALL_GEOMEAN                            -0.0084    -0.0081           0           0           0           0
```

Which again, shows minimal difference between the manually and automatically vectorized code.

### Performance of the DOUBLE PRECISION manually vectorized ABC function with 256-bit vectors (AVX instructions)

Comparison results:

``` {style=tango,linenos=false}
Comparing abc_base_dbl (from ./abc.x) to abc_256_dbl (from ./abc.x)
Benchmark                                     Time        CPU    Time Old    Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------
[abc_base_dbl vs. abc_256_dbl]/4096        -0.4700    -0.4699       12537        6644       12506        6629
[abc_base_dbl vs. abc_256_dbl]/32768       -0.4609    -0.4606      101533       54741      101257       54613
[abc_base_dbl vs. abc_256_dbl]/262144      -0.4548    -0.4546      814199      443934      811995      442898
[abc_base_dbl vs. abc_256_dbl]/2097152     -0.3759    -0.3758     6699499     4180956     6682306     4171199
[abc_base_dbl vs. abc_256_dbl]/16777216    -0.3386    -0.3385    56663574    37476558    56520373    37388836
OVERALL_GEOMEAN                            -0.4224    -0.4222           0           0           0           0
```

Like the single precision 256-bit results, we can see a decreasing speedup as the tests increase in size. The best
speedup achieved is ~1.9x and slowest is ~1.5x. 

## Improving SIMD Performance By Aligning Memory

When explaining the `_mm_storeu_px` and `_mm_loadu_px` functions, I mentioned that these, respectively, store and load
data to/from "aligned" memory. When using the "classic" `malloc` or `new` functions to allocate memory for arrays, these
arrays may be "unaligned". This means its integer address is the computer's memory, is not divisible by its size (in 
bytes). I'm not sure of the details of why, but this means that accessing this memory can be more expensive. It may also
negatively affect caching behaviour.

Because of this unalignment, we must use the `storeu` and `loadu` functions so that they can properly store and load
data to/from locations that are unaligned. However, there are `store` and `load` functions that can be used on aligned
data, which can be more performant.

### Migrating the existing code to use aligned data

The existing code allocates memory for our arrays using:

```cpp {style=tango,linenos=false}
float *a = new float[nelem], *b = new float[nelem], *c = new float[nelem];
```

for single precision data (replace `float` with `double` for the double precision version). To ensure the arrays are
aligned, you must use the `std::aligned_alloc` function, which comes with the `cstdlib` header. Our allocation becomes:

```cpp {style=tango,linenos=false}
const size_t size = sizeof(float) * nelem, alignment = 32;
float *a = static_cast<float*>(std::aligned_alloc(alignment, size));
float *b = static_cast<float*>(std::aligned_alloc(alignment, size));
float *c = static_cast<float*>(std::aligned_alloc(alignment, size));
```

Replace `float` with double and `32` with `64`, and you have the double precision version. These arrays also have to be
deallocated with `free` instead of `delete`.

Then, you need to replace all instances of `storeu` and `loadu` with `store` and `load`.

### Performance of aligned vs unaligned data for SIMD instructions

The results below compare run times between the unaligned and aligned versions of each code for each of the datatypes
and vector sizes.

#### Aligned memory: single precision 128-bit

``` {style=tango,linenos=false}
Comparing abc_128_sgl (from ./abc.x) to abc_128_sgl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_128_sgl vs. abc_128_sgl_aligned]/4096        -0.0639    -0.0635        5807        5436        5792        5424
[abc_128_sgl vs. abc_128_sgl_aligned]/32768       -0.0965    -0.0963       50260       45408       50129       45302
[abc_128_sgl vs. abc_128_sgl_aligned]/262144      -0.0604    -0.0600      390059      366517      389004      365662
[abc_128_sgl vs. abc_128_sgl_aligned]/2097152     -0.0609    -0.0605     3161569     2968982     3152943     2962053
[abc_128_sgl vs. abc_128_sgl_aligned]/16777216    +0.0281    +0.0285    25573480    26291967    25502874    26230381
OVERALL_GEOMEAN                                   -0.0247    -0.0244           0           0           0           0
```

#### Aligned memory: single precision 256-bit

``` {style=tango,linenos=false}
Comparing abc_256_sgl (from ./abc.x) to abc_256_sgl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_256_sgl vs. abc_256_sgl_aligned]/4096        -0.1635    -0.1634        3312        2770        3303        2764
[abc_256_sgl vs. abc_256_sgl_aligned]/32768       -0.1261    -0.1260       27567       24090       27497       24034
[abc_256_sgl vs. abc_256_sgl_aligned]/262144      -0.1344    -0.1342      225293      195017      224719      194562
[abc_256_sgl vs. abc_256_sgl_aligned]/2097152     -0.1191    -0.1191     1833080     1614842     1828802     1611060
[abc_256_sgl vs. abc_256_sgl_aligned]/16777216    -0.0395    -0.0392    18248268    17528127    18201311    17487204
OVERALL_GEOMEAN                                   -0.0699    -0.0697           0           0           0           0
```

#### Aligned memory: double precision 128-bit

``` {style=tango,linenos=false}
Comparing abc_128_dbl (from ./abc.x) to abc_128_dbl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_128_dbl vs. abc_128_dbl_aligned]/4096        -0.0647    -0.0647       12000       11224       11972       11198
[abc_128_dbl vs. abc_128_dbl_aligned]/32768       -0.0703    -0.0705       97467       90611       97240       90383
[abc_128_dbl vs. abc_128_dbl_aligned]/262144      -0.0627    -0.0631      780496      731551      778675      729556
[abc_128_dbl vs. abc_128_dbl_aligned]/2097152     -0.0579    -0.0583     6742195     6351631     6726466     6334012
[abc_128_dbl vs. abc_128_dbl_aligned]/16777216    +0.0270    +0.0270    52430133    53845667    52307789    53719950
OVERALL_GEOMEAN                                   -0.0234    -0.0236           0           0           0           0
```

#### Aligned memory: double precision 256-bit

``` {style=tango,linenos=false}
Comparing abc_256_dbl (from ./abc.x) to abc_256_dbl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_256_dbl vs. abc_256_dbl_aligned]/4096        -0.0969    -0.0968        6658        6012        6641        5998
[abc_256_dbl vs. abc_256_dbl_aligned]/32768       -0.0931    -0.0927       54958       49842       54809       49726
[abc_256_dbl vs. abc_256_dbl_aligned]/262144      -0.1182    -0.1179      448770      395737      447559      394813
[abc_256_dbl vs. abc_256_dbl_aligned]/2097152     -0.0735    -0.0735     4322564     4004882     4312476     3995495
[abc_256_dbl vs. abc_256_dbl_aligned]/16777216    -0.0479    -0.0475    38081022    36256189    37975969    36171577
OVERALL_GEOMEAN                                   -0.0436    -0.0434           0           0           0           0
```

### Effect of `std::aligned_malloc`

Almost all test sizes perform better than the unaligned counterparts. The only exceptions are the largest tests for the
128-bit vector code - these seem to perform almost 3% worse. This probably is related to the fact that the `a` and `b`
arrays are entirely on RAM. This is also consistent with the 256-bit vector tests where the largest tests perform the
worst when compared to their unaligned conterparts.

The 256-bit vector codes have the most increased performance - espcially for the tests that can fit in L1 and L2 cache.
For example, the biggest improvement is seen in the single precision 128-bit vector code, for the problem size of 4,096
elements. The 16% reduction in run time makes it ~2.3 faster than its corresponding base code (up from ~2x).

Most of the other tests gain at least 4% from its unaligned counterpart.

## Adding compiler optimisations

Compiling the code with `-O2` and rerunning the tests with aligned memory:

#### Aligned memory + lvl 2 optimisations: single precision 128-bit
 
``` {style=tango,linenos=false}
Comparing abc_128_sgl (from ./abc.x) to abc_128_sgl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_128_sgl vs. abc_128_sgl_aligned]/4096        +0.0099    +0.0103         630         636         628         635
[abc_128_sgl vs. abc_128_sgl_aligned]/32768       +0.0474    +0.0476        6316        6616        6300        6600
[abc_128_sgl vs. abc_128_sgl_aligned]/262144      +0.0001    +0.0001      105151      105166      104905      104919
[abc_128_sgl vs. abc_128_sgl_aligned]/2097152     -0.0035    -0.0033      907374      904171      905084      902062
[abc_128_sgl vs. abc_128_sgl_aligned]/16777216    +0.0273    +0.0275    14132072    14517526    14096031    14483670
OVERALL_GEOMEAN                                   +0.0190    +0.0192           0           0           0           0
```

#### Aligned memory + lvl 2 optimisations: single precision 256-bit

``` {style=tango,linenos=false}
Comparing abc_256_sgl (from ./abc.x) to abc_256_sgl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_256_sgl vs. abc_256_sgl_aligned]/4096        -0.4919    -0.4921         632         321         631         320
[abc_256_sgl vs. abc_256_sgl_aligned]/32768       -0.2773    -0.2774        6209        4488        6195        4476
[abc_256_sgl vs. abc_256_sgl_aligned]/262144      +0.0066    +0.0064      104733      105426      104488      105162
[abc_256_sgl vs. abc_256_sgl_aligned]/2097152     +0.0063    +0.0059      908430      914187      906310      911683
[abc_256_sgl vs. abc_256_sgl_aligned]/16777216    +0.0078    +0.0073    14697929    14812229    14663625    14771162
OVERALL_GEOMEAN                                   -0.0882    -0.0885           0           0           0           0
```

#### Aligned memory + lvl 2 optimisations: double precision 128-bit

``` {style=tango,linenos=false}
Comparing abc_128_dbl (from ./abc.x) to abc_128_dbl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_128_dbl vs. abc_128_dbl_aligned]/4096        +0.0612    +0.0617        1567        1663        1563        1660
[abc_128_dbl vs. abc_128_dbl_aligned]/32768       +0.0766    +0.0770       12643       13611       12608       13579
[abc_128_dbl vs. abc_128_dbl_aligned]/262144      -0.0037    -0.0033      211424      210649      210844      210148
[abc_128_dbl vs. abc_128_dbl_aligned]/2097152     +0.0810    +0.0815     2597400     2807846     2590158     2801291
[abc_128_dbl vs. abc_128_dbl_aligned]/16777216    +0.0345    +0.0299    30125298    31163848    30049063    30948143
OVERALL_GEOMEAN                                   +0.0411    +0.0405           0           0           0           0
```

#### Aligned memory + lvl 2 optimisations: double precision 256-bit

``` {style=tango,linenos=false}
Comparing abc_256_dbl (from ./abc.x) to abc_256_dbl_aligned (from ./abc.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[abc_256_dbl vs. abc_256_dbl_aligned]/4096        -0.2838    -0.2839        1558        1116        1555        1113
[abc_256_dbl vs. abc_256_dbl_aligned]/32768       -0.2637    -0.2640       12501        9205       12472        9180
[abc_256_dbl vs. abc_256_dbl_aligned]/262144      -0.0077    -0.0081      213428      211785      212930      211203
[abc_256_dbl vs. abc_256_dbl_aligned]/2097152     +0.0229    +0.0224     2699745     2761498     2693422     2753700
[abc_256_dbl vs. abc_256_dbl_aligned]/16777216    +0.0322    +0.0320    30455778    31437745    30384702    31356069
OVERALL_GEOMEAN                                   -0.0423    -0.0427           0           0           0           0
```

### Effects of adding compiler optimisation

Quite interestingly, the `-O2` compiler optimisations amplify the performance of SIMD instructions for
the tests that fit within L1 and L2 cache. The 128-bit tests are brought much closer to ideal performance for those
small problem sizes. The 128-bit tests do not seem to perform very differently with or without aligned memory. Aligning 
the data allocations even seem to cause the 128-bit tests to  perform worse by a little bit! In contrast, the 256-bit
vector instructions apparently require aligned memory to benefit form `-O2` optimisations in terms of speedup.

| Test ID | Best speedup over base |
| --- | --- |
| `abc_128_sgl` | 4.18x |
| `abc_128_sgl_aligned` | 3.97x |
| `abc_256_sgl` | 3.97x |
| `abc_256_sgl_aligned` | 5.84x |
| `abc_128_dbl` | 1.78x |
| `abc_128_dbl_aligned` | 1.74x | 
| `abc_256_dbl` | 1.92x |
| `abc_256_dbl_aligned` | 2.6x |

Note that the non-vectorized code also benefits from using aligned data, so the best speedups shown above, comparing 
like-for-like i.e., aligned vectorized, with aligned non-vectorized; and non-aligned vectorized, with non-aligned 
non-vectorized.

## Vectorizing A Simple Pair Sweep

One way to perform the force calculation sweep in SPH, is to use a double loop like:

```cpp
for (int i = 0; i < nelem-1; ++i) {
    for (int j = i < nelem; ++j) {
        // compare particle i and j
        // calculate forces
    }
}
```

This is generally not a preferred way to do the force calculations, as the number of operations in this algorithm scales
with the square of the number of particles. But it's useful for playing around with manual vectorization.

The base case we will use is this:

```cpp
void sweep_base_sgl(const int nelem, float* a, float* b) {
    int k = 0
    for (int i = 0; i < nelem-1; i += 8) {
        for (int j = i; j < nelem; ++j) {
            tmp = a[i]-a[j];
            b[i] += tmp;
            b[j] -= tmp;
        }
    }
}
```

`i` is incremented by 8 to simplify vectorization. Swap out `float` with `double`, to get the double precision version.

Important to note here, is that this loop cannot automatically be vectorized by the compiler. For example, if we compile
our code with `-O3 -fopt-info-vec-missed`, `-O3` will try to vectorize the loop, and the other option will let you know
when it cannot vectorize a given loop. At compilation, I get some important info in the output:

```
sweep.cpp:51:27: missed: couldn't vectorize loop
sweep.cpp:51:27: missed: not vectorized: multiple nested loops.
sweep.cpp:51:27: missed: couldn't vectorize loop
sweep.cpp:51:27: missed: not vectorized: inner-loop count not invariant.
sweep.cpp:52:31: missed: couldn't vectorize loop
```

Which explains that the compiler missed the vectorization of these loops because:
* the loops are nested, and
* the inner loop looping variable, `j`, is not constant for every loop.
Hence, this is a meaningful candidate to vectorize.

###  Vectorizing the direct pair sweep with 128-bit vectors (SSE instructions)

This attempt to vectorize the loop will turn the outer loop into a vector of all the same variable, and then the inner
loop will load multiple sequential elements to operate the outer-loop vector. This looks like:

```cpp
void sweep_128_sgl(const int nelem, float* a, float* b) {
    float tmp[4];
    for (int i = 0; i < nelem-1; i += 8) {
        __m128 vi = _mm_set1_ps(a[i]);
        for (int j = i; j < nelem; j += 4) {
            __m128 vj = _mm_loadu_ps(&a[j]);
            __m128 vdiff = _mm_sub_ps(vi, vj);
            __m128 vbj = _mm_loadu_ps(&b[j]);
            vbj = _mm_sub_ps(vbj, vdiff);
            _mm_storeu_ps(&b[j], vbj);
            _mm_storeu_ps(tmp, vdiff);
            b[i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];
        }
    }
}
```

Now this code looks more complicated than the ABC addition example used earlier. 
1. `__m128 vi = _mm_set1_ps(a[i]);` is creating a 128-bit vector to hold 4 single precision elements. These elements are all initialized with the value at `a[i]`. This vector is used for the entire duration of the inner loop.
2. `__m128 vj = _mm_loadu_ps(&a[j]);` loads the 4 following single precision elements at the location of `a[j]`.
3. `__m128 vdiff = _mm_sub_ps(vi, vj);` subtracts the `vj` vector from the `vi` vector and stores it in the `vdiff` vector. This is equivalent to the `tmp = a[i]-a[j];` step in the base code.
4. `__m128 vbj = _mm_loadu_ps(&b[j]);` loads the 4 following elements at the location of `b[j]`. This is so that the `vdiff` vector can be subtracted.
5. `vbj = _mm_sub_ps(vbj, vdiff);` subtracts `vdiff` from `vbj`. This is equivalent to `b[j] -= tmp;` in the base code.
6. `_mm_storeu_ps(&b[j], vbj);` takes the calculates vector and places it back in the location at `b[j]`.
7. `_mm_storeu_ps(tmp, vdiff);` stores the calculated difference vector in a temporary array of 4 elements. This is so that the elements of the vector can be serially added to `b[i]`.
8. `b[i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];` serially adds the 4 elements of the difference vector to `b[i]`. This is equivalent to `b[i] += tmp;` in the base code.

### Performance of the manually vectorized sweep function:  with `-O3` optimizations

Here, compiler optimizations are used as it has been established that automatic vectorization doesn't work for this 
code, and these optimizations are needed to get the best out of the manual vectorization. The tests used here are much
smaller due to the long duration of this sweep algorithm. At worst, the loaded arrays are stored within L2 cache.

```
Comparing sweep_base_sgl (from ./sweep.x) to sweep_128_sgl (from ./sweep.x)
Benchmark                                      Time        CPU     Time Old    Time New      CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_128_sgl]/4096     -0.7510    -0.7510      2839589      706938      2832949      705285
[sweep_base_sgl vs. sweep_128_sgl]/8192     -0.7503    -0.7503     11420092     2851359     11393386     2844676
[sweep_base_sgl vs. sweep_128_sgl]/16384    -0.7492    -0.7492     45814773    11488096     45707635    11461231
[sweep_base_sgl vs. sweep_128_sgl]/32768    -0.7490    -0.7490    183477650    46046978    183048596    45938907
OVERALL_GEOMEAN                             -0.7499    -0.7499            0           0            0           0
```

The manual vectorization with the compiler optimizations are about 4x faster than the code without the manual
vectorization for 128-bit vectors. The results are the same with or without aligned data, similar to the results from
the ABC test with 128-bit vectors. Near-optimal speedup is gained for 256-bit vectors with a speedup of ~7.7x. These
results are shown in the table below. Again, results are similar with aligned data.

```
Comparing sweep_base_sgl (from ./sweep.x) to sweep_256_sgl (from ./sweep.x)
Benchmark                                      Time        CPU     Time Old    Time New      CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_sgl]/4096     -0.8704    -0.8704      2838588      367947      2831950      367085
[sweep_base_sgl vs. sweep_256_sgl]/8192     -0.8710    -0.8710     11421675     1473129     11394899     1469684
[sweep_base_sgl vs. sweep_256_sgl]/16384    -0.8703    -0.8703     45804432     5940806     45696865     5925814
[sweep_base_sgl vs. sweep_256_sgl]/32768    -0.8703    -0.8703    183489696    23804857    183059879    23744846
OVERALL_GEOMEAN                             -0.8705    -0.8705            0           0            0           0
```

Although not shown here, without the compiler optimizations, the speedup of the manually vectorized code with 128-bit
vectors is at best, ~1.5x faster; and ~2.7x faster for 256-bit vectors.

The double precision versions of the code also obtain optimal performance gain the 128-bit vectors. But unlike the
single precision version, which didn't obtain optimal performance for 256-bit vectors, the double precision data
managed to obtain very close to optimal (3.97x).

```
Comparing sweep_base_dbl (from ./sweep.x) to sweep_128_dbl (from ./sweep.x)
Benchmark                                      Time        CPU     Time Old    Time New      CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------
[sweep_base_dbl vs. sweep_128_dbl]/4096     -0.5008    -0.5008      2839639     1417415      2832999     1414100
[sweep_base_dbl vs. sweep_128_dbl]/8192     -0.4996    -0.4997     11420212     5714520     11393505     5700021
[sweep_base_dbl vs. sweep_128_dbl]/16384    -0.4995    -0.4996     45816958    22930231     45709819    22872302
[sweep_base_dbl vs. sweep_128_dbl]/32768    -0.4798    -0.4799    183502279    95457124    183073179    95216351
OVERALL_GEOMEAN                             -0.4949    -0.4949            0           0           0           0
```

```
Comparing sweep_base_dbl (from ./sweep.x) to sweep_256_dbl (from ./sweep.x)
Benchmark                                      Time        CPU     Time Old    Time New     CPU Old     CPU New
---------------------------------------------------------------------------------------------------------------
[sweep_base_dbl vs. sweep_256_dbl]/4096     -0.7478    -0.7479      2841982      716732     2835337      714913
[sweep_base_dbl vs. sweep_256_dbl]/8192     -0.7470    -0.7470     11418812     2889162    11392066     2881874
[sweep_base_dbl vs. sweep_256_dbl]/16384    -0.7426    -0.7426     45815691    11794693    45708548    11767115
[sweep_base_dbl vs. sweep_256_dbl]/32768    -0.7508    -0.7508    186231849    46409514   185796344    46292393
OVERALL_GEOMEAN                             -0.7470    -0.7470            0           0           0           0
```

While not shown here, without the compiler optimisations, the 128-bit double precision manually vectorized code is
up to 30% slower with 128-bit vectors, and at best, 1.62x faster with 256-bit vectors.