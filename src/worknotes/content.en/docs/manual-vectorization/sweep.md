---
title: Vectorizing A Simple Pair Sweep
weight: 4
---

# Vectorizing A Simple Pair Sweep

One way to perform the force calculation sweep in SPH, is to use a double loop like:

```cpp {style=tango,linenos=false}
for (int i = 0; i < nelem-1; ++i) {
    for (int j = i+1; j < nelem; ++j) {
        // compare particle i and j
        // calculate forces
    }
}
```

This is generally not a preferred way to do the force calculations, as the number of operations in this algorithm scales
with the square of the number of particles. But it's useful for playing around with manual vectorization.

The base case we will use is this:

```cpp {style=tango,linenos=false}
void sweep_base_sgl(const int nelem, float* a, float* b) {
    int k = 0
    for (int i = 0; i < nelem-1; i += 8) {
        for (int j = i+1; j < nelem; ++j) {
            tmp = a[i]-a[j];
            b[i] += tmp;
            b[j] -= tmp;
        }
    }
}
```

`i` is incremented by 8 to simplify vectorization. Swap out `float` with `double`, to get the double precision version.

Important to note here, is that this loop cannot automatically be vectorized by `g++`. For example, if we compile
our code with `-O3 -fopt-info-vec-missed`, `-O3` will try to vectorize the loop, and the other option will let you know
when it cannot vectorize a given loop. At compilation, I get some important info in the output:

``` {style=tango,linenos=false}
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

## Why aligned `store` and `load` functions can't be used

Recalling what is required for the `store` and `load` functions to work: that the memory address being accessed must be
aligned on vector-width boundaries. In the nested loop considered here, `i` starts at element 0, but `j` starts at 
`i + 1`, which would be `j = 1` when `i = 0`. So while the address of `a[0]` would be aligned to a vector width, the
next element, `a[1]`, would not be. Consequently, once the inner loop starts, the load/store functions would fail with a
`segmentation fault` error. More generally, the outer loop would typically be incrementing by 1, which would mean
accesses to `a` would mostly be unaligned.

## Doing the Vectorization

###  Vectorizing the direct pair sweep with 128-bit vectors (SSE instructions)

This attempt to vectorize the loop will turn the outer loop into a vector of all the same variable, and then the inner
loop will load multiple sequential elements to operate the outer-loop vector. This looks like:

```cpp {style=tango,linenos=false}
void sweep_128_sgl(const int nelem, float* a, float* b) {
    float tmp[4];
    for (int i = 0; i < nelem-1; i += 8) {
        __m128 vi = _mm_set1_ps(a[i]);
        for (int j = i+1; j < nelem; j += 4) {
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

The 256-bit vector version can be obtained by replacing `__m128` types with `__m256`, and `_mm` function prefixes with 
`_mm256`. Double precision versions are obtained by adding `d` to the vector types, and replacing `ps` with `pd` in the
function suffixes.

### Performance of the manually vectorized sweep function: with `-O2` optimizations

Here, compiler optimizations are used as it has been established that automatic vectorization doesn't work for this 
code, and these optimizations are needed to get the best out of the manual vectorization. The tests used here are much
smaller due to the long duration of this sweep algorithm. At worst, the loaded arrays are stored within L2 cache.

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep1d_base_sgl.json) to sweep_128_sgl (from results/sweep1d_128_sgl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_128_sgl]/4096    -0.7448   -0.7448     2837605     724git074     2830970     722381
[sweep_base_sgl vs. sweep_128_sgl]/8192    -0.7433   -0.7433    11412353    2929379    11385665    2922530
[sweep_base_sgl vs. sweep_128_sgl]/16384   -0.7411   -0.7411    45808257   11859336    45701186   11831605
[sweep_base_sgl vs. sweep_128_sgl]/32768   -0.7412   -0.7412   183443120   47483586   183014137   47372294
OVERALL_GEOMEAN
```

The manual vectorization with the compiler optimizations are about 3.9x faster than the code without the manual
vectorization for 128-bit vectors. Near-optimal speedup is gained for 256-bit vectors with a speedup of ~7.3x (see table
below).

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep1d_base_sgl.json) to sweep_256_sgl (from results/sweep1d_256_sgl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_sgl]/4096    -0.8603   -0.8603     2837605     396449     2830970     395517
[sweep_base_sgl vs. sweep_256_sgl]/8192    -0.8623   -0.8623    11412353    1571707    11385665    1568032
[sweep_base_sgl vs. sweep_256_sgl]/16384   -0.8589   -0.8589    45808257    6462080    45701186    6446971
[sweep_base_sgl vs. sweep_256_sgl]/32768   -0.8454   -0.8454   183443120   28365528   183014137   28299199
OVERALL_GEOMEAN                            -0.8569   -0.8569           0          0           0          0
```

Although not shown here, without the compiler optimizations, the speedup of the manually vectorized code with 128-bit
vectors is at best, ~1.5x faster; and ~2.7x faster for 256-bit vectors.

The double precision versions of the code also obtain close to optimal performance gain using 128-bit vectors. Like the
single precision version,the double precision data managed to obtain very close to optimal (3.78x).

``` {style=tango,linenos=false}
Comparing sweep_base_dbl (from results/sweep1d_base_dbl.json) to sweep_128_dbl (from results/sweep1d_128_dbl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_dbl vs. sweep_128_dbl]/4096    -0.4898   -0.4898     2837990    1448083     2831355    1444697
[sweep_base_dbl vs. sweep_128_dbl]/8192    -0.4868   -0.4868    11418819    5860700    11392117    5846937
[sweep_base_dbl vs. sweep_128_dbl]/16384   -0.4857   -0.4857    45805303   23556194    45698184   23501027
[sweep_base_dbl vs. sweep_128_dbl]/32768   -0.4882   -0.4882   184497440   94427345   184065995   94206550
OVERALL_GEOMEAN                            -0.4876   -0.4876           0          0           0          0
```

``` {style=tango,linenos=false}
Comparing sweep_base_dbl (from results/sweep1d_base_dbl.json) to sweep_256_dbl (from results/sweep1d_256_dbl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_dbl vs. sweep_256_dbl]/4096    -0.7354   -0.7354     2837990     750796     2831355     749041
[sweep_base_dbl vs. sweep_256_dbl]/8192    -0.7337   -0.7337    11418819    3040832    11392117    3033721
[sweep_base_dbl vs. sweep_256_dbl]/16384   -0.7331   -0.7331    45805303   12227361    45698184   12198775
[sweep_base_dbl vs. sweep_256_dbl]/32768   -0.7253   -0.7253   184497440   50679024   184065995   50560536
OVERALL_GEOMEAN                            -0.7319   -0.7319           0          0           0          0
```

While not shown here, without the compiler optimisations, the 128-bit double precision manually vectorized code is
up to 30% slower with 128-bit vectors, and at best, 1.62x faster with 256-bit vectors.

You may have noticed that the incrementing of `b[i]` looks like it's serialized, but near-optimal performance is
obtained in the tests. This is because `g++` and `clang++` will automatically convert `b[i] += tmp[0] + ...;` to
assembly using SIMD instructions. I couldn't figure out a way to prevent the compilers from doing this without also
making the intel intrinsic functions unavailable as well. In principle, the serializations of the update of `b[i]` would
prevent optimal performance, and the manually vectorized reductions [discussed here](sumreduction.md) and [here](faster-sumreduce.md)
would be needed to improve performance. 

## 2D version

A meaningful version of this sweep function would require at least 2 dimensions as few engineering problems occur in 1D.
Updating the sweep function to work in 2D:

```cpp {style=tango,linenos=false}
void sweep_base_sgl(const int nelem, float* ax, float*, ay, float* bx, float* by) {
    int k = 0
    for (int i = 0; i < nelem-1; i += 8) {
        for (int j = i+1; j < nelem; ++j) {
            tmp = ax[i]-ax[j];
            bx[i] += tmp;
            bx[j] -= tmp;
            tmp = ay[i]-ay[j];
            by[i] += tmp;
            by[j] -= tmp;
        }
    }
}
```

And the corresponding manually vectorized version (128-bit vectors, single precision data): 

```cpp {style=tango,linenos=false}
void sweep_128_sgl(const int nelem, float* ax, float *ay, float* bx, float* by,) {
    float tmp[4];
    for (int i = 0; i < nelem-1; i += 8) {
        __m128 vix = _mm_set1_ps(ax[i]);
        __m128 viy = _mm_set1_ps(ay[i]);
        for (int j = i+1; j < nelem; j += 4) {
            __m128 vj = _mm_loadu_ps(&ax[j]);
            __m128 vdiff = _mm_sub_ps(vix, vj);
            __m128 vbj = _mm_loadu_ps(&bx[j]);
            vbj = _mm_sub_ps(vbj, vdiff);
            _mm_storeu_ps(&bx[j], vbj);
            _mm_storeu_ps(tmp, vdiff);
            bx[i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];

            vj = _mm_loadu_ps(&ay[j]);
            vdiff = _mm_sub_ps(viy, vj);
            vbj = _mm_loadu_ps(&by[j]);
            vbj = _mm_sub_ps(vbj, vdiff);
            _mm_storeu_ps(&by[j], vbj);
            _mm_storeu_ps(tmp, vdiff);
            by[i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];
        }
    }
}
```

Which differs from the 1D version by basically duplicating the code to work on both `ax/bx` and `ay/by`.

### Performance of the 2D version

Performance of the 128-bit, single precision version:

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_128_sgl (from results/sweep2d_128_sgl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_128_sgl]/4096    -0.7374   -0.7374     2872377     754223     2865660     752460
[sweep_base_sgl vs. sweep_128_sgl]/8192    -0.7342   -0.7342    11506013    3057812    11479107    3050661
[sweep_base_sgl vs. sweep_128_sgl]/16384   -0.7311   -0.7311    46017767   12372116    45909943   12343182
[sweep_base_sgl vs. sweep_128_sgl]/32768   -0.7296   -0.7296   184173481   49800358   183742778   49683899
OVERALL_GEOMEAN                            -0.7331   -0.7331           0          0           0          0
```

and the double precision version:

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_128_dbl (from results/sweep2d_128_dbl.json)
Benchmark                                     Time       CPU    Time Old    Time New     CPU Old     CPU New
------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_128_dbl]/4096    -0.4732   -0.4732     2872377     1513260     2865660     1509721
[sweep_base_sgl vs. sweep_128_dbl]/8192    -0.4439   -0.4439    11506013     6397967    11479107     6383008
[sweep_base_sgl vs. sweep_128_dbl]/16384   -0.4514   -0.4514    46017767    25243093    45909943    25184062
[sweep_base_sgl vs. sweep_128_dbl]/32768   -0.4497   -0.4497   184173481   101358663   183742778   101121153
OVERALL_GEOMEAN                            -0.4547   -0.4547           0           0           0           0
```

The single precision version gets a max speedup of 3.8x and the double precision version gets 1.9x which are both pretty
close to optimal! Now what about the 256-bit version?

single precision:

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_256_sgl (from results/sweep2d_256_sgl.json)
Benchmark                                     Time       CPU    Time Old   Time New     CPU Old    CPU New
----------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_sgl]/4096    -0.7104   -0.7104     2872377     831729     2865660     829784
[sweep_base_sgl vs. sweep_256_sgl]/8192    -0.7079   -0.7079    11506013    3361005    11479107    3353127
[sweep_base_sgl vs. sweep_256_sgl]/16384   -0.7016   -0.7016    46017767   13732233    45909943   13700120
[sweep_base_sgl vs. sweep_256_sgl]/32768   -0.6624   -0.6624   184173481   62176512   183742778   62031113
OVERALL_GEOMEAN                            -0.6962   -0.6962           0          0           0          0
```

and double precision:

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_256_dbl (from results/sweep2d_256_dbl.json)
Benchmark                                     Time       CPU    Time Old    Time New     CPU Old     CPU New
------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_dbl]/4096    -0.4926   -0.4926     2872377     1457335     2865660     1453927
[sweep_base_sgl vs. sweep_256_dbl]/8192    -0.4868   -0.4868    11506013     5905197    11479107     5891388
[sweep_base_sgl vs. sweep_256_dbl]/16384   -0.4857   -0.4857    46017767    23664846    45909943    23609391
[sweep_base_sgl vs. sweep_256_dbl]/32768   -0.4381   -0.4381   184173481   103495512   183742778   103253479
OVERALL_GEOMEAN                            -0.4762   -0.4762           0           0           0           0
```

Now the speedup has barely changed, and has, in fact, gotten slower for the single precision version! Profiling this
code reveals that a big chunk of time is spent on the loading of the `vdiff` vector to the `tmp` variable and then
adding 8 (single precision) or 4 (double precision) elements to `bx/by[i]`. 

Swapping out the code would change the load and add combination:

```cpp {style=tango,linenos=false}
_mm_storeu_ps(tmp, vdiff);
bx[i] += tmp[0]+tmp[1]+tmp[2]+tmp[3];
```

and replacing it with something like

```cpp {style=tango,linenos=false}
bx[i] += reduce_256_sgl(vdiff);
```

where `reduce_256_sgl`, uses the function definitions:

```cpp {style=tango,linenos=false}
static inline __attribute__((always_inline))
float reduce_128_sgl(__m128 a) {
    __m128 shuf = _mm_movehdup_ps(a);
    __m128 sums = _mm_add_ps(a, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline __attribute__((always_inline))
float reduce_256_sgl(__m256 a) {
    __m128 a_lo = _mm256_castps256_ps128(a);
    __m128 a_hi = _mm256_extractf128_ps(a, 1);
    a_lo = _mm_add_ps(a_lo, a_hi);
    return reduce_128_sgl(a_lo);
}
```

where `inline __attribute__((always_inline))` ensures that the compiler inlines the function no matter what. The double
precision version would be that described in [the faster sum reductions page](faster-sumreduce.md). The performance
results from using these functions for the single precision version:

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_256_sgl_reduce2 (from results/sweep2d_256_sgl_reduce2.json)
Benchmark                                             Time       CPU    Time Old   Time New     CPU Old     CPU New
-------------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_sgl_reduce2]/4096    -0.8014   -0.8014     2872377     570401     2865660      569065
[sweep_base_sgl vs. sweep_256_sgl_reduce2]/8192    -0.7940   -0.7940    11506013    2370599    11479107     2365055
[sweep_base_sgl vs. sweep_256_sgl_reduce2]/16384   -0.7642   -0.7642    46017767   10848874    45909943    10823504
[sweep_base_sgl vs. sweep_256_sgl_reduce2]/32768   -0.7883   -0.7883   184173481   38997633   183742778    38906450
OVERALL_GEOMEAN                                    -0.7874   -0.7874           0          0           0           0
```

which raises the performance from ~3.5x to ~5x! Not optimal, but a big improvement, and potentially providing sufficient
motivation for the 256-bit manual vectorisation. The double precision performance results also show a similar
improvement, raising the speedup from ~2x to ~2.4x.

``` {style=tango,linenos=false}
Comparing sweep_base_sgl (from results/sweep2d_base_sgl.json) to sweep_256_dbl_reduce2 (from results/sweep2d_256_dbl_reduce2.json)
Benchmark                                             Time       CPU    Time Old   Time New     CPU Old    CPU New
------------------------------------------------------------------------------------------------------------------
[sweep_base_sgl vs. sweep_256_dbl_reduce2]/4096    -0.5912   -0.5912     2872377    1174242     2865660    1171496
[sweep_base_sgl vs. sweep_256_dbl_reduce2]/8192    -0.5372   -0.5372    11506013    5324447    11479107    5311997
[sweep_base_sgl vs. sweep_256_dbl_reduce2]/16384   -0.5334   -0.5334    46017767   21473573    45909943   21423354
[sweep_base_sgl vs. sweep_256_dbl_reduce2]/32768   -0.5680   -0.5680   184173481   79555471   183742778   79369428
OVERALL_GEOMEAN                                    -0.5581   -0.5581           0          0           0          0
```