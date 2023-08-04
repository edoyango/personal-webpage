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

### Performance of the manually vectorized sweep function: with `-O3` optimizations

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