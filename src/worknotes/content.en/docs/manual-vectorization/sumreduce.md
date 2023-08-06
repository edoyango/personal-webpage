---
title: Vector Sum Reduction
weight: 3
---

# Vector Sum Reduction

We now we can add two SIMD vectors together element-by-element. However, I'm sure you can imagine scenarios where you
might want to add all the elements in a vector together i.e., a sum-reduction. This page will explain how to do this
with 128-bit and 256-bit vectors the approach is different for each vector width. I would show how to do this with
512-bit vectors, but I don't have a CPU with AVX512 instructions handy.

## Reducing 4 floats

### base case

We'll assume a 1D array that has every 8 elements reduced into an element of another array:

```cpp
void reduce_base_sgl(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        for (int j = 0; j < 8; ++j) {
            b[i/8] += a[i+j]; // assume b is already zeroed
        }
    }

}
```

The loop tries to mimick cases where many small sum-reductions are performed.

### 128-bit vectorization (SSE)

In the first demonstration of vectorizing the sumreduction loop above, we'll make use of a few new
functions:

```cpp
__m128 v = _mm_set_ss(val);      // creates a vector where the lowermost element is val, and the rest are 0
__m128 v = _mm_add_ss(a, b);     // adds only the lowermost elements of a and b. The rest are same as a.
__m128 v = _mm_hadd_ps(a, b);    // performs a "horizontal" add (see explanation below)
_mm_store_ss(&a, v);             // stores the lower-most value in the vector
```

The `_mm_hadd_xx` function will be the workhorse of the vectorized reduction. It takes two vectors and
adds pairs from each half of both vectors. The first element will be the sum of the second half of the
first vector, `a2+a3`. The second element will be the first half of the first vector, `a0+a1`. The third
and fourth elements will be the same, but for the second input vector. 

```cpp
// float a[4] {a0, a1, a2, a3};   
// float b[4] {b0, b1, b2, b3};

_mm128 v = _mm_hadd_ps(a, b);

// v = [a2+a3, a0+a1, b2+b3, b0+b1];
```

Because `hadd` adds two vectors in this manner, it can be executed twice in a row to sum all elements of
an `__m128` vector:

```cpp
// float a[4] {a0, a1, a2, a3};

// first horizontal add
_mm128 v = _mm_hadd_ps(a, a);

// v = [a2+a3, a0+a1, a2+a3, a0+a1]

// second horizontal add
v = _mm_hadd_ps(v, v);

// v = [v2+v3, v0+v1, v2+v3, v0+v1] 
//   = [a2+a3+a0+a1, a2+a3+a0+a1, a2+a3+a0+a1, a2+a3+a0+a1]
```

The double `hadd` functions reduces the number of sum operations by half, compared to a serial sum of all the elements
int the vector. Implementing these new functions to vectorize our loop:

```cpp
void reduce_128_sgl(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128 vb = _mm_set_ss(b[i/8]); // initializes a vector with the sum-reduction variable
        for (int j = 0; j < 8; j += 4) { // start the reduction loop
            __m128 va = _mm_loadu_ps(&a[i+j]); // load the data
            va = _mm_hadd_ps(va, va);    // first horizontal add
            va = _mm_hadd_ps(va, va);    // second horizontal add
            vb = _mm_add_ss(va, vb);     // add result to the sum-reduction vector variable
        }
        _mm_store_ss(&b[i/8], vb);       // store the left-most value in the sum-reduction variable
    }
}
```

The use of the `__m128 vb` vector helps speed up the loop slightly by mitigating the need to increment
the `float *b` array directly in the inner loop. In this case, it reduces the number of stores by one.

In the double precision version, it would reduce the stores by 3, since the inner loop would have 4
iterations instead of only 2. Note that with the double precision version, reducing a 128-bit double vector to one value
doesn't reduce the number of additions. Consequently, there shouldn't be much difference between codes.

### Performance (128-bit)

Compilation command (using the Google benchmark framework):

```bash
g++ -o reduce.x reduce.cpp -msse4 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -std=c++17 -O2
```

#### Single precision

```
Comparing reduce_base_sgl (from ./reduce.x) to reduce_128_sgl (from ./reduce.x)
Benchmark                                            Time        CPU     Time Old     Time New      CPU Old    CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_base_sgl vs. reduce_128_sgl]/4096         -0.5562    -0.5562         2810         1247         2802       1244
[reduce_base_sgl vs. reduce_128_sgl]/32768        -0.5555    -0.5555        22439         9973        22383       9950
[reduce_base_sgl vs. reduce_128_sgl]/262144       -0.5546    -0.5545       179534        79961       179048      79775
[reduce_base_sgl vs. reduce_128_sgl]/2097152      -0.5547    -0.5545      1437233       640012      1433349     638519
[reduce_base_sgl vs. reduce_128_sgl]/16777216     -0.4209    -0.4207     12413446      7188664     12379323    7171888
[reduce_base_sgl vs. reduce_128_sgl]/134217728    -0.3994    -0.3993    100100389     60117657     99844433    9977347
OVERALL_GEOMEAN                                   -0.6329    -0.6328            0            0            0          0
```

The tests that fit on L1-L3 cache benefit the most with about 2.2x speedup. The largest test shows the worst speedup, at
 ~1.7x speedup. The ~2x speedup obtained in these tests are inline with expectations as the SIMD reduction shown here 
 only reduces adds by half.

#### Double Precision

```
Comparing reduce_base_dbl (from ./reduce.x) to reduce_128_dbl (from ./reduce.x)
Benchmark                                            Time        CPU     Time Old     Time New      CPU Old    CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_base_dbl vs. reduce_128_dbl]/4096         -0.1113    -0.1109         1625         1444         1621       1441
[reduce_base_dbl vs. reduce_128_dbl]/32768        -0.0930    -0.0926        13207        11979        13171      11951
[reduce_base_dbl vs. reduce_128_dbl]/262144       -0.0288    -0.0285       110962       107761       110662     107510
[reduce_base_dbl vs. reduce_128_dbl]/2097152      -0.0254    -0.0252      1072861      1045634      1070112    1043194
[reduce_base_dbl vs. reduce_128_dbl]/16777216     -0.0052    -0.0051     13032581     12964964     12998883    2933011
[reduce_base_dbl vs. reduce_128_dbl]/134217728    +0.0069    +0.0071    107110710    107847148    106838485    7594330
OVERALL_GEOMEAN                                   -0.0447    -0.0444            0            0            0          0
```

As expected, the performance difference is minimal, although there appears to be some meaningful improvements to the
test sizes that fit within L1 and L2 cache (i.e., the 4,096 and 32,768 element tests). I'm not sure on this, but I am
guessing that the improvements are due to SIMD vectors being used to store the intermediate reduction data. 

Another interesting observation is that the double precision version base version performs faster on the smallest case
than the equivalent for the single precision version. This is probably CPU-specific, as I don't observe this on my
desktop CPU (Intel i7-10750H).

### 256-bit vectorization (AVX)

256-bit vectors double the vector width before. This means 8 elements in a single precisoin (`__m128`) vector, and 4
elements in a double precision vector. For single precision data, the `_mm256_hadd_ps` function works like:

```cpp
// float a[8] {a0, a1, a2, a3, a4, a5, a6, a7};
// float b[8] {b0, b1, b2, b3, b4, b5, b6, b7};

_mm256 v = _mm256_hadd_ps(a, b);

// v = [a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7];
```

If we were to apply this three times, in an attempt to apply the `hadd` strategy to perform the reduction:

```cpp
// float a[8] {a0, a1, a2, a3, a4, a5, a6, a7};

// first horizontal add
_mm256 v = _mm256_hadd_ps(a, a);

// v = [a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7];

// second horizontal add
_mm256 v = _mm256_hadd_ps(v, v);

// v = [v0+v1,       v2+v3,       v0+v1,       v2+v3,       v4+v5,       v6+v7,       v4+v5,       v6+v7];
// v = [a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a0+a1+a2+a3, a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7, a4+a5+a6+a7]

// third horizontal add
_mm256 v = _mm256_hadd_ps(v, v);

// v = [v0+v1,                   v2+v3, v0+v1, v2+v3, v4+v5,                   v6+v7, v4+v5, v6+v7];
// v   [a0+a1+a2+a3+a0+a1+a2+a3, ...  , ...  , ...  , a4+a5+a6+a7+a4+a5+a6+a7, ...  , ...  , ...]
```

Where the repeated entries are skipped in the last addition. Each element of the bottom 128 bits of the vector have
`2*(a0+a1+a2+a3)` and the top 128 bits have `2*(a4+a5+a6+a7)`. Hopefully it's clear that simply performing three `hadd`s
in a row would not be possible to perform our reduction, at least for 8 elements of single precision data. 

A similar problem occurs when performing two consecutive `hadd` operations on double precision data. The
`_mm256_hadd_pd` function works like:

```cpp
// float a[4] {a0, a1, a2, a3};
// float b[4] {b0, b1, b2, b3};

_mm256 v = _mm256_hadd_pd(a, b);

// v = [a0+a1, b0+b1, a2+a3, b2+b3];
```

And so trying to apply this twice for the purposes of reduction:

```cpp
// float a[4] {a0, a1, a2, a3};

// first horizontal add
_mm256 v = _mm256_hadd_pd(a, a);

// v = [a0+a1, a0+a1, b2+b3, b2+b3]

// second horizontal add
_mm256 v = _mm256_hadd_pd(v, v);

//v = [v0+v1,       v0+v1, v2+v3,       v2+v3]
//  = [a0+a1+a0+a1, ...,   a2+a3+a2+a3, ...]
```

Which should illustrate that double `_mm256_hadd_pd` has the same limitation as `_mm256_hadd_ps`, as the two halves of 
the resultant 256-bit vector merely contains double of the sum of all elements in each of the corresponding 128-bit
halves.

`__m256` vectors can be reduced by first splitting the vector into two `__m128` vectors, adding the two halves together
and then performing the 128-bit vector reduction shown previously. For example, for single precision:

```cpp
void reduce_256_sgl(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128 vb = _mm_set_ss(b[i/8]); // initializes a vector with the sum-reduction variable
        for (int j = 0; j < 8; j += 8) { // start the reduction loop
            __m256 va = _mm256_loadu_ps(&a[i+j]); // load the data
            __m128 va_low = _mm256_castps256_ps128(va); // "truncate" the 256-bit vector to get the lower 128-bit vector
            __m128 va_hi = _mm256_extractf128_ps(va, 1); // extract the upper 128-bit vector
            va_low = _mm_add_ps(va_low, va_hi); // add the two halves together
            va_low = _mm_hadd_ps(va_low, va_low);    // first horizontal add
            va_low = _mm_hadd_ps(va_low, va_low);    // second horizontal add
            vb = _mm_add_ss(va_low, vb);     // add result to the sum-reduction vector variable
        }
        _mm_store_ss(&b[i/8], vb);       // store the left-most value in the sum-reduction variable
    }
}
```

Note that in this verison, 8 elements are loaded at a time into the 256-bit vector, which removes the need for the inner
loop. But, the inner loop is kept so that it is more comparable to the base and 128-bit versions.

To extract the lower 128-bit part of the 256-bit vector, the `_mm256_castps256_ps128` function is used. This function
casts an `__m256` vector to a `__m128` vector by truncating the upper 128-bits from the input vector. The upper half
is extracted using `_mm256_extractf128_ps`. The second argument being `1`, is asking for the second 128-bit part of the
vector. Passing `0` would provide the first half, and could also be used to extract the lower 128-bit half, but
`_mm256_castps256_ps128` is apparently faster. The next step is to add these two halves using the `_mm_add_ps` function
as used previously. The remainder of the reduction follows the same double `hadd` idiom used to reduce the 128-bit
single precision vectors.

The same strategy can be employed with double precision 256-bit vectors, except that only one `_mm_hadd_pd` is used,
like when performing the reduction with 128-bit double precision vectors.

### Performance (256-bit)

Compilation command (using the Google benchmark framework):

```bash
g++ -o reduce.x reduce.cpp -mavx -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -std=c++17 -O2
```

#### Single precision

```
Comparing reduce_base_sgl (from ./reduce.x) to reduce_256_sgl (from ./reduce.x)
Benchmark                                            Time        CPU    Time Old    Time New     CPU Old     CPU New
--------------------------------------------------------------------------------------------------------------------
[reduce_base_sgl vs. reduce_256_sgl]/4096         -0.6599    -0.6598        2351         800        2345         798
[reduce_base_sgl vs. reduce_256_sgl]/32768        -0.6597    -0.6596       18800        6398       18752        6383
[reduce_base_sgl vs. reduce_256_sgl]/262144       -0.6538    -0.6538      150427       52078      150075       51955
[reduce_base_sgl vs. reduce_256_sgl]/2097152      -0.6308    -0.6308     1217060      449307     1214213      448256
[reduce_base_sgl vs. reduce_256_sgl]/16777216     -0.4012    -0.4011    10883815     6517219    10856018     6501927
[reduce_base_sgl vs. reduce_256_sgl]/134217728    -0.3710    -0.3709    89385959    56219686    89156887    56087815
OVERALL_GEOMEAN                                   -0.5810    -0.5810           0           0           0           0
```

Best speedup obtained is now ~2.9x and worst is ~1.6x. Given that for every 8 elements added, there are three addition
operations (2 horizontal, 1 normal), a would hope to expect that this implementation would achieve around 8/3 ~= 2.7x
speedup over the non-vectorized base version. The better performance for smaller problems might be because normal adds
are cheaper than horizontal adds. The minimum speedup of ~1.6x is actually worse than that achieved by the 
equivalent 128-bit vector version. 

#### Double precision

```
Comparing reduce_base_dbl (from ./reduce.x) to reduce_256_dbl (from ./reduce.x)
Benchmark                                            Time        CPU     Time Old     Time New      CPU Old      CPU New
------------------------------------------------------------------------------------------------------------------------
[reduce_base_dbl vs. reduce_256_dbl]/4096         -0.4205    -0.4205         1658          961         1655          959
[reduce_base_dbl vs. reduce_256_dbl]/32768        -0.3905    -0.3904        13779         8398        13744         8378
[reduce_base_dbl vs. reduce_256_dbl]/262144       -0.1923    -0.1922       113466        91645       113180        91431
[reduce_base_dbl vs. reduce_256_dbl]/2097152      -0.1299    -0.1297      1037855       903028      1035202       900907
[reduce_base_dbl vs. reduce_256_dbl]/16777216     -0.0097    -0.0094     12856464     12732138     12823510     12702366
[reduce_base_dbl vs. reduce_256_dbl]/134217728    -0.0087    -0.0085    107383159    106447126    107107940    106198075
OVERALL_GEOMEAN                                   -0.2097    -0.2095            0            0            0            0
```

Here, the speedup is between ~1x and ~1.7x. The maximum speedup is significantly better than the equivalent for 128-bit
vectors, but of course, gets worse as data gets more distant from the CPU due to the test sizes. The ~1.7x speedup
achieved falls short of the 2x expected, probably because of the extractions of the 128-bit vectors from the 256-bit
vector, as well as the relatively more expensive horizontal add operation.

Like with the single precision version, the test sizes that are in RAM, do not show any meaningful speedup over the
128-bit equivalent. Although it seems that this one is at least not any worse.

## Conclusion

The horizontal add intrinsics were introduced here, as well as other needed, to perform vectorized reductions of chunks
of data in an array. With single precision data, the speedups achieved for all tests were meaningful. For tests that fit
within cache, the achieved speedup was around expectations of using the horizontal add strategy. These results
correspond with the upper end of the ranges.

|    |Single | Double |
| --- | --- | --- |
| 128-bit | 1.7x-2.2x | 1x-1.1x |
| 256-bit | 1.6x-2.9x | 1x-1.7x |

In contrast, double precision vectorization attempts with 128-bit vectors only achieved 1.1x at best, for problems
that fit in cache, and did not obtain any speedup for tests that could only fit in RAM. The 256-bit vectorization
attempt improved the upper end to 1.7x, but still failed to improve the speed for tests in RAM.