---
title: Faster Vector Sum Reduction
weight: 4
---

# Faster Vector Sum Reduce

In [my page introducing vectorized reductions](sumreduce.md), I show how to use horizontal adds (`hadd`) functions to
perform the reduction. However, this [doesn't produce optimal assembly](https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction/35270026), so isn't the fastest way to do things.

This page will demonstrate a faster version to do each of the examples shown previously.

## Single precision, SSE instructions

This section will show a faster sum-reduce example which makes use of 128-bit single precision vectors and SSE
instructions. 

After loading the vector `va`, a new vector is created using `_mm_movehdup_ps`, using `a` as an input. This function
duplicates the second and fourth elements, and duplicates them to one element lower.

```cpp {style=tango,linenos=false}
// __m128 va = _mm_set_ps(a0, a1, a2, a3);

__m128 shuf = _mm_movehdup_ps(va);

// shuf = [a1, a1, a3, a3]
```

The resultant vector is then added back to `va`:

```cpp {style=tango,linenos=false}
__m128 sums = _mm_add_ps(va, shuf);

// sums = [a0+a1, a1+a1, a2+a3, a3+a3]
```

`shuf` and `sums` are combined together with `_mm_movehl_ps`. This function replaces the lower two elements of the first
input vector, with the upper two elements of the second vector.

```cpp {style=tango,linenos=false}
shuf = _mm_movehl_ps(shuf, sums);

// shuf = [sums2, sum3, shuf2, shuf3]
//      = [a2+a3, a3+a3, a3, a3]
```

We finally add the lowermost element of `shuf` and `sums` to get the reduced sum in the lowermost element of the `sums`
vector, and the remaing elements can be ignored.

```cpp {style=tango,linenos=false}
sums = _mm_add_ss(sums, shuf);

// sums = [sums0+shuf0, sums1, sums2, sums3]
//      = [a0+a1+a2+a3, a1+a1, a2+a3, a3+a3]
```

Implemented into our previous example:

```cpp {style=tango,linenos=false}
void reduce_128_sgl_SSE3(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128 vb = _mm_set_ss(b[i/8]);
        for (int j = 0; j < 8; j += 4) {
            __m128 va = _mm_loadu_ps(&a[i+j]);
            __m128 shuf = _mm_movehdup_ps(va);
            __m128 sums = _mm_add_ps(va, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            vb = _mm_add_ss(vb, sums);
        }
        _mm_store_ss(&b[i/8], vb);
    }
}
```

### Performance

Compilation command:

```bash {style=tango,linenos=false}
g++ -o reduce.x reduce.cpp -msse4 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -std=c++17 -O2
```

Comparison with base case:

``` {style=tango,linenos=false}
Comparing reduce_base_sgl (from reduce_base_sgl.json) to reduce_128_sgl_SSE3 (from reduce_128_sgl_SSE3.json)
Benchmark                                                Time       CPU   Time Old   Time New    CPU Old    CPU New
-------------------------------------------------------------------------------------------------------------------
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/4096        -0.6108   -0.6107       1625        633       1621        631
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/32768       -0.6270   -0.6271      13565       5059      13531       5046
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/262144      -0.5984   -0.5983     108413      43541     108135      43439
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/2097152     -0.5380   -0.5379     872616     403151     870408     402208
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/16777216    -0.2613   -0.2613    8405297    6209082    8385642    6194461
[reduce_base_sgl vs. reduce_128_sgl_SSE3]/134217728   -0.2114   -0.2114   68570641   54072282   68394712   53933850
OVERALL_GEOMEAN                                       -0.4997   -0.4996          0          0          0          0
```

Comparison with horizontal add strategy:

``` {style=tango,linenos=false}
Comparing reduce_128_sgl (from reduce_128_sgl.json) to reduce_128_sgl_SSE3 (from reduce_128_sgl_SSE3.json)
Benchmark                                               Time       CPU   Time Old   Time New    CPU Old    CPU New
------------------------------------------------------------------------------------------------------------------
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/4096        -0.4926   -0.4925       1247        633       1243        631
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/32768       -0.4930   -0.4930       9978       5059       9953       5046
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/262144      -0.4553   -0.4553      79935      43541      79748      43439
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/2097152     -0.3698   -0.3697     639709     403151     638091     402208
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/16777216    -0.1334   -0.1332    7164522    6209082    7146262    6194461
[reduce_128_sgl vs. reduce_128_sgl_SSE3]/134217728   -0.1041   -0.1042   60358462   54072282   60204987   53933850
OVERALL_GEOMEAN                                      -0.3602   -0.3602          0          0          0          0
```

The new vectorized reduction is between 1.1x to 2x faster than the previous attempt using horizontal adds. This brings
up the speedup compared to the original to ~1.3x to ~2.6x times. Like every other example, performance gains are 
greater the closer the data is to the CPU core.

## Single precision, AVX instructions

This implementation is not much different from the previous version, in that the reduction using a 256-bit vector is
performed by splitting the 256-bit vector into 2 128-bit vectors, adding them, and performing the same reduction
algorithm for 128-bit vectors:

```cpp
void reduce_256_sgl_AVX(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128 vb = _mm_set_ss(b[i/8]);
        for (int j = 0; j < 8; j += 8) {
            __m256 va = _mm256_loadu_ps(&a[i+j]);
            __m128 va_lo = _mm256_castps256_ps128(va);
            __m128 va_hi = _mm256_extractf128_ps(va, 1);
            va_lo = _mm_add_ps(va_lo, va_hi);
            __m128 shuf = _mm_movehdup_ps(va_lo);
            __m128 sums = _mm_add_ps(va_lo, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            vb = _mm_add_ss(sums, vb);
        }
        _mm_store_ss(&b[i/8], vb);
    }
}
```

### Performance

First comparing this new version to the first 256-bit vectorized version:

``` {style=tango,linenos=false}
Comparing reduce_256_sgl (from reduce_256_sgl.json) to reduce_256_sgl_AVX (from reduce_256_sgl_AVX.json)
Benchmark                                              Time       CPU   Time Old   Time New    CPU Old    CPU New
-----------------------------------------------------------------------------------------------------------------
[reduce_256_sgl vs. reduce_256_sgl_AVX]/4096        -0.1999   -0.2001        800        640        798        638
[reduce_256_sgl vs. reduce_256_sgl_AVX]/32768       -0.1937   -0.1935       6419       5175       6401       5162
[reduce_256_sgl vs. reduce_256_sgl_AVX]/262144      -0.1721   -0.1723      52211      43227      52079      43108
[reduce_256_sgl vs. reduce_256_sgl_AVX]/2097152     -0.0966   -0.0966     446301     403176     445074     402074
[reduce_256_sgl vs. reduce_256_sgl_AVX]/16777216    -0.0417   -0.0417    6538546    6266122    6520420    6248835
[reduce_256_sgl vs. reduce_256_sgl_AVX]/134217728   -0.0436   -0.0436   57544865   55037665   57385371   54886129
OVERALL_GEOMEAN                                     -0.1271   -0.1272          0          0          0          0
``` 

The performance has definitely improved, albeit marginally for the test sizes in RAM. But compared to the 128-bit
version, it actually performs very similarly (perhaps even a bit worse):

``` {style=tango,linenos=false}
Comparing reduce_128_sgl_SSE3 (from reduce_128_sgl_SSE3.json) to reduce_256_sgl_AVX (from reduce_256_sgl_AVX.json)
Benchmark                                                   Time       CPU   Time Old   Time New    CPU Old    CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/4096        +0.0101   +0.0102        634        640        632        638
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/32768       +0.0233   +0.0236       5057       5175       5043       5162
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/262144      +0.0210   +0.0208      42337      43227      42231      43108
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/2097152     +0.0043   +0.0041     401449     403176     400435     402074
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/16777216    +0.0047   +0.0045    6236505    6266122    6220706    6248835
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/134217728   -0.0041   -0.0043   55265558   55037665   55122733   54886129
OVERALL_GEOMEAN                                          +0.0099   +0.0098          0          0          0          0
```

I'm not 100% sure why this happens. Firstly, there's an additional `_mm_add_ps` function call, which eats into potential
gains from using 256-bit vectors. And I believe the additional memory operations eat into the remaining gains. The 
`_mm256_castps256_ps128` should be low cost or free as it doesn't move any data around. However, `_mm256_extractf128_ps` 
has to copy the data from the top half of the  original `va` vector into a different register. Which, I think, is more
costly than an `_mm_hadd_ps` function call.

Ultimately, from this test, it's probably not worth implementing vector reduction with 256-bit vectors compared to the
fast 128-bit version.

### Performance with LLVM compiler (`clang++`)

The LLVM compilers are a different family of compilers form the GNU compilers and can translate the C++ code to assembly
differently. In the case of C++, we use the `clang++` compiler. Let's compare the performance of our test programs using
`clang++`.

The compilation command for the manually vectorized code is:

```bash
clang++ -o reduce.x reduce.cpp -mavx2 -Ibenchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2
```

The base case needed additional options to ensure no SSE/AVX instructions were introdued:

```
-mno-sse -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-avx -mno-avx2
```

First comparing the faster version of the 256-bit vector code with the faster version of the 128-bit vector code.

```
Comparing reduce_128_sgl_SSE3 (from reduce_128_sgl_SSE3-clang.json) to reduce_256_sgl_AVX (from reduce_256_sgl_AVX-clang.json)
Benchmark                                                   Time       CPU   Time Old   Time New    CPU Old    CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/4096        -0.4445   -0.4445        572        318        571        317
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/32768       -0.4693   -0.4693       4708       2499       4697       2493
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/262144      -0.4916   -0.4916      39230      19943      39138      19896
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/2097152     -0.5283   -0.5283     338017     159451     337229     159080
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/16777216    -0.8006   -0.8006    6395979    1275520    6381010    1272545
[reduce_128_sgl_SSE3 vs. reduce_256_sgl_AVX]/134217728   -0.8162   -0.8162   55513176   10205195   55383675   10181392
OVERALL_GEOMEAN                                          -0.6294   -0.6294          0          0          0          0
```

The AVX code is significantly faster for all the test sizes. For the smallest problem sizes, the AVX code is 1.8x faster
than the new SSE code. Furthermore, the larger the problem gets and further away the data is located from the CPU, the
speedups also increase (up to 5.4x at the largest test size!). Comparing to the base case:

```
Comparing reduce_base_sgl (from reduce_base_sgl-clang.json) to reduce_256_sgl_AVX (from reduce_256_sgl_AVX-clang.json)
Benchmark                                               Time       CPU   Time Old   Time New    CPU Old    CPU New
------------------------------------------------------------------------------------------------------------------
[reduce_base_sgl vs. reduce_256_sgl_AVX]/4096        -0.7450   -0.7450       1247        318       1244        317
[reduce_base_sgl vs. reduce_256_sgl_AVX]/32768       -0.7494   -0.7494       9970       2499       9947       2493
[reduce_base_sgl vs. reduce_256_sgl_AVX]/262144      -0.7505   -0.7505      79935      19943      79749      19896
[reduce_base_sgl vs. reduce_256_sgl_AVX]/2097152     -0.7506   -0.7506     639364     159451     637869     159080
[reduce_base_sgl vs. reduce_256_sgl_AVX]/16777216    -0.8318   -0.8318    7581972    1275520    7564185    1272545
[reduce_base_sgl vs. reduce_256_sgl_AVX]/134217728   -0.8375   -0.8375   62785954   10205195   62639140   10181392
OVERALL_GEOMEAN                                      -0.7815   -0.7815          0          0          0          0
```

We now get a speedup of between 3.9x and 6.2x. The lesson here, is that it is important to benchmark your manually
vectorized code with other compilers, as the translation to assembly and the heuristics used to automatically optimise
the code can vary.

## Double precision, SSE instructions 

This algorithm tricks the compiler to produce optimal code. I don't understand it, but will explain the specific
functions used.

```cpp {style=tango,linenos=false}
// __m128d va = _mm_set_pd(a0, a1);

__m128 undef = _mm_undefined_ps();

// undef = [?, ?, ?, ?]
```

This function call simply creates a 128-bit *single precision* vector with undefined values. Explicitly saying the
vector is undefined apparently helps the compiler optimise the code better.

```cpp {style=tango,linenos=false}
__m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(va));

// _mm_castpd_ps(va) = [a0, a0.5, a1, a1.5]
// shuftmp = [a1, a1.5, ?, ?]
```

I found this pretty clever when I saw it. The motivation for this trick is that there is no `_mm_movehl_ps` equivalent 
for double precision data. What happens here is:

1. `va` is cast to a single precision vector.
    * This doesn't change the data itself. It only changes how functions interepret it.
    * i.e., the 128 bits in the vector are still the same.
2. The upper 64 bits of `va` (i.e., the second element), is placed in the first element position of `undef`.
3. The resultant vector is stored in `shuftmp`.

This effectively moves the 2nd element of `va` into the first place of `undef`, despite not having an equivalent
`movehl` function for double precision. This will be converted back into a double precision vector next.

```cpp {style=tango,linenos=false}
__m128d shuf = _mm_castps_pd(shuftmp)

// shuf = [a1, ?]
```

`_mm_castps_pd` converts `shuftmp` back into a double precision vector, which is saved as `shuf`.

```cpp {style=tango,linenos=false}
va = _mm_add_sd(va, shuf)

// va = [a0+a1, ?]
```

which has the reduced sum in the lower value of `va`. Implemented into the previous example:

```cpp {style=tango,linenos=false}
void reduce_128_sgl_SSE2(const int nelem, float* a, float* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128d vb = _mm_set_sd(b[i/8]);
        for (int j = 0; j < 8; j += 2) {
            __m128 va = _mm_loadu_ps(&a[i+j]);
            __m128 undef = _mm_undefined_ps();
            __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(va));
            __m128d shuf = _mm_castps_pd(shuftmp);
            va = _mm_add_sd(va, shuf);
            vb = _mm_add_sd(vb, va);
        }
        _mm_store_sd(&b[i/8], vb);
    }
}
```

### Performance

Compilation command:

```bash {style=tango,linenos=false}
g++ -o reduce.x reduce.cpp -msse4 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -std=c++17 -O2
```

Comparison with base case:

``` {style=tango,linenos=false}
Comparing reduce_base_dbl (from reduce_base_dbl.json) to reduce_128_dbl_SSE2 (from reduce_128_dbl_SSE2.json)
Benchmark                                                Time       CPU    Time Old    Time New     CPU Old     CPU New
-----------------------------------------------------------------------------------------------------------------------
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/4096        -0.1979   -0.1978        1627        1305        1623        1302
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/32768       -0.1842   -0.1843       13462       10982       13431       10956
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/262144      -0.1666   -0.1667      112035       93353      111772       93135
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/2097152     -0.1767   -0.1765      934021      768984      931648      767179
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/16777216    -0.0174   -0.0172    12864721    12640848    12831738    12611107
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/134217728   -0.0080   -0.0078   107276227   106417505   107002000   106168603
OVERALL_GEOMEAN                                       -0.0729   -0.0724           0           0           0           0
``` 

The gains are modest, with a speedup of 1.2x at most. Comparison with horizontal add strategy:

``` {style=tango,linenos=false}
Comparing reduce_128_dbl (from reduce_128_dbl.json) to reduce_128_dbl_SSE2 (from reduce_128_dbl_SSE2.json)
Benchmark                                               Time       CPU    Time Old    Time New     CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/4096        -0.1037   -0.1033        1456        1305        1452        1302
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/32768       -0.0828   -0.0828       11973       10982       11945       10956
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/262144      -0.1349   -0.1347      107906       93353      107633       93135
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/2097152     -0.1437   -0.1435      898008      768984      895718      767179
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/16777216    -0.0293   -0.0291    13022105    12640848    12988742    12611107
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/134217728   -0.0180   -0.0178   108369477   106417505   108088106   106168603
OVERALL_GEOMEAN                                      -0.0667   -0.0991           0           0           0           0
```

Which shows that this faster version is 1.1-1.4x faster than the horizontal add version.

### Performance with LLVM compiler (`clang++`)

Like with the single precision AVX code, we can see whether using `clang++` changes performance. The compiler options
remain the same for the manually vectorized code:

```bash
clang++ -o reduce.x reduce.cpp -mavx2 -Ibenchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -O2
```

The base case needed additional options to ensure no SSE/AVX instructions were introdued:

```
-mno-sse -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-avx -mno-avx2
```

```
Comparing reduce_128_dbl (from reduce_128_dbl-clang.json) to reduce_128_dbl_SSE2 (from reduce_128_dbl_SSE2-clang.json)
Benchmark                                               Time       CPU    Time Old    Time New     CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/4096        -0.4131   -0.4131        1247         732        1244         730
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/32768       -0.0490   -0.0490        9970        9482        9946        9459
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/262144      +0.0132   +0.0132       84054       85161       83857       84962
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/2097152     +0.0415   +0.0416      692974      721728      691305      720034
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/16777216    -0.0231   -0.0230    12876249    12578212    12844822    12548782
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/134217728   -0.0154   -0.0154   108116751   106446645   107853527   106196771
OVERALL_GEOMEAN                                      -0.0904   -0.0903           0           0           0           0
```

And now it looks like the state of things have improved! This new version, compiled with `clang++`, shows an
improvement, at least for the smallest test size. The others show a minor change, again demonstrating that it's
important to test your code with multiple compilers!

## Double precision, AVX instructions

Like the 256-bit vectorized code so far, the pattern is to split the 256-bit vector into two 128-bit vectors, add the
two halves together, and then perform a 128-bit sum reduction. Using the faster version of the single precision 128-bit
reduction, the code looks like:

```cpp
void reduce_256_dbl_AVX(const int nelem, double* a, double* b) {

    for (int i = 0; i < nelem; i += 8) {
        __m128d vb = _mm_set_sd(b[i/8]);
        for (int j = 0; j < 8; j += 4) {
            __m256d va = _mm256_loadu_pd(&a[i+j]);
            __m128d va_low = _mm256_castpd256_pd128(va);
            __m128d va_hi = _mm256_extractf128_pd(va, 1);
            va_low = _mm_add_pd(va_low, va_hi);
            __m128 undef = _mm_undefined_ps();
            __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(va_low));
            __m128d shuf = _mm_castps_pd(shuftmp);
            va_low = _mm_add_sd(va_low, shuf);
            vb = _mm_add_sd(vb, va_low);
        }
        _mm_store_sd(&b[i/8], vb);
    }
}
```

### Performance

```
Comparing reduce_256_dbl (from reduce_256_dbl.json) to reduce_256_dbl_AVX (from reduce_256_dbl_AVX.json)
Benchmark                                                           Time             CPU      Time Old      Time New       CPU Old       CPU New
------------------------------------------------------------------------------------------------------------------------------------------------
[reduce_256_dbl vs. reduce_256_dbl_AVX]/4096                     -0.1645         -0.1645           956           799           954           797
[reduce_256_dbl vs. reduce_256_dbl_AVX]/32768                    -0.1435         -0.1435          8371          7170          8351          7153
[reduce_256_dbl vs. reduce_256_dbl_AVX]/262144                   -0.0466         -0.0466         78961         75280         78776         75103
[reduce_256_dbl vs. reduce_256_dbl_AVX]/2097152                  -0.0662         -0.0662        659969        616299        658426        614846
[reduce_256_dbl vs. reduce_256_dbl_AVX]/16777216                 -0.0201         -0.0201      12603210      12350490      12573727      12321468
[reduce_256_dbl vs. reduce_256_dbl_AVX]/134217728                -0.0169         -0.0169     106022048     104234811     105773315     103990081
OVERALL_GEOMEAN                                                  -0.0781         -0.0781             0             0             0             0
```

The improvement here is meaningful - up to ~16% for smaller problems. When compiled with `clang++`, the performance increase is more modest.

## Conclusion

The horizontal add strategy for reducing vectors is acceptable, but by using more subtle choices in vectorization, the
sum reduction can be improved significantly. This was most evident in the 128-bit single precision sum reduction shown
here. The improved algorithm shown on this page, was better than the horizontal add version by up to 2x!

The improved 256-bit version, obtained better performance over the horizontal add version, but only if the `clang++`
compiler is used. With `clang++`, the new 256-bit version is approximately between 2x and 5x faster than even the
updated 128-bit sum reduction.

In contrast, the double precision 128-bit and 256-bit improved versions showed here showed modest improvements compared
to the previous vectorized version. Both versions seemed to be a lot faster for the smallest test size when compiled
with `clang++` (around 2x faster).