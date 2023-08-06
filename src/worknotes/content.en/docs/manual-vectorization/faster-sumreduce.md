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

## Double precision, SSE instructions (Didn't end up being faster)

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
        __m128d vb = _mm_set_ss(b[i/8]);
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
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/4096        +0.0555   +0.0555        1627        1717        1623        1713
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/32768       +0.0168   +0.0166       13462       13689       13431       13654
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/262144      -0.0095   -0.0097      112035      110972      111772      110691
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/2097152     +0.1208   +0.1208      934021     1046876      931648     1044201
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/16777216    -0.0130   -0.0130    12864721    12697055    12831738    12664523
[reduce_base_dbl vs. reduce_128_dbl_SSE2]/134217728   -0.0086   -0.0086   107276227   106349711   107002000   106080057
OVERALL_GEOMEAN                                       +0.0259   +0.0258           0           0           0           0
```

Comparison with horizontal add strategy:

``` {style=tango,linenos=false}
Comparing reduce_128_dbl (from reduce_128_dbl.json) to reduce_128_dbl_SSE2 (from reduce_128_dbl_SSE2.json)
Benchmark                                               Time       CPU    Time Old    Time New     CPU Old     CPU New
----------------------------------------------------------------------------------------------------------------------
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/4096        +0.1796   +0.1794        1456        1717        1452        1713
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/32768       +0.1433   +0.1431       11973       13689       11945       13654
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/262144      +0.0284   +0.0284      107906      110972      107633      110691
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/2097152     +0.1658   +0.1658      898008     1046876      895718     1044201
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/16777216    -0.0250   -0.0250    13022105    12697055    12988742    12664523
[reduce_128_dbl vs. reduce_128_dbl_SSE2]/134217728   -0.0186   -0.0186   108369477   106349711   108088106   106080057
OVERALL_GEOMEAN                                      +0.0755   +0.0754           0           0           0           0
```