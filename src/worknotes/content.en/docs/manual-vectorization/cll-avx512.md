---
title: Vectorizing Cell-based Pair Search
weight: 4
---

# Vectorizing Cell-based Pair Search

To calculate the acceleration of SPH particles, one must first find the pairs. In the case where incompressible or
weakly-compressible assumptions are used, and where particles' kernel radius is fixed, [the cell-linked list](https://en.wikipedia.org/wiki/Cell_lists)
strategy is often employed. The basis of the algorithm is that described on one of [my other pages](../useful/grid-rows-spatial-hashing.md)
written in Fortran. Below is the C++ version of the main sweep code. 

```cpp {style=tango,linenos=false}
int npairs = 0;
for (int hashi = gridhash[0]; hashi <= gridhash[nelem-1]; ++hashi) {
    for (int ii = starts[hashi]; ii < starts[hashi+1]; ++ii) {
        for (int jj = ii+1; jj < starts[hashi+2]; ++jj) {
            sph::dr(x[ii], x[jj], y[ii], y[jj], dx[niac], dy[niac], r[niac]);
            pair_i[niac] = ii;
            pair_j[niac] = jj;
            niac += r[niac] < kh;
        }
        int hashbotleft = hashi-ngridy-1;
        for (int jj = starts[hashbotleft]; jj < starts[hashbotleft+3]; ++jj) {
            sph::dr(x[ii], x[jj], y[ii], y[jj], dx[niac], dy[niac], r[niac]);
            pair_i[niac] = ii;
            pair_j[niac] = jj;
            niac += r[niac] < kh;
        }
    }
}
```

Some distinctions between this code and the Fortran code is that the positions are seperated into two `std::vector`, and
distance vectors and magnitudes are stored in `std::vectors`, as opposed to being discarded immediately (the calculation
is performed in `sph::dr` function). This is useful because these values can be reused lated to calculate kernel values,
gradients, and particles' acceleration. 

## Vectorizing the code with AVX512 instructions

```cpp {style=tango,linenos=false}
/* NB: cutoff is selected as being 2.4x the average particle spacing 
and particles are arranged randomly in a unit square */
int npairs = 0;
__m512d vdx, vdy, vdr; // initializing 128-bit vectors for relative position vectors and magnitude
// declare vector to store relative positions of each element in an 8-element vector
__m256i zero2seven = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
// outer loop, iterating over each grid cell
for (int hashi = gridhash[0]; hashi <= gridhash[nelem-1]; ++hashi) {
    // inner loop, iterating over particles in the given grid cell
    for (int ii = starts[hashi]; ii < starts[hashi+1]; ++ii) {
        // initializing particle i's position
        __m512d vix = _mm_set1_pd(x[ii]);
        __m512d viy = _mm_set1_pd(y[ii]);

        // calculating where vectorized loop should end
        int len = starts[hashi+2] - (ii+1);
        int end = (len % 8 == 0) ? starts[hashi+2] : starts[hashi+2] - 8;
        int jj;

        // looping over j particles (vectorized)
        for (jj = ii+1; jj < end; jj += 8) {
            // load j particles' positions
            __m512d vjx = _mm512_loadu_pd(&x[jj]);
            __m512d vjy = _mm512_loadu_pd(&y[jj]);
            // calculate relative position vectors and magnitude
            sph::vdr_512(vix, viy, vjx, vjy, vdx, vdy, vdr);
            // mask to determine which elements in the __m128d vdr vector are less than the cutoff
            __mmask8 umask = _mm512_cmp_pd_mask(vdr, _mm512_set1_pd(cutoff), _CMP_LT_OQ);
            // use the mask to "compress" the vdr/vdx/vdy values into the lower elements of the vector
            _mm512_storeu_pd(&r[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdr));
            _mm512_storeu_pd(&dx[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdx));
            _mm512_storeu_pd(&dy[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdy));
            // storing pair_i and pair_j using similar compress
            _mm256_storeu_epi32(&pair_i[niac], _mm256_set1_epi32(ii));
            __m256i vjdx = _mm256_add_epi32(_mm256_set1_epi32(jj), zero2seven); // translate relative index vector by jj
            _mm256_storeu_epi32(&pair_j[niac], _mm256_mask_compress_epi32(_mm256_setzero_si256(), umask, vjdx));
            // increment npairs by summing the umask bits
            npairs += __builtin_popcount(umask);
        }

        // perform residual calculations
        for (; jj < starts[hashi+2]; ++jj) {
            sph::dr(x[ii], x[jj], y[ii], y[jj], dx[niac], dy[niac], r[niac]);
            pair_i[niac] = ii;
            pair_j[niac] = jj;
            niac += r[niac] < kh;
        }

        // repeat above process for adjacent cells
        int hashbotleft = hashi-ngridy-1;
        len = starts[hashbotleft+3] - starts[hashbotleft];
        end = (len % 8 == 0) ? starts[hashbotleft+3] : starts[hashbotleft+3] - 8;
        for (jj = starts[hashbotleft]; jj < end; jj += 8) {
            __m512d vjx = _mm512_loadu_pd(&x[jj]);
            __m512d vjy = _mm512_loadu_pd(&y[jj]);
            sph::vdr_512(vix, viy, vjx, vjy, vdx, vdy, vdr);
            __mmask8 umask = _mm512_cmp_pd_mask(vdr, _mm512_set1_pd(cutoff), _CMP_LT_OQ);
            _mm512_storeu_pd(&r[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdr));
            _mm512_storeu_pd(&dx[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdx));
            _mm512_storeu_pd(&dy[niac], _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdy));
            _mm256_storeu_epi32(&pair_i[niac], _mm256_set1_epi32(ii));
            __m256i vjdx = _mm256_add_epi32(_mm256_set1_epi32(jj), zero2seven);
            _mm256_storeu_epi32(&pair_j[niac], _mm256_mask_compress_epi32(_mm256_setzero_si256(), umask, vjdx));
            niac += __builtin_popcount(umask);
        }
        
        for (; jj < starts[hashbotleft+3]; ++jj) {
            sph::dr(x[ii], x[jj], y[ii], y[jj], dx[niac], dy[niac], r[niac]);
            pair_i[niac] = ii;
            pair_j[niac] = jj;
            niac += r[niac] < kh;
        }
    }
}
```

Here, I make use of the 512-bit vectors to find pairs of particles and requires AVX512f instructions. In addition to the
use of 512-bit vectors, AVX512f is needed for the `_mm*_cmp_*_mask` and `_mm*_mask_compress_*` instructions.
The latter function compresses elements into the lower elements of the resultant vector, based on a mask. _That mask can
be determined at runtime_, which is not the case fo SSE*/AVX2 shuffle and blend instructions.

The calculation of `vdx/vdy/vdr` follows a straightforward pattern, where data is loaded from each of the i particles
and the relative position vector and magnitude is calculated for each of the j particles. `pair_i` is stored using the
constant (relative to the iteration) `ii` value, and `pair_j` is stored using a similar mask and compress pattern.

`_mm_cmp_pd_mask` compares the values of the first argument (in this case, the `__m128d` vector of relative distance),
and compares it with the second argument, another vector (in this case, created by `_mm_set1_pd(cutoff)`). The
comparison function is determined by the third argument, which is `_CMP_LT_OQ` which is less than or equal to. In other
words, each element of the `vdr` vector is checked whether it is less than or equal to `cutoff`. The result is stored in
the mask `umask`. 

To illustrate this a bit better, consider the below simplied example of an iteration:

```cpp {style=tango,linenos=false}
// define cutoff
double cutoff = 1.;

// define i particle's position (0., 0.)
__m512d vix = _mm512_set1_pd(0.);
__m512d viy = _mm512_set1_pd(0.);

// define j particle's position
__m512d vjx = _mm512_set_pd(-1., -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.);
__m512d vjy = _mm512_set_pd(-1., -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.);

// calculate dx, dy, dr
__m512d vdx = _mm512_sub_pd(vix, vjx) // vdx = [1., 0.75, 0.5, 0.25, -0.25, -0.5, -0.75, -1.]
__m512d vdy = _mm512_sub_pd(viy, vjy) // vdy = [1., 0.75, 0.5, 0.25, -0.25, -0.5, -0.75, -1.]
__m512d vdr = _mm512_add_pd(
    _mm512_mul_pd(vdx, vdx),
    _mm512_mul_pd(vdy, vdy)
)                                     // vdr = [2., 1.125, 0.5, 0.125, 0.125, 0.5, 1.125, 2.]
vdr = _mm512_sqrt_pd(vdr)             // vdr = [1.414, 1.061, 0.707, 0.353, 0.353, 0.707, 1.061, 1.414]

/* calculate mask
the mask is 8 bits, each "1" bit means the corresponding index satistfies the comparison.
Below results in 0b00111100
*/
__mmask8 umask = _mm_cmp_pd_mask(vdr, _mm_set1_pd(cutoff), _CMP_LT_OQ)

// "compress" elements of vdr using mask and then store into r, and set other values to 0.
vdr =  _mm512_mask_compress_pd(_mm512_setzero_pd(), umask, vdr)

double r[8];
_mm512_storeu_pd(r, vdr); // this should result in r = [0.707, 0.353, 0.353, 0.707, 0., 0., 0., 0.]
```

## Performance relative to base

```
Comparing pair_search (from results/pairsearch2D-avx512.json) to pair_search_avx512_512 (from results/pairsearch2D-avx512.json)
Benchmark                                            Time       CPU   Time Old   Time New    CPU Old    CPU New
---------------------------------------------------------------------------------------------------------------
[pair_search vs. pair_search_avx512_512]/4096     -0.4131   -0.4131     460838     270454     459760     269821
[pair_search vs. pair_search_avx512_512]/16384    -0.3575   -0.3575    2282786    1466726    2277422    1463294
[pair_search vs. pair_search_avx512_512]/65536    -0.3265   -0.3265    9946220    6698604    9922961    6682938
[pair_search vs. pair_search_avx512_512]/131072   -0.3073   -0.3073   20956765   14516657   20907746   14482700
OVERALL_GEOMEAN                                   -0.4177   -0.4177          0          0          0          0
```

From the results above, we get a meaningful speedup of about 1.4-1.7x, albeit far from an ideal 8x speedup. A possible 
reason for the far-from-ideal speedup is high ratio of residual to vectorized iterations. This is relevant because here,
inner loops are vectorized and there are residual iterations for every time an inner loop is executed. In this case,
where the cutoff is set as 2.4x the average particle spacing, that means that on average, and ignoring edges of the
square that hte particles are contained in, each cell has only `2.4*2.4 = 5.76` particles. This implies that the first
inner (non-vectorized) loop will have only `2 * 5.76 - 1 = 10.52` iterations on average. The second loop will 
have `3 * 5.76 = 17.28` iterations on average. These are estimates, and the real averages are probably lower since
cells near at the edge of the domain will not have as many particles in them.

It's also important to note that `_mm*_mask_compress_*` is very expensive relative to all the other intrinsics used
here.

If I use 256-bit vectors instead, but maintain the same overall algorithm, I get speedups slightly better than when
using 512-bit vectors (see comparison below), which somewhat supports the view that residual iterations are relevant
(using smaller vectors reduces residual iterations). Note that the code will need AVX512VL instructions to make use of
the 256-bit compress instructions.

```
Comparing pair_search (from results/pairsearch2D-avx512.json) to pair_search_avx512_256 (from results/pairsearch2D-avx512.json)
Benchmark                                            Time       CPU   Time Old   Time New    CPU Old    CPU New
---------------------------------------------------------------------------------------------------------------
[pair_search vs. pair_search_avx512_256]/4096     -0.4218   -0.4218     460838     266450     459760     265827
[pair_search vs. pair_search_avx512_256]/16384    -0.3588   -0.3588    2282786    1463624    2277422    1460201
[pair_search vs. pair_search_avx512_256]/65536    -0.3186   -0.3186    9946220    6776961    9922961    6761108
[pair_search vs. pair_search_avx512_256]/131072   -0.3183   -0.3183   20956765   14285855   20907746   14252428
OVERALL_GEOMEAN                                   -0.4208   -0.4208          0          0          0          0
```

Note that using 128-bit vectors reduces speedup in this case to below that of using 512-bit vectors. I find that the
256-bit vectors provides the most consistent speedups, regardless of cutoff.

## Possible future improvements

Ideally, there would be a way to remove the need for using the `compress` instructions altogether. This could
potentially be circumvented by calculating distances of all *potential* pairs, and then perhaps sorting (e.g., with 
[Intel's avx512 sort](https://github.com/intel/x86-simd-sort)), or filtering; although I'm unsure if there are any
SIMD implementations of the latter.