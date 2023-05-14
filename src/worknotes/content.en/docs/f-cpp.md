---
title: A Basic Comparison of C++ vs Fortran
weight: 1
---

# A Basic Comparison of C++ vs Fortran

I'm an avid user of Fortran and this is pretty well known in my team. I'm not particularly evangalistic about using
Fortran, but I do feel it has its place in modern programming, despite the fact that it's one of the oldest programming
languages out there. This has kind of been echoed by other Fortran users. For example, in [Matsuoka et al. (2023)](https://arxiv.org/pdf/2301.02432.pdf),
They declare that "Fortran is dead, long live the DSL!" is a HPC myth. They go on to explain that the high performance 
of Fortran is not easily reproduced even in languages like C.

The first author, Satoshi Matsuoka, is pretty important in the HPC world currently. He is heavily involved in the HPC
community as a researcher as well as leading the development of multiple supercomputers. One of these supercomputers,
Fugaku, took the first spot of IO500 upon debut, which notably chose to use custom-built CPUs [Sato et al. (2022)](https://www.computer.org/csdl/magazine/mi/2022/02/09658212/1zw1njCTTeU)
based on the applications that they intended to run. So, while [Matsuoka et al. (2023)](https://arxiv.org/pdf/2301.02432.pdf)
is not particularly scientific in how they declare their myths, what it says does carry weight due to the author's
background and experience.

My colleague was (and still is) pretty skeptical about the performance advantages of Fortran over other languages (in
particular C++). I hadn't been able to erase his skepticism due to my naivety on the subject. I had "grown up with" (in
a programming sense) Fortran, and had only played with C++. So I figured, that if I was to continue to use Fortran, I
might as well be aware of how it compares to other languages.

Here, I use the "direct search" method of finding pairs of points that lay within a certain distance of eachother. I use
pair search algorithms within the context of SPH, where each of these pairs have a kernel weight associated with it. So
the algorithm shown here will also calculating this value for each pair.

The C++ code I'm writing here is supposed to be from the perspective of a beginner. Firstly because I am a beginner C++
coder, but also because this exercise is supposed to help beginners decide why they want to choose Fortran over C++ when
starting a new project. This hypothetical beginners knowledge of C++ will come from the [Learn C++](https://www.learncpp.com/)
website, as it is often recommended for beginners.

## Hardware

Everything on this page is run on the following platform:

* An Inspiron 7501 with
    * i7-10750H CPU
    * 24GB (16GB + 8GB) @ 2933MHz RAM
    * Windows 10
* WSL2 Ubuntu 20.04

The `g++`/`gfortran` compilers are used here (version 9.4.0).

## Testing scheme

Each version of the code is compiled multiple times with different flags for optimisations:
1. no flags i.e., default optimisation
2. `-O0` (no optimisation)
3. `-O3` (aggressive optimisations)
4. `-fstrict-aliasing` (strict aliasing - only for C++ code)

the strict aliasing optimisation was added because variables in Fortran are strictly "aliased", whereas this is
not the case for C++. This apparently allows for optimisations to be made when compiling all Fortran code. See [thisstackoverflow discussion](https://stackoverflow.com/questions/13078736/fortran-vs-c-does-fortran-still-hold-any-advantage-in-numerical-analysis-thes) 
for more details.

## The Fortran code

Without going into too much detail, the `dsearch` subroutine is really just a nested loop that loops over the points
twice to find which are within the cutoff distance (`scale_k*hsml`) of eachother. The loop is written to take advantage
of symmetry. Consequently, the final list of pairs (`pair_i`, and `pair_j`), are unique i.e., given a pair `(i, j)`, the
reversed pair `(j, i)` doesn't exist.

Note that the points' positions are generated randomly rather than deterministicly, so that data accesses are random.

```fortran {style=tango,linenos=false}
module kernel_and_consts

    use iso_fortran_env

    implicit none

    ! constants
    integer, parameter:: dims = 3
    real(real64), parameter:: scale_k = 2._real64
    real(real64), parameter:: pi = 3.141592653589793_real64

contains

    ! --------------------------------------------------------------------------
    real(real64) function system_clock_timer()
    ! higher-precision timing function than Fortran's inbuilt CPU_TIME

        use, intrinsic:: ISO_FORTRAN_ENV, only: int64

        integer(int64):: c, cr

        call SYSTEM_CLOCK(c, cr)

        system_clock_timer = dble(c)/dble(cr)

    end function system_clock_timer

    ! --------------------------------------------------------------------------
    real(real64) function kernel(r, hsml)
    ! scalar function that calculates the kernel weight given a distance

        real(real64), intent(in):: r, hsml
        real(real64):: q, factor

        factor = 21._real64/(256._real64*pi*hsml*hsml*hsml)
        
        kernel = factor*dim(scale_k, q)**4*(2._real64*q + 1._real64)

    end function kernel

    ! --------------------------------------------------------------------------
    subroutine dsearch(x, ntotal, hsml, pair_i, pair_j, w, niac)
    ! the direct pair search algorithm

        real(real64), intent(in):: x(dims, ntotal), hsml
        integer, intent(in):: ntotal
        integer, intent(out):: pair_i(:), pair_j(:), niac
        real(real64), intent(out):: w(:)
        integer:: i, j
        real(real64):: r

        niac = 0
        do i = 1, ntotal-1
            do j = i+1, ntotal
                r = sum((x(:, i)-x(:, j))**2)
                if (r < hsml*hsml*scale_k*scale_k) then
                    r = sqrt(r)
                    niac = niac + 1
                    pair_i(niac) = i
                    pair_j(niac) = j
                    w(niac) = kernel(r, hsml)
                end if
            end do
        end do

    end subroutine dsearch

end module kernel_and_consts

! ------------------------------------------------------------------------------
program main

    use kernel_and_consts

    implicit none
    integer, parameter:: nx(4) = [10, 20, 30, 40]
    integer:: ntotal, maxinter, i, j, n, niac
    real(real64):: dx, hsml, tstart, tstop
    real(real64), allocatable:: x(:, :), w(:)
    integer, allocatable:: pair_i(:), pair_j(:)

    do n = 1, 4

        ! define problem size
        ntotal = nx(n)**3          ! number of points
        maxinter = 150*ntotal      ! maximum number of iterations
        dx = 1._real64/dble(nx(n)) ! average spacing between points
        hsml = 1.5_real64*dx       ! smoothing length (an SPH term - defines cutoff)

        ! allocate and initalize data
        allocate(x(dims, ntotal), pair_i(maxinter), pair_j(maxinter), w(maxinter))
        niac = 0
        call random_number(x)

        ! begin direct search
        tstart = system_clock_timer()
        call dsearch(x, ntotal, hsml, pair_i, pair_j, w, niac)
        tstop = system_clock_timer()

        ! write times to terminal
        write(output_unit, "(A, I0)") "Size: ", ntotal;
        write(output_unit, "(A, F10.6, A)") "Time: ", tstop-tstart, "s";
        write(output_unit, "(A, I0)") "No. of pairs: ", niac;

        ! deallocate arrays to be resized
        deallocate(x, pair_i, pair_j, w)

    end do

end program main
```

Note that the definition and initialisation of the 2D array, `x` was contained in 3 (non-consecutive) lines in `main`:

```fortran {style=tango,linenos=false}
real(real64), allocatable:: x(:, :)
...
allocate(x(dims, ntotal), ...)
call random_number(x)
```

Note that the only "import" is the `iso_fortran_env`, which is used only for `real64` (real number precision), and
`output_unit` (writing to `stdout`).

When compiling and executing this code, the following times are:

| no. of points | no flags | `-O0` | `-O3` |
| ------------- | -------- | ----- | ----- |
| 1000          | 0.00351s | 0.00358s | 0.00143s |
| 8000          | 0.15779s | 0.15726s | 0.05317s | 
| 27000         | 1.71186s | 1.75904s | 0.52538s |
| 64000         | 9.71352s | 9.83662s | 2.91456s |

## C++ version 1: a vector of structures

A challenge that arises in C++ and scientific computing is storing multi-dimensional data. This is relevant here as we
need to store the position of the points in a 2D array - one dimension for the coordinates, and one for the points. My
first version of the C++ code uses a vector of structures, where the structure is just an array of length `dim` (in 
this case `dim` = 3). Vectors are often the recommended way to store array data. But for multi-dimensional data, there
are many ways to store that in a vector e.g., a vector of vectors or a vector of tructs. Here I opt to use a vector of
structs as it is a fairly common way to organise point data when the point is assocated with many values.

```cpp {style=tango,linenos=false}
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <chrono>

using namespace std;

constexpr int dim = 3;
constexpr double scale_k = 2.;
constexpr double pi = 3.141592653589793;

// declaring position struct which a vector will be made from
struct position {
    double x[dim];
};

double kernel(const double r, const double hsml) {

    const double q = r/hsml;
    const double max02q = max(0., scale_k-q);
    const double factor = 21./(256.*pi*hsml*hsml*hsml);
    return factor*max02q*max02q*max02q*max02q*(2.*q + 1.);

}

void dsearch(const vector<position>& pos,
             const int ntotal,
             const double hsml,
             vector<int>& pair_i, 
             vector<int>& pair_j, 
             vector<double>& w, 
             int& niac) {

    niac = 0;
    double r, dx;

    for (int i = 0; i < ntotal-1; ++i) {
        for (int j = i+1; j < ntotal; ++j) {

            r = 0.;
            for (int d = 0; d < dim; ++d) {
                dx = pos[i].x[d]-pos[j].x[d];
                r += dx*dx;
            }

            if (r < scale_k*scale_k*hsml*hsml) {
                pair_i[niac] = i;
                pair_j[niac] = j;
                r = sqrt(r);
                w[niac] = kernel(r, hsml);
                niac++;
            }

        }
    }
}

int main() {

    const double from = 0.;
    const double to = 1.;
    random_device dev;
    mt19937 generator(dev());
    uniform_real_distribution<double> distr(from, to);

    constexpr int nx[4] {10, 20, 30, 40};

    int ntotal, maxinter;
    double dx, hsml;
    for (int n = 0; n < 4; ++n) {
        ntotal = nx[n]*nx[n]*nx[n];
        maxinter = ntotal*150;
        dx = 1./static_cast<double>(nx[n]);
        hsml = 1.5*dx;

        // declaring all vectors
        vector<position> pos(ntotal);
        vector<int> pair_i(maxinter), pair_j(maxinter);
        vector<double> w(maxinter);
        int niac = 0;

        for (int i = 0; i < ntotal; ++i) {
            for (int d = 0; d < dim; ++d) {
                pos[i].x[d] = distr(generator);
            }
        }

        auto start = chrono::high_resolution_clock::now();
        dsearch(pos, ntotal, hsml, pair_i, pair_j, w, niac);
        auto stop = chrono::high_resolution_clock::now();

        double time = static_cast<double> (chrono::duration_cast<chrono::microseconds>(stop-start).count())/1000000.;
        cout << "Size: " << ntotal << "\n";
        cout << "Time: " << time << "s\n";
        cout << "No. of pairs: " << niac << "\n";
    }

    return 0;
} 
```

Some comments about this code:
* The structure of the code is kept similar to the Fortran code for comparison purposes.
* There are many more "imports" needed than the Fortran code. `algorithm` is needed for `max`, `cmath` is needed for
`sqrt`, `random` is needed for the random number generator, `iostream` is needed for IO, `chrono` is needed for timing,
and `vector` is needed for the `vector` datatype. I would argue that this is a bit of a barrier to productivity for
beginners as many functions and types are located in disaparate headers that one needs to lookup or remember.
* The `dim` function isn't available in the `algorithm` header.
* The random number generator is more verbose to setup and use compared to the `random_number` subroutine in Fortran.
* The absence of a `sum` function that works on arrays requires the manual writing of the calculation of `r` (inter-point)
distance. 
* the `pow` function can be slow under default/no optimisations (see [this discussion on stackeroverflow](https://stackoverflow.com/questions/2940367/what-is-more-efficient-using-pow-to-square-or-just-multiply-it-with-itself)), so the quartic power was manually repeated.
* The `chrono` high precision timers are available without the need to program your own (unlike in the Fortran code).


When compiling and executing this code, the following times are:

| no. of points | no flags | `-O0` | `-O3` | `-fstrict-aliasing` |
| ------------- | -------- | ----- | ----- | ---------------- |
| 1000          | 0.009426s | 0.008331s | 0.001344s | 0.009052s |
| 8000          | 0.493711s | 0.51553s | 0.06409s | 0.528323s |
| 27000         | 5.74999s | 6.02984s | 0.646653 | 6.14586s |
| 64000         | 32.4458s | 33.8446s | 3.45058s | 34.3546s |

We can see that all but the aggressively optimised version of this code are disgustingly - over 3x the time taken for
the default optimisations Fortran version! However, the aggressive compiler optimisations bring doown the run time
significantly - but still slower than the Fortran version.

## C++ version 2: manual loop unrolling

Let's look at the loop that calculates the square of the inter-point distance:

```cpp {style=tango,linenos=false}
r = 0.;
for (int d = 0; d < dim; ++d) {
    dx = pos[i].x[d]-pos[j].x[d];
    r += dx*dx;
}
```

We can manually unroll this loop, since we know that `dim = 3`:

``` {style=tango,linenos=false}
r = (pos[i].x[0]-pos[j].x[0])*(pos[i].x[0]-pos[j].x[0]) + 
    (pos[i].x[1]-pos[j].x[1])*(pos[i].x[1]-pos[j].x[1]) +
    (pos[i].x[2]-pos[j].x[2])*(pos[i].x[2]-pos[j].x[2]);
```

Compiling this and running it we get:

| no. of points | no flags | `-O0` | `-O3` | `-fstrict-aliasing` |
| ------------- | -------- | ----- | ----- | ---------------- |
| 1000          | 0.017024s | 0.015104s | 0.001302s | 0.013916s |
| 8000          | 0.7424s | 0.793698s | 0.065268s | 0.805783s |
| 27000         | 8.92672s | 9.0975s | 0.590568s | 9.22796s |
| 64000         | 50.4143s | 51.6696s | 3.20469s | 52.8981s |

And the aggressive optimisation version reduces in run time by about 10%, at the expense of all the other versions
almost doubling! The code has also lost some flexibility in that the calculation of `r` no longer updates as we change
`dim`.

## C++ version 3a: 2D (partially) dynamic arrays

Arrays are also an option in C++, and in some cases, like when dealing with matrices, 2D arrays might be more intuitive.

I make use of one of the ways described in [Learn C++'s tutorials](https://www.learncpp.com/cpp-tutorial/pointers-to-pointers/).

To make this change, we change:

```cpp {style=tango,linenos=false}
struct position {
    double x[dim];
};
...
vector<position> pos(ntotal);
vector<int> pair_i(maxinter), pair_j(maxinter);
vector<double> w(maxinter);
```

to

```cpp {style=tango,linenos=false}
auto x = new double [ntotal][dim];
int* pair_i = new int [maxinter];
int* pair_j = new int [maxinter];
```

We also have to update the notation form `pos[i].x[d]` to `x[i][d]` as appropriate. The arrays now have to be manually
deleted, so I add the the following code to just before the loop through the problem sizes ends:

```cpp {style=tango,linenos=false}
delete[] x;
delete pair_i;
delete pair_j;
delete w;
```

which are now mandatory to avoid memory leaks. Finally, we have to change the `dsearch` function declaration from

```cpp {style=tango,linenos=false}
void dsearch(const vector<position>& pos,
             const int ntotal,
             const double hsml,
             vector<int>& pair_i, 
             vector<int>& pair_j, 
             vector<double>& w, 
             int& niac) {
```

to

```cpp {style=tango,linenos=false}
void dsearch(const double x[][dim],
             const int ntotal,
             const double hsml,
             int* pair_i, 
             int* pair_j, 
             double* w, 
             int& niac) {
```

I should note that the Learn C++ tutorial doesn't explain how to pass the 2D array to a function, and I eventually had to
Google this to figure out that the 2D array would be passed as `double x[][dim]`.

I should also note that this declaration only works because `dim` is a `constexpr`. One of the less simple approaches in
the Learn C++ tutorial would have to be used if `dim` wasn't constant. Fortran wouldn't experience this problem.

Compiling this and running, we get:

| no. of points | no flags | `-O0` | `-O3` | `-fstrict-aliasing` |
| ------------- | -------- | ----- | ----- | ---------------- |
| 1000          | 0.005682s | 0.006304s | 0.001652s | 0.005846s |
| 8000          | 0.304066s | 0.314797s | 0.046719s | 0.322617s |
| 27000         | 3.31205s | 3.53902s | 0.491014s | 3.55314s |
| 64000         | 18.8977s | 19.5868s | 2.81262s | 19.5882s |

Which shows the aggressively optimised program is now faster than the Fortran version! However, the other versions are
still much slower. I think `-O3` optimisation is safe for most applications, but this result is still meaningful as the
default optimisations are much safer and easier to debug.

## C++ version 3b: 2D (fully) dynamic arrays

In the current case, the `dim` variable is a `constexpr`, so `x` can be allocated with `auto x = new double [ntotal][dim];`.
In the case where `dim` is a run-time variable, then the declaration changes to:

```cpp {style=tango,linenos=false}
double** x = new double*[ntotal];
for (int i = 0; i < ntotal; ++i) {
    x[i] = new double[dim];
}
```

and the corresponding deallocation becomes:

```cpp {style=tango,linenos=false}
for (int i = 0; i < ntotal; ++i) {
    delete[] x[i];
}
delete[] x;
```

Both the allocation and deallocation statements are pretty gross and unintuitive for beginners. Furthermore, are mandatory
to avoid memory leaks. The definition of `x` in the `dsearch` declaration does become simpler however:

```cpp {style=tango,linenos=false}
void dsearch(double** x,
             const int ntotal,
             const double hsml,
             int* pair_i, 
             int* pair_j, 
             double* w, 
             int& niac) {
```

Compiling and running:

| no. of points | no flags | `-O0` | `-O3` | `-fstrict-aliasing` |
| ------------- | -------- | ----- | ----- | ---------------- |
| 1000          | 0.006434s | 0.005638s | 0.001538s | 0.005885s |
| 8000          | 0.295594s | 0.32687s | 0.057452s | 0.299215s |
| 27000         | 3.29056s | 3.3863s | 0.58139s | 3.31922s |
| 64000         | 18.6892s | 18.9621s | 3.03032s | 19.3092s |

And the speed of the aggresively optimised version has now dropped, although the less-optimised versions appear to have
gotten faster than version 3a.

## C++ version 4: contiguous 2D (partially) dynamic arrays

If you [do some digging around](https://stackoverflow.com/questions/21943621/how-to-create-a-contiguous-2d-array-in-c),
you may eventually find out that allocating 2D arrays like in version 3, does not guarantee that data will be
contiguous when it comes to C++. Consequently, you can change the declaration of the `x` 2D array to

```cpp {style=tango,linenos=false}
double** x = new double*[ntotal];
double* xdata = new double[ntotal*dim];
for (int i = 0; i < ntotal; ++i, xdata += dim)
    x[i] = xdata;
```

which is honestly pretty gross, but is necessary to ensure that the data is contiguous in memory. The corresponding
freeing of this array is done by

```cpp {style=tango,linenos=false}
delete x[0];
delete x;
```

The definition of `x` in the `dsearch` declaration remains the same as version 3b.

I clear disadvantage beeing the special definitions and deletions that would require some wrapping. It's also not very
beginner friendly from a conceptual standpoint. 

Now compiling and running:

| no. of points | no flags | `-O0` | `-O3` | `-fstrict-aliasing` |
| ------------- | -------- | ----- | ----- | ---------------- |
| 1000          | 0.005687s | 0.006379s | 0.001573s | 0.006154s |
| 8000          | 0.294367s | 0.305595s | 0.052221s | 0.316611s |
| 27000         | 3.17132s | 3.38158s | 0.537241s | 3.44193s |
| 64000         | 18.5441s | 19.2792s | 2.84747s | 19.533s |

The aggresively optimised version is now about as fast as the Fortran version, which is faster than version 3b, but slower
than 3a. 

### Aggresive compiler optimisations with strict aliasing

So far, the `-fstrict-aliasing` flag doesn't seem to make a difference when combined with the default optimisations. How
about combining it with `-O3`?

| no. of points | `-O3 -fstrict-aliasing` | 
| ------------- | -------- | 
| 1000          | 0.00136s |
| 8000          | 0.052659s |
| 27000         | 0.532555s | 
| 64000         | 2.8715s | 

Combining the `-fstrict-aliasing` with `-O3` doesn't seem to make a discernable difference, although it seems to slow
down version 4 by a trivial amount - consistent with the other versions.

## Discussion

It's clear that C++ can be just as fast as Fortran. However, I would argue that the speed wasn't as easy to arrive to as
what it was with Fortran. The Fortran code is written in a way not very different from how a beginner would write it and using
only intrinsics. Furthermore the only import was `iso_fortran_env` which wasn't mandatory (`real(real64)` declerations 
could be replaced with `double precision`, `_real64` suffixes could be replaced with `d0`, and `output_unit` could be
replaced with `*`); whereas all the headers for the C++ versions were mandatory. The rather subtle optimisation of the
calculation of `r` reduced the C++ code's flexibility, but was necessary to bring down the speed to the level that GNU's
Fortran compiler's `sum` function is at. The declaration of either partial or fully dynamic multi-dimensional
arrays is unintuitive, lengthy (wait until you see even higher dimensions!), and requires caution; whereas Fortran's 
allocation is easy to read and easy to write for higher dimensions. Finally, the important improvement of the code from
using non-contiguous dynamic arrays to contiguous is not easily found or quick to understand by a beginner (I have the
advantage of years of experience coding).

However, I should mention that version 3a of the C++ code compiled with the aggressive optimisations was an unexepected
and interesting result. I hypothesise that when `x` is declared in such a way, that the compiler forces the data to be
contiguous. I think further optimisations are made by leveraging the fact that the inner dimension (or the column) is constant in
size. 

But before closing I should mention that this experiment handled a rather uncommon case where nearly all the code being
written cannot leverage the plethora of C/C++ libraries available. I think the difference in how the `dsearch` function
was timed is a good illustration of a key difference between Fortran and C++. And that is that C++ has a lot of packages
(in the form of libraries and header files) that have a lot of high-level functionality that you can utilise, whereas
this is not the case for Fortran. With Fortran, you will often need to write your own subrouties to achieve certain
objectives, like how I needed to write my own high-precision timer function when the C++ code could leveragethe `chrono`
STL headers. But I should also add that it's not difficult to call C++ code from Fortran, so there is potential to
mix-and-match (see [this relatively simple example](https://fortran-lang.discourse.group/t/calling-c-from-fortran/3402)).

## Conclusion

Fortran was designed for maths (mainly calculus and linear algebra), and so its syntax favours those applications -
particularly in the use of arrays. It also has a pretty good collection of mathematical intrinsic functions that you
don't have to look to hard for. Consequently, if your problem can be expressed in multi-dimensional arrays (or tensors),
then it will much easier to write fast code. 

This simple expeirment demonstrates that C++ can definitly achieve the same performance for applications that Fortran was designed for. But, it's a *general purpose* language, so it takes a fair bit of work to apply it to the subset of problems that Fortran targets. For example, strict aliasing and contiguous data in memory are *required* in Fortran, whereas there is a fair
bit of effort needed to set that up in C++ when it comes ot multi-dimensional arrays.

So, I think I'll continue using Fortran, because at the end of the day, I enjoy working with numerical methods and
simulations, which ultimately was what Fortran was designed for.

One day I'll have a look at how the new kid on the block, Julia (and maybe Rust?), compares! 