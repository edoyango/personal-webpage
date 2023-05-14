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

```fortran
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

```fortran
real(real64), allocatable:: x(:, :)
...
allocate(x(dims, ntotal), ...)
call random_number(x)
```

When compiling and executing this code, the following times are:

| no. of points | no flags | `-O0` | `-O3` |
| ------------- | -------- | ----- | ----- |
| 1000          | 0.00351s | 0.00358 | 0.00143 |
| 8000          | 0.15779s | 0.15726 | 0.05317 | 
| 27000         | 1.71186s | 1.75904 | 0.52538 |
| 64000         | 9.71352s | 9.83662 | 2.91456 |

## C++ version 1: a vector of structures

A challenge that arises in C++ and scientific computing is storing multi-dimensional data. This is relevant here as we
need to store the position of the points in a 2D array - one dimension for the coordinates, and one for the 