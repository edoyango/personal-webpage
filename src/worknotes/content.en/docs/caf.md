---
title: Coarray Fortran Things
weight: 2
---

# Coarray Fortran Things

ignoring the basics. Best tutorial to start with is [this one](https://github.com/tkoenig1/coarray-tutorial/blob/main/tutorial.md). Basic, but couldn't find very beginner friendly ones.

## Install
[OpenCoarrays](http://www.opencoarrays.org/) is a library usable with gfortran and uses MPI 1-sided comms as the to perform the communications. install via linuxbrew or spack.

Intel compilers don't rely on external libraries and should be ready to use coarrays with intel-MPI. It only requires compilation with `-coarray` option.

## Compiling and Running
### GNU + OpenCoarrays
Compile: `caf -fcoarray=lib`

caf is just a wrapper around the `mpif90` command, which is in turn a wrapper around the underlying compiler (in this case `gfortran`). But you can always use the command `caf --show` to see the include flags and libraries, in case you don't want to use the caf wrapper.

run: `cafrun [MPI options] yourprogram`

To compile for a single image (i.e., serial): `gfortran -fcoarray=single ...` and execute like any other executable (`./myprogram`).

### Intel

Compile: `ifort -coarray`. Run like any other program: `./myprogram`. Control the number of images with the `FOR_COARRAY_NUM_IMAGES` environment variable.

To compile for a single image (serial): `ifort -coarray=single`. However, unlike GNU+OpenCoarrays, you can still execute this executable with `mpiexec [MPI options] myprogram`.

note the intel compiler caf shared library requires intel mpi (even if compiling with `-coarray=single`).
## Defining static coarrays

For variables with copies on all images, just add `[*]` to end of variable declaration e.g.,

```fortran {style=tango,linenos=false}
integer:: a[*] ! a has a copy on each image
real:: b(2, 2)[*] ! each image will have b, a 2x2 array
logical, codimension[*]:: c(3, 3, 3) ! the codimension property can be used instead.
```

user defined types are defined in the normal way as well.

```fortran {style=tango,linenos=false}
type(my_type):: d[*]
type(my_type2), codimension[*]:: e(10)
```

## Defining and allocating allocatable coarrays

Like static variables, allocatable variables are defined with the `[:]` suffix or with the codimension property. The colon being used instead of the * is needed and means the number of images is being deferred (maybe you can allocate variables only on some images then? Needs testing).

```fortran {style=tango,linenos=false}
real, allocatable:: f(:)[:]
integer, allocatable, codimension[:]:: g(:,:)
```

Remember that when you allocate the array, you must allocate them using the same size. The program will continue without issue, but references to other images' arrays will be misplaced. This is because each image uses the allocated size of its local copy as a template for other images' copies of the variable (from what I can tell - tested with gfortran and opencoarrays).
## Coarrays in subroutines/functions

the codimension property can be used to declare input variables of a subroutine.

Coarrays in subroutines/functions must either be: passed as an argument, defined as allocatable, or with the save attribute to be declared. The save attribute is for when you know the number of images to be used in the subroutine. allocatable must be used otherwise.

## Some notes on performance

`MPI/osh_put` are generally faster than gets, so keep that in mind when writing algorithms. ([source](https://fortran.bcs.org/2017/GnuCoArrays.pdf))

single node performance can be subpar compared to MPI ([Source](https://fortran-lang.discourse.group/t/coarrays-not-ready-for-prime-time/2715))

More to come...