---
title: Speeding Up LIGGGHTS with Intel Compilers and Compiler Options
weight: 2
---

# Speeding Up LIGGGHTS with Intel Compilers and Compiler Options

This page looks at using basic optimization options and [Intel OneAPI Compilers](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.ndxx16) (for x86 CPU architectures) to reduce the run-time of LIGGGHTS.

The page will first show the difference in performance of the Intel compilers compared to the GNU compilers and also looks at different compiler options. After hopefully convincing you of why you should use the intel compilers, the page then goes on to explain how to build LIGGGHTS with the Intel compilers.

For those unaware, GNU compilers are shipped with most Linux systems, including Ubuntu. You can invoke the GNU C compiler with `gcc`.

Everything on this page is run on the following platform:

* An Inspiron 7501 with
    * i7-10750H CPU
    * 24GB (16GB + 8GB) @ 2933MHz RAM
    * Windows 10
* WSL2 Ubuntu 20.04

The tests use the `chute-wear` example files in `LIGGGHTS-PUBLIC/examples/LIGGGHTS/Tutorials_public/chute_wear`.

For this investigation, I'm using the VTK and OpenMPI packages from APT, which give me VTK v6.3.0 and OpenMPI v4.0.3.

## Compilers' and Compiler Options' Performance

### The compiler options to investigate
The `-O2` optimisation option that the `Makefile.mpi` file specifies is a default level of optimisation that GNU and Intel compilers use. On the other hand, the `-funroll-loops` is an extra optimisation that [unrolls loops](https://en.wikipedia.org/wiki/Loop_unrolling) which may improve speed of the compiled program.

Here, we're going to look at the following options:

* `-O3`, which is increasing the optimisation level from default. This isn't guaranteed to improve run-times.
* `-march=native`/`-xhost`, which, respectively, ask the GNU and Intel compilers to compile code that can better take advantage of the CPU architecture on the computer.
* `-flto`/`-ipo`, which, respectively, ask the GNU and Intel compilers to perform interprocedural optimization, which may improve speed of the compiled program.

I don't consider using options like `-Ofast` or `-ffast-math` because they can reduce the accuracy of numerical operations, which is not ideal, as time-integration simulations' results can be sensitive to numerical accuracy.

### The Tests

| Test No. | Compiler Options |
| --- | --- |
| 1 | `CCFLAGS = -O2 -funroll-loops -fPIC`, `LINKFLAGS = -O2 -fPIC` |
| 2 | `CCFLAGS = -O2 -fPIC`, `LINKFLAGS = -O2 -fPIC` |
| 3 | `CCFLAGS = -O3 -fPIC`, `LINKFLAGS = -O3 -fPIC` |
| 4 | `CCFLAGS = -O3 -fPIC -march=native (-xhost)`, `LINKFLAGS = -O3 -fPIC -march=native (-xhost)` |
| 5 | `CCFLAGS = -O3 -fPIC -flto (-ipo)`, `LINKFLAGS -O3 -fPIC -flto (-ipo)` |

Where the options in brackets are the equivalent option for the Intel Compilers. Each test is run 5 times and the average is reported in the results below.

### Results

![liggghts-compiler-options-compare-1](/worknotes/imgs/liggghts-compiler-options-compare-1.png)

*The average (of 5 runs) run-time of LIGGGHTS chute-wear example, when LIGGGHTS is built with either GNU and Intel compilers and with different compiler options. The runs where the LIGGGHTS executable was built with the `-ipo` Intel Compiler option failed, so no results reported.*

### Discussion
The Intel compiler performed better in all tests, where the fastest average run-time of the Intel Compiler tests ran 2s (~11%) faster than the fastest average run-time of the GNU Compiler tests.

The Intel compilers meaningfully benefitted from the `-xhost` compiler option, whereas the GNU compiler didn't benefit from its equivalent (`-march=native`). In fact, it was slower! I should probably look into it further.

I failed to make use of the Intel `-ipo` option, which I would like to work as this option has been quite useful in my Fortran simulation code. I suspect it's because the VTK libraries are dynamic, not static.

The `-O3` option didn't really make a difference, and the loop unrolling seemed to slow down the LIGGGHTS code. However, I need to see if both of those effects are observed with larger simulation (the `chute-wear` example uses only ~800 concurrent DEM particles).

## Getting the Intel compilers
The Intel OneAPI compilers are free for anybody to use. For LIGGGHTS, only the base kit is necessary.

Once you have installed the compilers by following Intel's instructions, you will need to make sure you have the VTK and OpenMPI APT packages.

## Modifying the make File to use the Intel compiler

The basic [LIGGGHTS installation documentation](https://www.cfdem.com/media/DEM/docu/Section_start.html#start-2-2) will suggest users to build LIGGGHTS using the `make auto` command. But, here, we'll be using the `make mpi` command, which uses the `Makefile.mpi` Make file. LIGGGHTS assumes we're using the default APT packages, so we have to modify the Makefile. Edit `LIGGGHTS-PUBLIC/src/MAKE/Makefile.mpi` so that the following lines have been modified:

```
CC = icpx # this is the Intel OneAPI C++ compiler
LINK = icpx

# These tell Make where your OpenMPI libraries and include files are. You may need to modify them to suit you.
MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_PATH = -L/usr/lib/x86_64-linux-gnu/openmpi/lib
MPI_LIB = -lmpi

# These tell Make where your VTK libraries and include files. You may need to modify them to suit you. I had to change the default 6.2 to 6.3 to reflect the version of the APT package.
VTK_INC =  -I/usr/include/vtk-6.3
VTK_PATH = -L/usr/lib/x86_64-linux-gnu
VTK_LIB = -lvtkCommonCore-6.3 -lvtkIOCore-6.3 -lvtkIOXML-6.3 -lvtkIOLegacy-6.3 -lvtkCommonDataModel-6.3 -lvtkIOParallel-6.3 -lvtkParallelCore-6.3 -lvtkParallelMPI-6.3 -lvtkIOImage-6.3 -lvtkCommonExecutionModel-6.3 -lvtkFiltersCore-6.3 -lvtkIOParallelXML-6.3
```

If you've modified the Make file correctly for your system, the `make mpi` command run inside the `src` directory should build the `lmp_mpi` executable.
Changing

## Changing the LIGGGHTS default compiler options
The default `make auto` command uses some default compiler options. While you can add to the default compiler options by editing the `Makefile.user` file, it's less straightforward to change the compiler options. To change them, you will want to edit the `LIGGGHTS_PUBLIC/src/MAKE/Makefile.mpi` file and use the `make mpi` command instead. At the time of writing, the default compiler and linker options are:

```
CCFLAGS = -O2 -funroll-loops -fstrict-aliasing -Wall -Wno-unused-result -fPIC
LINKFLAGS = -O2 -fPIC
```

Note that `-O2` and `-funroll-loops` are the optimisation options, and the rest are compiler warnings, although `-fPIC` is neither, and is telling the compiler to produce [Position Independent Code](https://en.wikipedia.org/wiki/Position-independent_code).

To change the compiler options, change the code following the `CCFLAGS =` or `LINKFLAGS =`. E.g.,

```
CCFLAGS = -O3 -xhost -fPIC
LINKFLAGS = -O3 -xhost -fPIC
```

would be the best of the options [looked at on this page](/worknotes/docs/cfdem/liggghts-intel-comp/#results). The meaning of the above changes are explained [above](/worknotes/docs/cfdem/liggghts-intel-comp/#the-compiler-options-to-investigate).