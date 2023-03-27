---
title: Getting Started
weight: 1
---

# Getting Started

This page will help you get setup and started to run the GraSPH code and is aimed at beginners. The instructions assume you are working on an Ubuntu machine with x86_64 CPU architecture. a Virtual Machine (VM) or WSL2 will work fine too. If you're using a different OS or CPU architecture, you will need modify the package install commands and paths used in the compilation setup. This page assumes you are working in a terminal.

Steps are seperated based on which version you wish to run. See the table of contents on the right side of the page.

## Getting the GraSPH code

Clone the GraSPH repo with
```
git clone https://github.com/edoyango/GraSPH.git
cd GraSPH
```

## For running in serial

### Prerequisites
To run the serial version of the code, all you need is a working Fortran compiler and the HDF5 libraries. A good compiler to start with is `gfortran`.

```
sudo apt-get install gfortran libhdf5-dev
```

### Setting up compilation environment
```
export FCFLAGS=-I/usr/include/hdf5/serial
export LDFLAGS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/serial:$LD_LIBRARY_PATH
export FC=gfortran
```

### Running the GraSPH code
```
cd GraSPH
make mode=serial -f makefiles/Makefile.caf
./sph 10000 1000 1000
```

## For running on multiple CPUs

### Prerequisites
If you wish to run GraSPH in parallel with CPUs, you will need an MPI implementation, the HDF5 MPI libraries, as well as the OpenCoarrays library. All of these are available via APT. Below commands use the OpenMPI implementation, but other MPI implementations should work fine too.

```
sudo apt-get install openmpi-bin libopenmpi-dev libhdf5-dev libhdf5-openmpi-dev libcoarrays-openmpi-dev
```

### Setting up the compilation environment
```
export FCFLAGS="-I/usr/include/hdf5/openmpi -I/usr/lib/x86_64-linux-gnu/fortran/gfortran-mod-15"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5 /usr/lib/x86_64-linux-gnu/open-coarrays/openmpi/lib/libcaf_mpi.a"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/openmpi:$LD_LIBRARY_PATH
export FC=mpifort
```

### Running the GraSPH code
```
cd GraSPH
make mode=caf -f makefiles/Makefile.caf
mpiexec -n <ncpus> sph 10000 1000 1000
```

## For running with GPUs

### Prerequisites

#### CUDA drivers and toolkit
You will need to install the CUDA drivers and toolkit. To get the installer, follow the instructions on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

#### Compiler
To run Fortran code on CUDA GPUs, you will have to use `nvfortran`, NVIDIA's Fortran compiler available through the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk). You will then have to use this compiler to build the HDF5 libraries.

#### Build HDF5 with NVHPC via Spack
Spack is my preferred tool to help automate building software. To get Spack:
```
git clone https://github.com/spack/spack.git
```
which clones the repo into the `spack` directory in the current working directory. Once cloned, setup the environment with
```
source spack/share/spack/setup-env.sh
```
You can then add the `nvhpc` compilers to Spack with
```
spack compiler add /opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers
```
For example,
```
$ spack compiler add /opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers
==> Added 1 new compiler to /home/edwardy/.spack/linux/compilers.yaml
    nvhpc@23.1
==> Compilers are defined in the following files:
    /home/edwardy/.spack/linux/compilers.yaml
```
You can then build hdf5 with the command
```
spack install hdf5+fortran%nvhpc
```
which is saying "build the `hdf5` package with Fortran libraries using the `nvhpc` compilers.

### Setting up the compilation environment
```
export HDF5_ROOT=`spack location -i hdf5%nvhpc`
export FCFLAGS=-I$HDF5_ROOT/include
export LDFLAGS=-L$HDF5_ROOT/lib
export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH
export FC=nvfortran
```

### Running the GraSPH code
```
cd GraSPH
make -f makefiles/Makefile.cuda
./sph 10000 1000 1000
```

## Example Video
This is run with for 10000 steps printing every 50 steps, running in `caf` mode.
![Dambreak.gif](/docs/imgs/Dambreak.gif)