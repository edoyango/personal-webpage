---
title: mpiDevices
weight: 15
---

# mpiDevices

## Description

Code to get started with using MPI with nvfortran. All it does is check compute mode:

* default (0): multiple host threads can use a single GPU
* exclusive (1): one host thread can use a single GPU at a time
* prohibited (2): No host threads can use the GPU
* exclusive process (3): Single contect cna be created by a single process, but that process can be current to all threads of that process.

GPUs in default mode will allow for multiple MPI processes to be assigned to a single GPU, whereas exclusive and exclusive process will only allow one MPI process per GPU.

## Code (C++)

``` {style=tango,linenos=false}
To do...
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
program mpiDevices

    use cudafor
    use mpi

    implicit none
    ! global array size
    integer, parameter:: n = 1024*1024
    ! MPI variables
    integer:: procid, numprocs, ierr
    ! device
    type(cudaDeviceProp):: prop
    integer(int_ptr_kind()):: freeB, totalB, freeA, totalA
    real, device, allocatable:: d(:)
    integer:: i, j, istat, devid

    ! MPI initialization
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, procid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    ! print compute mode for device
    istat = cudaGetDevice(devid)
    istat = cudaGetDeviceProperties(prop, devid)

    do i = 1, numprocs
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (procid == i) write(*, &
            "('[',i0,'] using device: ', i0, ' in compute mode: ', i0)") &
            procid, devid, prop%computeMode
    end do

    ! get memory use before large allocations
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    istat = cudaMemGetInfo(freeB, totalB)

    ! now allocate arrays, one rank at a time
    do j = 0, numprocs-1

        ! allocate on device associated with rank j
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (procid == j) allocate(d(n))

        ! Get free memory after allocation
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        istat = cudaMemGetInfo(freeA, totalA)

        write(*, "(' [',i0,'] after allocation on rank: ', i0, &
            ', device arrays allocated: ', i0)") &
            procid, devid, (freeB-freeA)/n/4

    end do

    deallocate(d)

    call MPI_FINALIZE(ierr)

end program mpiDevices
```