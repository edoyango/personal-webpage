---
title: mpiDeviceUtil
weight: 16
---

# mpiDeviceUtil

## Description

Basic module to assign MPI processes to unique GPUs. This is modified from the code in the book to use `MPI_ALLGATHER`, an explicit definition of the `quicksort` subroutine, and the use of `hostnm` Fortran intrinsic function instead of the `MPI_GET_PROCESSOR_NAME`.

## Code (C++)

``` {style=tango,linenos=false}
To do...
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
module mpiDeviceUtil
contains

    subroutine assignDevice(procid, numprocs, dev)

        use mpi
        use cudafor

        implicit none
        integer:: numprocs, procid, dev
        character(len=100), allocatable:: hosts(:)
        character(len=100):: hostname
        integer:: namelength, color, i
        integer:: newComm, newProcid, ierr
        logical:: mpiInitialized

        ! allocate array of hostnames
        allocate(hosts(0:numprocs-1))

        ! every process collects the hostname of all the nodes
        call hostnm(hostname)

        call MPI_ALLGATHER(hostname, 100, MPI_CHARACTER, hosts, 100, &
            MPI_CHARACTER, MPI_COMM_WORLD, ierr)

        call MPI_BARRIER(MPI_COMM_WORLD,ierr)

        ! sort the list of names
        call quicksort(hosts, 100, 1, numprocs)

        call MPI_BARRIER(MPI_COMM_WORLD,ierr)

        ! assign the same color to the same node
        color = 0
        do i = 0, numprocs-1
            if (i > 0) then
                if ( hosts(i-1) /= hosts(i) ) color = color + 1
            end if
            if ( hostname <= hosts(i) ) exit
        end do

        call MPI_COMM_SPLIT(MPI_COMM_WORLD, color, 0, newComm, ierr)
        call MPI_COMM_RANK(newComm, newProcid, ierr)

        dev = newProcid
        ierr = cudaSetDevice(dev)

        deallocate(hosts)

    end subroutine assignDevice

    ! quicksort.f -*-f90-*-
    ! Author: t-nissie
    ! License: GPLv3
    ! Gist: https://gist.github.com/t-nissie/479f0f16966925fa29ea
    !!
    recursive subroutine quicksort(a, strlen, first, last)
    
        implicit none
        integer, intent(in):: strlen, first, last
        character(strlen), intent(inout):: a(*)
        character(strlen)::  x, t
        integer:: i, j

        x = a( (first+last) / 2 )
        i = first
        j = last
        do
           do while (a(i) < x)
              i=i+1
           end do
           do while (x < a(j))
              j=j-1
           end do
           if (i >= j) exit
           t = a(i);  a(i) = a(j);  a(j) = t
           i=i+1
           j=j-1
        end do
        if (first < i-1) call quicksort(a, strlen, first, i-1)
        if (j+1 < last)  call quicksort(a, strlen, j+1, last)

    end subroutine quicksort

end module mpiDeviceUtil

program main

    use mpi
    use cudafor
    use mpiDeviceUtil

    implicit none
    integer, parameter:: n = 1024*1024
    ! mpi
    character(len=100):: hostname
    integer:: procid, numprocs, ierr, namelength
    ! device
    type(cudaDeviceProp):: prop
    integer(int_ptr_kind()):: freeB, totalB, freeA, totalA
    real, device, allocatable:: d(:)
    integer:: deviceID, i, istat

    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, procid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    ! get and set unique device
    call assignDevice(procid, numprocs, deviceID)

    ! print hostname and device ID for each rank
    call hostnm(hostname)
    do i = 0, numprocs-1
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (i == procid) write(*, &
            "('[',i0,'] host: ', a, ',  device: ', i0)") &
            procid, trim(hostname), deviceID
    end do

    ! get memory use before large allocations
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    istat = cudaMemGetInfo(freeB, totalB)

    ! allocate memory on each device
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    allocate(d(n))

    ! get free memory after allocation
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    istat = cudaMemGetInfo(freeA, totalA)

    do i = 0, numprocs-1
        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        if (i == procid) write(*, &
            "('  [', i0, '] ', 'device arrays allocated: ', i0)") &
            procid, (freeB-freeA)/n/4
    end do

    deallocate(d)

    call MPI_FINALIZE(ierr)

end program main
```
