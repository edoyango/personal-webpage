---
title: directTransfer
weight: 13
---

# directTransfer

## Description

Demonstration code showing the difference in data transfer rates when transfering directly between peer GPUs, versus without p2p transfer.

Running this on the GPUs at work, showed the transfer between GPUs with p2p disabled was maybe 5% slower, but this wasn't produced reliably. The differnece was not nearly as dramatic as those in the book.

## Code (C++)
```
To do...
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
program directTransfer

    use cudafor

    implicit none
    integer, parameter:: N = 4*1024*1024
    real, pinned, allocatable:: a(:), b(:)
    real, device, allocatable:: a_d(:), b_d(:)

    ! these hold free and total memory before and after
    ! allocation, used to verify allocation is happening
    ! on proper devices
    integer(int_ptr_kind()), allocatable:: freeBefore(:), totalBefore(:), freeAfter(:), totalAfter(:)
    integer:: istat, nDevices, i, accessPeer, timingDev
    type(cudaDeviceProp):: prop
    type(cudaEvent):: startEvent, stopEvent
    real:: time

    ! allocate host arrays
    allocate(a(N), b(N))
    allocate(freeBefore(0: nDevices-1), totalBefore(0: nDevices-1))
    allocate(freeAfter(0: nDevices-1), totalAfter(0: nDevices-1))

    ! get device info(including total and free memory)
    ! before allocating a_d and b_d on devices 0 and 1
    istat = cudaGetDeviceCount(nDevices)
    if(nDevices < 2) then
        write(*,*) 'Need at least two CUDA capable devices '
        stop
    end if
    write(*,"('Number of CUDA -capable devices: ', i0,/)") nDevices
    do i = 0, nDevices-1
        istat = cudaGetDeviceProperties(prop, i)
        istat = cudaSetDevice(i)
        istat = cudaMemGetInfo(freeBefore(i), totalBefore(i))
    end do
    istat = cudaSetDevice(0)
    allocate(a_d(N))
    istat = cudaSetDevice(1)
    allocate(b_d(N))

    ! print out free memory before and after allocation
    write(*,"('Allocation summary ')")
    do i = 0, nDevices-1
        istat = cudaGetDeviceProperties(prop, i)
        write(*,"(' Device ', i0, ': ', a)") i, trim(prop%name)
        istat = cudaSetDevice(i)
        istat = cudaMemGetInfo(freeAfter(i), totalAfter(i))
        write(*,"(' Free memory before: ', i0, ', after: ', i0, ', difference: ',i0,/)") &
            freeBefore(i), freeAfter(i), freeBefore(i)-freeAfter(i)
    end do

    ! check whether devices 0 and 1 can use P2P
    if(nDevices > 1) then
        istat = cudaDeviceCanAccessPeer(accessPeer, 0, 1)
        if(accessPeer == 1) then
            write(*,*) 'Peer access available between 0 and 1'
        else
            write(*,*) 'Peer access not available between 0 and 1'
        end if
    end if

    ! initialize
    a = 1.0
    istat = cudaSetDevice(0)
    a_d = a
    ! perform test twice, timing on both sending GPU
    ! and receiving GPU
    do timingDev = 0, 1
        write(*,"(/,'Timing on device ', i0, /)") timingDev

        ! create events on the timing device
        istat = cudaSetDevice(timingDev)
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)

        if(accessPeer == 1) then
            ! enable P2P communication
            istat = cudaSetDevice(0)
            istat = cudaDeviceEnablePeerAccess(1, 0)
            istat = cudaSetDevice(1)
            istat = cudaDeviceEnablePeerAccess(0, 0)

            ! transfer(implicitly) across devices
            b_d = -1.0
            istat = cudaSetDevice(timingDev)
            istat = cudaEventRecord(startEvent,0)
            b_d = a_d
            istat = cudaEventRecord(stopEvent,0)
            istat = cudaEventSynchronize(stopEvent)
            istat = cudaEventElapsedTime(time, startEvent, stopEvent)
            b = b_d
            if(any(b /= a)) then
                write(*,"('Transfer failed ')")
            else
                write(*,"('b_d=a_d transfer(GB/s): ', f)") N*4/time/1.0E+6
            end if
        end if

        ! transfer via cudaMemcpyPeer()
        if(accessPeer == 0) istat = cudaSetDevice(1)
        b_d = -1.0

        istat = cudaSetDevice(timingDev)
        istat = cudaEventRecord(startEvent,0)
        istat = cudaMemcpyPeer(b_d, 1, a_d, 0, N)
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        if(accessPeer == 0) istat = cudaSetDevice(1)
        b = b_d
        if(any(b /= a)) then
            write(*,"('Transfer failed ')")
        else
            write(*,"('cudaMemcpyPeer transfer(GB/s): ', f)") N*4/time/1.0E+6
        end if

        ! cudaMemcpyPeer with P2P disabled
        if(accessPeer == 1) then
            istat = cudaSetDevice(0)
            istat = cudaDeviceDisablePeerAccess(1)
            istat = cudaSetDevice(1)
            istat = cudaDeviceDisablePeerAccess(0)
            b_d = -1.0
            istat = cudaSetDevice(timingDev)
            istat = cudaEventRecord(startEvent,0)
            istat = cudaMemcpyPeer(b_d, 1, a_d, 0, N)
            istat = cudaEventRecord(stopEvent,0)
            istat = cudaEventSynchronize(stopEvent)
            istat = cudaEventElapsedTime(time, startEvent, stopEvent)
            istat = cudaSetDevice(1)
            b = b_d
            if(any(b /= a)) then
                write(*,"('Transfer failed ')")
            else
                write(*,"('cudaMemcpyPeer transfer w/ P2P ', ' disabled(GB/s): ', f)") N*4/time/1.0E+6
            end if
        end if
        
        ! destroy events associated with timingDev
        istat = cudaEventDestroy(startEvent)
        istat = cudaEventDestroy(stopEvent)
    end do

    ! clean up
    deallocate(freeBefore, totalBefore, freeAfter, totalAfter)
    deallocate(a, b, a_d, b_d)

end program directTransfer
```