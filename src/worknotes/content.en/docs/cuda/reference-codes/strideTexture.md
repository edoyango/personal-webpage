---
title: strideTexture
weight: 10
---

# strideTexture

## Description

A demonstration showing how the use of textured memory pointers can improve strided global memory access.

I find that using textured memory pointers didn't improve anything reliably on my NVIDIA 1650.

The deprecation is also why a C++ version is not provided here.

## Code (Fortran)

```fortran {style=tango,linenos=false}
module kernels_m

    real, texture, pointer:: aTex (:)

contains

    attributes(global) subroutine stride(b, a, s)

        real:: b(*), a(*)
        integer, value:: s
        integer:: i, is
        i = blockDim%x*( blockIdx%x-1)+ threadIdx%x
        is = (blockDim%x*( blockIdx%x-1)+ threadIdx%x) * s
        b(i) = a(is)+1

    end subroutine stride

    attributes(global) subroutine strideTex(b, s)

        real:: b(*)
        integer, value:: s
        integer:: i, is
        i = blockDim%x*( blockIdx%x-1)+ threadIdx%x
        is = (blockDim%x*( blockIdx%x-1)+ threadIdx%x) * s
        b(i) = aTex(is)+1

    end subroutine strideTex

end module kernels_m

program strideTexture

    use cudafor
    use kernels_m

    implicit none
    integer, parameter:: nMB = 4 ! transfer size in MB
    integer, parameter:: n = nMB *1024*1024/4
    integer, parameter:: blockSize = 256
    real, device, allocatable, target:: a_d(:), b_d (:)
    type(cudaEvent):: startEvent, stopEvent
    type(cudaDeviceProp):: prop
    integer:: i, istat, ib
    real:: time

    istat = cudaGetDeviceProperties(prop, 0)
    write(*,'(/," Device: ",a)') trim(prop%name)
    write(*,'(" Transfer size (MB): ",i0,/)') nMB

    allocate(a_d(n*33), b_d(n))

    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)

    write(*,*) 'Global version '
    write(*,*) 'Stride, Bandwidth (GB/s)'
    call stride<<<n/blockSize,blockSize>>>(b_d, a_d, 1)
    do i = 1, 32
        a_d = 0.0
        istat = cudaEventRecord(startEvent,0)
        call stride<<<n/blockSize, blockSize>>>(b_d, a_d, i)
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) i, 2*n*4/time*1.e-6
    enddo

    ! bind the texture
    aTex => a_d

    write(*,*) 'Texture version '
    write(*,*) 'Stride, Bandwidth (GB/s)'
    call strideTex<<<n/blockSize,blockSize>>>(b_d, 1)
    do i = 1, 32
        a_d = 0.0
        istat = cudaEventRecord(startEvent,0)
        call strideTex<<<n/blockSize,blockSize>>>(b_d, i)
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) i, 2*n*4/time*1.e-6
    enddo

    ! unbind the texture
    nullify(aTex)
    istat = cudaEventDestroy(startEvent)
    istat = cudaEventDestroy(stopEvent)
    deallocate(a_d, b_d)

end program strideTexture
```