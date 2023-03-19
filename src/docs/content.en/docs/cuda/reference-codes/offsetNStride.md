---
title: offsetNStride
weight: 9
---

# offsetNStride

## Description

Demonstration of how coalesced access to GPU global memory i.e., accessing memory in strides of 16 (half-warp) or 32 (warp) can reduce the number of transactions made and reduce run times.

My NVIDIA 1650 behaves like K20 and C2050 used in the book, where 0-stride accesses are fastest, and everything else is worse (by only a little). 

## Code (C++)

```c++ {style=tango,linenos=false}
#include <stdio.h>
#include <precision.h>

__global__ void offset(userfp_t* a, int s) {
    int i = blockDim.x*blockIdx.x + threadIdx.x + s;
    a[i] += 1.0;
}

__global__ void stride(userfp_t* a, int s) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x)*s;
    a[i] += 1.0;
}

int main() {
    
    const int nMB = 4, n = nMB*1024*1024/sizeof(userfp_t), blockSize = 256;
    const size_t nbytes = n*sizeof(userfp_t);
    userfp_t *a_d, *b_d;
    cudaDeviceProp prop;
    cudaEvent_t startEvent, stopEvent;
    float time;

    // array dimensions are 33*n for stride cases
    cudaMalloc(&a_d, 33*nbytes);
    cudaMalloc(&b_d, 33*nbytes);

    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", nMB);

    if (sizeof(userfp_t)==sizeof(double)) {
        printf("Double Precision\n");
    } else {
        printf("Single Precision\n");
    }

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("Offset, Bandwidth (GB/s):\n");
    offset<<<n/blockSize, blockSize>>>(b_d, 0);
    for (int i = 0; i < 33; ++i) {
        cudaMemset(&a_d, 0, 33*nbytes);
        cudaEventRecord(startEvent, 0);
        offset<<<n/blockSize, blockSize>>>(a_d, i);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("%d %f\n", i, 2.0*nbytes/time*1.e-6);
    }

    printf("\nStrid, Bandwidth (GB/s):\n");
    stride<<<n/blockSize, blockSize>>>(b_d, 1);
    for (int i = 1; i < 33; ++i) {
        cudaMemset(&a_d, 0, 33*nbytes);
        cudaEventRecord(startEvent, 0);
        stride<<<n/blockSize, blockSize>>>(a_d, i);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("%d %f\n", i, 2.0*nbytes/time*1.e-6);
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
module kernels_m

    use precision_m

contains

    attributes(global) subroutine offset(a, s)

        real(f):: a(*)
        integer, value:: s
        integer:: i
        i = blockDim%x*(blockIdx%x-1)+ threadIdx%x + s
        a(i) = a(i) + 1._f

    end subroutine offset

    attributes(global) subroutine stride(a, s)

        real(f):: a(*)
        integer, value:: s
        integer:: i
        i = (blockDim%x*(blockIdx%x-1)+ threadIdx%x) * s
        a(i) = a(i) + 1._f

    end subroutine stride

end module kernels_m

program offsetNStride

    use cudafor
    use kernels_m

    implicit none
    integer, parameter:: nMB = 4 ! transfer size in MB
    integer, parameter:: n = nMB*1024*1024/f
    integer, parameter:: blockSize = 256

    ! array dimensions are 33*n for stride cases
    real(f), device:: a_d(33*n), b_d(33*n)
    type(cudaEvent):: startEvent, stopEvent
    type(cudaDeviceProp):: prop
    integer:: i, istat
    real:: time

    istat = cudaGetDeviceProperties(prop, 0)
    write(*,'(/," Device: ",a)') trim(prop%name)
    write(*,'(" Transfer size (MB): ",i0)') nMB
    if (kind(a_d) == sf) then
        write(*,'(a,/)') 'Single Precision '
    else
        write(*,'(a,/)') 'Double Precision '
    end if

    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)

    write(*,*) 'Offset, Bandwidth (GB/s):'
    call offset <<<n/blockSize,blockSize >>>(b_d, 0)
    do i = 0, 32
        a_d = 0._f
        istat = cudaEventRecord(startEvent,0)
        call offset <<<n/blockSize,blockSize >>>(a_d, i)
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) i, 2*n*f/time *1.e-6
    end do

    write(*,*)
    write(*,*) 'Stride, Bandwidth (GB/s):'
    call stride <<<n/blockSize,blockSize >>>(b_d, 1)
    do i = 1, 32
        a_d = 0._f
        istat = cudaEventRecord(startEvent,0)
        call stride <<<n/blockSize,blockSize >>>(a_d, i)
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        istat = cudaEventElapsedTime(time, startEvent, stopEvent)
        write(*,*) i, 2*n*f/time*1.e-6
    end do

    istat = cudaEventDestroy(startEvent)
    istat = cudaEventDestroy(stopEvent)

end program offsetNStride
```
