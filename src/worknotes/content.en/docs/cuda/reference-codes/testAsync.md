---
title: testAsync
weight: 8
---

# testAsync

## Description

Program to demonstrate the run-time improvements when performing asynchronous memory transfers and execution.

In the book, they show three different batching approaches, which yield different results based on the GPU. On my NVIDIA 1650 (Turing architecture), I see V1 producing best results most of the time. On my workplace's P100 (Keplar), A30 (Ampere), and A100 (Ampere) GPUs, the lowest run times are reliably obtained with V2 (but only about 1% difference).

## Code (C++)
```c++ {style=tango,linenos=false}
#include <stdio.h>

/* this code demonstrates strategies hiding data transfer via asynchronous 
data copies in multiple streams */
__global__ void kernel(float* __restrict__ a, int offset) {
    int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
    float x = i, s = sin(x), c = cos(x);
    a[i] += sqrt(s*s + c*c);
}

int main() {

    const int blockSize = 256, nStreams = 4;
    const int n = 4*1024*blockSize*nStreams, streamSize = n/nStreams;
    size_t nbytes = n*sizeof(float), streamBytes = streamSize*sizeof(float);
    int ierr;
    float *a, *a_d;
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaStream_t stream[nStreams];
    float time, err;
    int offset;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf(" Device: %s\n\n", prop.name);

    // allocate pinned host memory and device memory
    ierr = cudaMallocHost(&a, nbytes);
    if (ierr != 0) {
        printf("Allocation of a failed\n");
        exit(1);
    }
    cudaMalloc(&a_d, nbytes);

    // create events and streams
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    // baseline case - sequential transfer and execute
    memset(a, 0, nbytes);
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(a_d, a, nbytes, cudaMemcpyHostToDevice);
    kernel<<<n/blockSize, blockSize>>>(a_d, 0);
    cudaMemcpy(a, a_d, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Time for sequential transfer and execute (ms): %f\n", time);
    err = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(a[i]-1.0) > err) err = abs(a[i]-1.0);
    }
    printf(" max error: %f\n", err);

    // asynchronous version 1: loop over (copy, kernel, copy)
    memset(a, 0, nbytes);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, 
            cudaMemcpyHostToDevice, stream[i]);
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, 
            cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Time for asynchronous V1 transfer and execute (ms): %f\n", time);
    err = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(a[i]-1.0) > err) err = abs(a[i]-1.0);
    }
    printf(" max error: %f\n", err);

    // asynchronous version 2:
    // loop over copy, loop over kernel, loop over copy
    memset(a, 0, nbytes);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, 
            cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, 
            cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Time for asynchronous V2 transfer and execute (ms): %f\n", time);
    err = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(a[i]-1.0) > err) err = abs(a[i]-1.0);
    }
    printf(" max error: %f\n", err);

    // asynchronous version 3:
    // loop over copy, loop over (kernel, event), loop over copy
    memset(a, 0, nbytes);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, 
            cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
        cudaEventRecord(dummyEvent, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, 
            cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Time for asynchronous V3 transfer and execute (ms): %f\n", time);
    err = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(a[i]-1.0) > err) err = abs(a[i]-1.0);
    }
    printf(" max error: %f\n", err);

    cudaFree(a_d);
    free(a);
}
```

## Code (Fortran)
```fortran {style=tango,linenos=false}
! This code demonstrates strategies hiding data transfers via
! asynchronous data copies in multiple streams
module kernels_m
contains

    attributes(global) subroutine kernel(a, offset)
        implicit none
        real:: a(*)
        integer, value:: offset
        integer:: i
        real:: c, s, x
    
        i = offset + threadIdx%x + (blockIdx%x-1)* blockDim%x
        x = i; s = sin(x); c = cos(x)
        a(i) = a(i) + sqrt(s**2+c**2)
    end subroutine kernel

end module kernels_m

program testAsync

    use cudafor
    use kernels_m

    implicit none
    integer, parameter:: blockSize = 256, nStreams = 4
    integer, parameter:: n = 4*1024* blockSize*nStreams
    real, pinned, allocatable:: a(:)
    real, device:: a_d(n)
    integer(kind=cuda_stream_kind):: stream(nStreams)
    type(cudaEvent) :: startEvent, stopEvent, dummyEvent
    real:: time
    integer:: i, istat, offset, streamSize = n/nStreams
    logical:: pinnedFlag
    type(cudaDeviceProp):: prop

    istat = cudaGetDeviceProperties(prop , 0)
    write(*,"(' Device: ', a,/)") trim(prop%name)

    ! allocate pinned host memory
    allocate(a(n), STAT=istat , PINNED=pinnedFlag)
    if (istat /= 0) then
        write(*,*) 'Allocation of a failed'
        stop
    else
        if (.not. pinnedFlag) write(*,*) 'Pinned allocation failed'
    end if

    ! create events and streams
    istat = cudaEventCreate(startEvent)
    istat = cudaEventCreate(stopEvent)
    istat = cudaEventCreate(dummyEvent)
    do i = 1, nStreams
        istat = cudaStreamCreate(stream(i))
    end do

    ! baseline case - sequential transfer and execute
    a = 0
    istat = cudaEventRecord(startEvent, 0)
    a_d = a
    call kernel <<<n/blockSize, blockSize >>>(a_d, 0)
    a = a_d
    istat = cudaEventRecord(stopEvent , 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    write(*,*) 'Time for sequential ', 'transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.0))

    ! asynchronous version 1: loop over {copy , kernel , copy}
    a = 0
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, nStreams
        offset = (i-1)* streamSize
        istat = cudaMemcpyAsync(a_d(offset+1), a(offset+1), streamSize, &
            stream(i))
        call kernel <<<streamSize/blockSize, blockSize, 0, stream(i)>>>(a_d, &
            offset)
        istat = cudaMemcpyAsync(a(offset+1), a_d(offset+1), streamSize, &
            stream(i))
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time , startEvent , stopEvent)
    write(*,*) 'Time for asynchronous V1 ', 'transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.))

    ! asynchronous version 2:
    ! loop over copy , loop over kernel , loop over copy
    a = 0
    istat = cudaEventRecord(startEvent, 0)
    do i = 1, nStreams
        offset = (i-1)*streamSize
        istat = cudaMemcpyAsync(a_d(offset+1), a(offset+1), streamSize, &
            stream(i))
    end do
    do i = 1, nStreams
        offset = (i-1)*streamSize
        call kernel <<<streamSize/blockSize, blockSize, 0, stream(i)>>>(a_d, &
            offset)
    end do
    do i = 1, nStreams
        offset = (i-1)*streamSize
        istat = cudaMemcpyAsync (a(offset +1), a_d(offset+1), streamSize, &
            stream(i))
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time , startEvent , stopEvent)
    write(*,*) 'Time for asynchronous V2 ', 'transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.))

    ! asynchronous version 3:
    ! loop over copy , loop over {kernel, event},
    ! loop over copy
    a = 0
    istat = cudaEventRecord(startEvent ,0)
    do i = 1, nStreams
        offset = (i-1)* streamSize
        istat = cudaMemcpyAsync(a_d(offset+1), a(offset+1), streamSize, &
            stream(i))
    end do
    do i = 1, nStreams
        offset = (i-1)* streamSize
        call kernel <<<streamSize/blockSize, blockSize, 0, stream(i)>>>(a_d, &
            offset)
        istat = cudaEventRecord(dummyEvent, stream(i))
    end do
    do i = 1, nStreams
        offset = (i-1)* streamSize
        istat = cudaMemcpyAsync(a(offset+1), a_d(offset+1), streamSize, &
            stream(i))
    end do
    istat = cudaEventRecord(stopEvent, 0)
    istat = cudaEventSynchronize(stopEvent)
    istat = cudaEventElapsedTime(time, startEvent, stopEvent)
    write(*,*) 'Time for asynchronous V3 ', 'transfer and execute (ms): ', time
    write(*,*) ' max error: ', maxval(abs(a-1.))

    ! cleanup
    istat = cudaEventDestroy(startEvent)
    istat = cudaEventDestroy(stopEvent)
    istat = cudaEventDestroy(dummyEvent)
    do i = 1, nStreams
        istat = cudaStreamDestroy(stream(i))
    end do
    deallocate(a)

end program testAsync
```