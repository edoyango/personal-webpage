---
title: p2pBandwidth
weight: 14
---

# p2pBandwidth

## Description

Program to measure the bandwidth of memory transfer between a GPUs on a multi-GPU host.

A useful technique shown here, is the use of derived types with device member arrays to manage each GPU's instance of the device arrays.

## Code (C++)

```c++ {style=tango,linenos=false}
#include <stdio.h>
#include <cstdlib>

struct distributedArray {
    float* a_d;
};

__global__ void setVal(float* __restrict__ array, float val) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    array[i] = val;

}

int main() {

    const int N = 4*1024*1024;
    const size_t nbytes = N*sizeof(float);
    
    // displaying number of GPUs
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    // displaying GPU names
    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }
    printf("\n");
    
    distributedArray distArray[nDevices];

    // enable p2p access between GPUs (if possible)
    int access;
    for (int j = 0; j < nDevices; ++j) {
        // sets GPU
        cudaSetDevice(j);
        // allocate device array on the set GPU
        cudaMalloc(&distArray[j].a_d, nbytes);  
        setVal<<<N/1024, 1024>>>(distArray[j].a_d, j);
        for (int i = j+1; i < nDevices; ++i) {
            cudaDeviceCanAccessPeer(&access, j, i);
            if (access == 1) {
                cudaSetDevice(j);
                cudaDeviceEnablePeerAccess(i, 0);
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // allocating array to record inter-GPU bandwidths
    float bandwidth[nDevices][nDevices];
    for (int i = 0; i < nDevices; ++i) {
        for (int j = 0; j < nDevices; ++j) {
            bandwidth[i][j] = 0.0;
        }
    }

    // measure p2p bandwidth between each pair of GPUs
    cudaEvent_t startEvent, stopEvent;
    float time;
    float *array = (float*)malloc(nbytes);
    
    for (int j = 0; j < nDevices; ++j) {
        cudaSetDevice(j);
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        for (int i = 0; i < nDevices; ++i) {
            if (i == j) continue;
            cudaEventRecord(startEvent, 0);
            cudaMemcpyPeer(distArray[j].a_d, j, distArray[i].a_d, i, nbytes);
            cudaEventRecord(stopEvent, 0);
            cudaEventSynchronize(stopEvent);
            cudaEventElapsedTime(&time, startEvent, stopEvent);
            cudaMemcpy(array, distArray[j].a_d, nbytes, cudaMemcpyDeviceToHost);
            for (int k = 0; k < N; ++k) {
                if (array[k] != i) {
                    printf("Transfer between GPUs %d and %d failed!\n", j, i);
                    std::exit(1);
                }
            }
            bandwidth[j][i] = N*sizeof(float)/time/1.0e6;
        }
        setVal<<<N/1024, 1024>>>(distArray[j].a_d, j);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    printf("Bandwidth (GB/s) for transfer size (MB): %f\n", nbytes/1024.0/1024.0);
    printf(" S\\R   0");
    for (int i = 1; i < nDevices; ++i) printf("     %3d", i);
    printf("\n");
    for (int j = 0; j < nDevices; ++j) {
        printf("%3d", j);
        for (int i = 0; i < nDevices; ++i) {
            if (i==j) {
                printf("    0   ");
            } else {
                printf("%8.2f", bandwidth[j][i]);
            }
        }
        printf("\n");
    }

    // cleanup
    for (int j = 0; j < nDevices; ++j) {
        cudaFree(distArray[j].a_d);
    }

}
```

## Code (Fortran)
```fortran {style=tango,linenos=false}
program p2pBandwidth

    use cudafor

    implicit none
    integer, parameter:: N = 4*1024*1024
    type distributedArray
        real, device, allocatable:: a_d(:)
    end type distributedArray
    type(distributedArray), allocatable:: distArray(:)

    real, allocatable:: bandwidth(:,:)
    real:: array(N), time
    integer:: nDevices, access, i, j, istat
    type(cudaDeviceProp):: prop
    type(cudaEvent):: startEvent, stopEvent

    ! displaying number of GPUs
    istat = cudaGetDeviceCount(nDevices)
    write(*,"('Number of CUDA-capable devices: ', i0,/)") nDevices

    ! displaying GPU names
    do i = 0, nDevices-1
        istat = cudaGetDeviceProperties(prop, i)
        write(*,"('Device ', i0, ': ', a)") i, trim(prop%name)
    end do
    write(*,*)

    ! creating device array for each GPU
    allocate(distArray(0:nDevices-1))

    ! enable p2p access between GPUs (if possible)
    do j = 0, nDevices-1
        istat = cudaSetDevice(j)           ! sets GPU
        allocate(distArray(j)%a_d(N))      ! allocates device array on the set GPU
        distArray(j)%a_d = j
        do i = j+1, nDevices-1
            istat = cudaDeviceCanAccessPeer(access, j, i)
            if (access == 1) then
                istat = cudaSetDevice(j)
                istat = cudaDeviceEnablePeerAccess(i, 0)
                istat = cudaSetDevice(i)
                istat = cudaDeviceEnablePeerAccess(j, 0)
            end if
        end do
    end do

    ! allocating array to record inter-GPU bandwidths
    allocate(bandwidth(0:nDevices-1, 0:nDevices-1))
    bandwidth(:,:) = 0.

    ! measure p2p bandwidth between each pair of GPUs
    do j = 0, nDevices-1
        istat = cudaSetDevice(j)
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)
        do i = 0, nDevices-1
            if (i == j) cycle
            istat = cudaMemcpyPeer(distArray(j)%a_d, j, distArray(i)%a_d, i, N)
            istat = cudaEventRecord(startEvent, 0)
            istat = cudaMemcpyPeer(distArray(j)%a_d, j, distArray(i)%a_d, i, N)
            istat = cudaEventRecord(stopEvent, 0)
            istat = cudaEventSynchronize(stopEvent)
            istat = cudaEventElapsedTime(time, startEvent, stopEvent)

            array = distArray(j)%a_d
            if (all(array == i)) bandwidth(j,i) = N*4/time/1.0E+6
        end do
        distArray(j)%a_d = j
        istat = cudaEventDestroy(startEvent)
        istat = cudaEventDestroy(stopEvent)
    end do

    write(*,"('Bandwidth (GB/s) for transfer size (MB): ', f9.3,/)") N*4./1024**2
    write(*,"(' S\\R   0')", advance='no')
    do i = 1, nDevices-1
        write(*,"(5x,i3)", advance='no') i
    end do
    write(*,*)

    do j = 0, nDevices-1
        write(*,"(i3)", advance='no') j
        do i = 0, nDevices-1
            if (i==j) then
                write(*,"(4x,'0',3x)", advance='no')
            else
                write(*,"(f8.2)", advance='no') bandwidth(j,i)
            end if
        end do
        write(*,*)
    end do

    ! cleanup
    do j = 0, nDevices-1
        deallocate(distArray(j)%a_d)
    end do
    deallocate(distArray, bandwidth)

end program p2pBandwidth
```