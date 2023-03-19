---
title: deviceQuery
weight: 1
---

# deviceQuery

## Description

Function to query the properties of the NVIDIA GPUs detected on the system.

## Code (C++)
```c++ {linenos=false,style=tango}
#include <stdio.h>

int main() {

    int nDevices;

    cudaGetDeviceCount(&nDevices);

    if (nDevices == 0) {
        printf("No CUDA devices found\n");
    } else if (nDevices == 1) {
        printf("One CUDA device found\n");
    } else {
        printf("%d CUDA devices found\n", nDevices);
    }

    // Loop over devices and print properties
    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; ++i) {

        printf("Device Number: %d\n", i);

        cudaGetDeviceProperties(&prop, i);

        // General device info
        printf("  Device Name: %s\n");
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Multiprocessor: %d\n", 
            prop.maxThreadsPerMultiProcessor);
        printf("  Global Memory (GB): %f\n\n", 
            prop.totalGlobalMem/(1024.0*1024.0*1024.0));

        // Execution Configuration
        printf("  Execution Configuration Limits\n");
        printf("    Max Grid Dims: %d x %d x %d\n", prop.maxGridSize[0], 
            prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("    Max Block Dims: %d x %d x %d\n", prop.maxThreadsDim[0], 
            prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("    Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

    }

    return 0;
}
```

## Code (Fortran)
```fortran {style=tango,linenos=false}
program deviceQuery

    use cudafor
    
    implicit none
    type(cudaDeviceProp):: prop
    integer:: nDevices=0, i, ierr

    ! Number of CUDA-capable devices
    ierr = cudaGetDeviceCount(nDevices)

    if (nDevices == 0) then
        write(*,"(/, 'No CUDA devices found',/)")
        stop
    else if (nDevices == 1) then
        write(*,"(/,'One CUDA device found',/)")
    else
        write(*,"(/,i0,' CUDA devices found',/)") nDevices
    end if

    ! Loop over devices and print properties
    do i = 0, nDevices-1

        write(*,"('Device Number: ',i0)") i

        ierr = cudaGetDeviceProperties(prop, i)

        ! General device info
        write(*,"('  Device Name: ',a)") trim(prop%name)
        write(*,"('  Compute Capability: ',i0,'.',i0)") prop%major, &
            prop%minor
        write(*,"('  Number of Multiprocessors: ',i0)") prop%multiProcessorCount
        write(*,"('  Max Threads per Multiprocessor: ',i0)") &
            prop%maxThreadsPerMultiprocessor
        write(*,"('  Global Memory (GB): ',f9.3,/)") &
            prop%totalGlobalMem/1024.**3

        ! Execution Configuration
        write(*,"('  Execution Configuration Limits')") 
        write(*,"('    Max Grid Dims: ',2(i0,' x '),i0)") prop%maxGridSize
        write(*,"('    Max Block Dims: ',2(i0,' x '),i0)") prop%maxThreadsDim
        write(*,"('    Max Threads per Block: ',i0,/)") prop%maxThreadsPerBlock

    end do

end program deviceQuery
```