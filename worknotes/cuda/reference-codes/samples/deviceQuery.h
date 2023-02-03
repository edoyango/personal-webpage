#ifndef DEVICEQUERY
#define DEVICEQUERY
#include <stdio.h>

int deviceQuery() {

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
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Global Memory (GB): %f\n\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));

        // Execution Configuration
        printf("  Execution Configuration Limits\n");
        printf("    Max Grid Dims: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], 
            prop.maxGridSize[2]);
        printf("    Max Block Dims: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
            prop.maxThreadsDim[2]);
        printf("    Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

    }

    return 0;
}
#endif