#include <stdio.h>

int main() {

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("  Device Number: %d\n", i);
        printf("    Memory Clock Rate (kHz): %d\n", prop.memoryClockRate);
        printf("    Memory Bush Width (bits): %d\n", prop.memoryBusWidth);
        printf("    Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8)*1.e-6);
    }
}