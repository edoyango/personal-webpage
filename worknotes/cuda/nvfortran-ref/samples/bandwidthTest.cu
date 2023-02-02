#include <stdio.h>
#include <cstdlib>

int main() {

    const int nElements = 4*1024*1024;
    size_t nbytes = nElements*sizeof(float);
    float *a_pageable, *b_pageable;
    float *a_pinned, *b_pinned;
    float *a_d;
    int ierr_a, ierr_b;
    cudaDeviceProp prop;
    cudaEvent_t startEvent, stopEvent;
    float time = 0.0;

    // pageable host memory
    a_pageable = (float*)malloc(nbytes);
    b_pageable = (float*)malloc(nbytes);

    // pinned host memory
    ierr_a = cudaMallocHost((void**)&a_pinned, nbytes);
    ierr_b = cudaMallocHost((void**)&b_pinned, nbytes);
    if (ierr_a != 0 || ierr_b != 0) {
        printf("Allocation of a_pinned/b_pinned failed\n");
        std::exit(1);
    }

    // initializing
    for (int i = 0; i < nElements; ++i) a_pageable[i] = i;
    memcpy(a_pinned, a_pageable, nbytes);
    memset(b_pageable, 0.0, nbytes);
    memset(b_pinned, 0.0, nbytes);

    // device memory
    cudaMalloc((void**)&a_d, nbytes);

    // output device info and transfer size
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %f\n", nbytes/1024./1024.);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // pageable data transfers
    printf("\nPageable transfers\n");
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(a_d, a_pageable, nbytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Host to Device bandwidth (GB/s): %f\n", nbytes/time/1.e+6);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(b_pageable, a_d, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Device to Host bandwidth (GB/s): %f\n", nbytes/time/1.e+6);

    for (int i = 0; i < nElements; ++i) {
        if (a_pageable[i] != b_pageable[i]) {
            printf("*** Pageable transfers failed ***\n");
            break;
        }
    }

    cudaMemset(a_d, 0.0, nbytes);

    // pinned data transfers
    printf("\nPinned transfers\n");
    cudaEventRecord(startEvent, 0);
    cudaMemcpy(a_d, a_pinned, nbytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Host to Device bandwidth (GB/s): %f\n", nbytes/time/1.e+6);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(b_pinned, a_d, nbytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("  Device to Host bandwidth (GB/s): %f\n", nbytes/time/1.e+6);

    for (int i = 0; i < nElements; ++i) {
        if (a_pinned[i] != a_pinned[i]) {
            printf("*** Pinned transfers failed ***\n");
            break;
        }
    }

    printf("\n");

    // cleanup
    cudaFree(a_d);
    cudaFreeHost(a_pinned);
    cudaFreeHost(b_pinned);
    free(a_pageable);
    free(b_pageable);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

}