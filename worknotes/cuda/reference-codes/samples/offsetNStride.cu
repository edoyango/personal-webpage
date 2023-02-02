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