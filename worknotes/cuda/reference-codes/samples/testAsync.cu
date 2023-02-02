#include <stdio.h>

// this code demonstrates strategies hiding data transfer via asynchronous data copies in multiple streams
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
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
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
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
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
        cudaMemcpyAsync(&a_d[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a_d, offset);
        cudaEventRecord(dummyEvent, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i) {
        offset = i*streamSize;
        cudaMemcpyAsync(&a[offset], &a_d[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
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