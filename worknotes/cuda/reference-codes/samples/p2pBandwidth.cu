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
        cudaSetDevice(j);                       // sets GPU
        cudaMalloc(&distArray[j].a_d, nbytes);  // allocates device array on the set GPU
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