#include <stdio.h>

__global__ void base(float *a, float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = sin(b[i]);
}

__global__ void memory(float *a, float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = b[i];
}

__global__ void math(float *a, float b, int flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = sin(b);
    if (v*flag == 1.0) a[i] = v;
}

// this exists because cudaMemSet is weird
__global__ void setval(float *a, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = val;
}

int main() {

    const int n = 8*1024*1024, blockSize = 256;
    
    float *a, *a_d, *b_d;
    a = (float*)malloc(n*sizeof(float));

    cudaMalloc(&a_d, n*sizeof(float));
    cudaMalloc(&b_d, n*sizeof(float));
    setval<<<n/blockSize, blockSize>>>(b_d, 1.0);

    base<<<n/blockSize, blockSize>>>(a_d, b_d);
    memory<<<n/blockSize, blockSize>>>(a_d, b_d);
    math<<<n/blockSize, blockSize>>>(a_d, 1.0, 0);

    cudaMemcpy(a_d, a, n*sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", a[0]);

}