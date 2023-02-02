// This code shows how dynamically and statically allocated
// shared memory are used to reverse a small array
#include <stdio.h>

__global__ void staticReverse(float* d, int n) {
    __shared__ float s[64];
    int t = threadIdx.x, tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

__global__ void dynamicReverse1(float* d, int n) {
    extern __shared__ float s[];
    int t = threadIdx.x, tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

// only one thread can initialize the shared memory buffer
__global__ void dynamicReverse2(float* d, int n) {
    __shared__ float *s ;
    int t = threadIdx.x, tr = n - t - 1;
    if (t==0) s = new float[n];
    __syncthreads();
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
    if (t==0) delete(s);
}

// The Fortran dynamic Reverse3 doesn't have an equivalent in C++

int main() {
    const int n = 64, nbytes = n*sizeof(float);
    float *a, *r, *d, *d_d;
    dim3 grid, tBlock;
    float maxerror;

    a = (float*)malloc(nbytes);
    r = (float*)malloc(nbytes);
    d = (float*)malloc(nbytes);
    cudaMalloc(&d_d, nbytes);

    tBlock = dim3(n, 1, 1);
    grid = dim3(1, 1, 1);

    for (int i = 0; i < n; ++i) {
        a[i] = i;
        r[i] = n-i-1;
    }

    // run version with static shared memory
    cudaMemcpy(d_d, a, nbytes, cudaMemcpyHostToDevice);
    staticReverse<<<grid, tBlock>>>(d_d, n);
    cudaMemcpy(d, d_d, nbytes, cudaMemcpyDeviceToHost);
    maxerror = 0.;
    for (int i = 0; i < n; ++i) {
        if (abs(r[i]-d[i]) > maxerror) maxerror = abs(r[i]-d[i]);
    }
    printf("Static case max error: %f\n", maxerror);

    // run dynamic shared memory version 1
    cudaMemcpy(d_d, a, nbytes, cudaMemcpyHostToDevice);
    dynamicReverse1<<<grid, tBlock>>>(d_d, n);
    cudaMemcpy(d, d_d, nbytes, cudaMemcpyDeviceToHost);
    maxerror = 0.;
    for (int i = 0; i < n; ++i) {
        if (abs(r[i]-d[i]) > maxerror) maxerror = abs(r[i]-d[i]);
    }
    printf("Static case max error: %f\n", maxerror);

    // run dynamic shared memory version 2
    cudaMemcpy(d_d, a, nbytes, cudaMemcpyHostToDevice);
    dynamicReverse2<<<grid, tBlock>>>(d_d, n);
    cudaMemcpy(d, d_d, nbytes, cudaMemcpyDeviceToHost);
    maxerror = 0.;
    for (int i = 0; i < n; ++i) {
        if (abs(r[i]-d[i]) > maxerror) maxerror = abs(r[i]-d[i]);
    }
    printf("Static case max error: %f\n", maxerror);

    free(a);
    free(r);
    free(d);
    cudaFree(d_d);
}