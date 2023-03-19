---
title: limitingFactor
weight: 4
---

# limitingFactor

## Description

Code to test whether computation or memory transfer is the bottleneck. Compiled program intended to be run with `nvprof`.

The book demonstrates the effect of compiling with `-Mcuda=fastmath`, which shows a significant speedup in the "base" and "math" kernels (note they use very old C2050 and K20 GPUs).

## Code (C++)
```c++ {style=tango,linenos=false}
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
```
## Code (Fortran)

```fortran {style=tango,linenos=false}
module kernel_m
contains

    attributes(global) subroutine base(a,b)
        real:: a(*), b(*)
        integer:: i
        i = (blockIdx%x-1) * blockDim%x + threadIdx%x
        a(i) = sin(b(i))
    end subroutine base

    attributes(global) subroutine memory(a,b)
        real:: a(*), b(*)
        integer:: i
        i = (blockIdx%x-1) * blockDim%x + threadIdx%x
        a(i) = b(i)
    end subroutine memory

    attributes(global) subroutine math(a, b, flag)
        real:: a(*)
        real, value:: b
        integer, value:: flag
        real:: v
        integer:: i
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        v = sin(b)
        if (v*flag == 1) a(i) = v
    end subroutine math

end module kernel_m

program limitingFactor

    use cudafor
    use kernel_m 

    implicit none
    integer, parameter:: n=8*1024*1024, blockSize=256
    real:: a(n)
    real, device:: a_d(n), b_d(n)    

    b_d = 1.    

    call base<<<n/blockSize, blockSize>>>(a_d, b_d)
    call memory<<<n/blockSize, blockSize>>>(a_d, b_d)
    call math<<<n/blockSize, blockSize>>>(a_d, 1.0, 0)

    a = a_d

    write(*,*) a(1)
			
end program limitingFactor
```