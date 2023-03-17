---
title: sharedExample
weight: 11
---

# sharedExample

## Description

A sample code to demonstrate how the compiler uses various types of memory. This information is availed when compiling with `-Mcuda=ptxinfo`.

## Code (C++)

```c++ {style=tango,linenos=false}
/* This code shows how dynamically and statically allocated
 shared memory are used to reverse a small array */
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
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
! This code shows how dynamically and statically allocated
! shared memory are used to reverse a small array
module reverse_m

	implicit none
	integer, device:: n_d
	
contains

	attributes(global) subroutine staticReverse(d)

		real:: d(:)
		integer:: t, tr
		real, shared:: s(64)

		t = threadIdx%x
		tr = size(d)-t+1

		s(t) = d(t)
		call syncthreads()
		d(t) = s(tr)

	end subroutine staticReverse

	attributes(global) subroutine dynamicReverse1(d)

		real:: d(:)
		integer:: t, tr
		real, shared:: s(*)

		t = threadIdx%x
		tr = size(d)-t+1

		s(t) = d(t)
		call syncthreads()
		d(t) = s(tr)

	end subroutine dynamicReverse1

	attributes(global) subroutine dynamicReverse2(d, nSize)

		real:: d(nSize)
		integer, value:: nSize
		integer:: t, tr
		real, shared:: s(nSize)

		t = threadIdx%x
		tr = nSize -t+1

		s(t) = d(t)
		call syncthreads()
		d(t) = s(tr)

	end subroutine dynamicReverse2

	attributes(global) subroutine dynamicReverse3(d)

		real:: d(n_d)
		real, shared:: s(n_d)
		integer:: t, tr

		t = threadIdx%x
		tr = n_d -t+1

		s(t) = d(t)
		call syncthreads()
		d(t) = s(tr)

	end subroutine dynamicReverse3

end module reverse_m

program sharedExample

	use cudafor
	use reverse_m

	implicit none
	integer, parameter:: n = 64
	real:: a(n), r(n), d(n)
	real, device:: d_d(n)
	type(dim3):: grid, tBlock
	integer:: i, sizeInBytes

	tBlock = dim3(n,1,1)
	grid = dim3 (1,1,1)

	do i = 1, n
		a(i) = i
		r(i) = n-i+1
	enddo

	sizeInBytes = sizeof(a(1))* tBlock%x

	! run version with static shared memory
	d_d = a
	call staticReverse<<<grid,tBlock>>>(d_d)
	d = d_d
	write(*,*) 'Static case max error:', maxval(abs(r-d))

	! run dynamic shared memory version 1
	d_d = a
	call dynamicReverse1<<<grid,tBlock,sizeInBytes>>>(d_d)
	d = d_d
	write(*,*) 'Dynamic case 1 max error:', maxval(abs(r-d))

	! run dynamic shared memory version 2
	d_d = a
	call dynamicReverse2<<<grid,tBlock,sizeInBytes>>>(d_d,n)
	d = d_d
	write(*,*) 'Dynamic case 2 max error:', maxval(abs(r-d))

	! run dynamic shared memory version 3
	n_d = n ! n_d declared in reverse_m
	d_d = a
	call dynamicReverse3<<<grid,tBlock,sizeInBytes>>>(d_d)
	d = d_d
	write(*,*) 'Dynamic case 3 max error:', maxval(abs(r-d))

end program sharedExample
```
