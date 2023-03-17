---
title: Error Handling
weight: 3
bookToc: false
---

# Error Handling

## Description

Basic functionality to check errors in CUDA functions and kernel subroutines.

The C++ code is very similar to the Fortran code, so I'm not including it.

## Code (Fortran)
```fortran {style=tango,linenos=false}
! the cuda GetErrorString function can be used to obtain error messages from error codes
ierr = cudaGetDeviceCount(nDevices)
if (ierr/= cudaSuccess) write (*,*) cudaGetErrorString(ierr)

! kernel errors are checked using cudaGetLastError
call increment <<<1,n>>>(a_d , b)
ierrSync = cudaGetLastError()
ierrAsync = cudaDeviceSynchronize()
if (ierrSync /= cudaSuccess) write(*,*) 'Sync kernel error', cudaGetErrorString(ierrSync)
if (ierrAsync /= cudaSuccess) write(*,*) 'Async kernel error:', cudaGetErrorString(cudaGetLastError())
```