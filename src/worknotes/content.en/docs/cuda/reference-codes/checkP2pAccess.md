---
title: checkP2PAccess
weight: 12
---

# checkP2pAccess

## Description

Tool to check peer-to-peer connectivity between GPUs connected to the motherboard.

## Code (C++)

```c++ {style=tango,linenos=false}
#include <stdio.h>

int main() {

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Number of CUDA-capable devices: %d\n", nDevices);

    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }

    int p2pOK[nDevices][nDevices];
    for (int i = 0; i < nDevices; ++i) {
        for (int j = i+1; j < nDevices; ++j) {
            cudaDeviceCanAccessPeer(&p2pOK[i][j], i, j);
            p2pOK[j][i] = p2pOK[i][j];
        }
    }
    printf("\n");

    for (int i = 0; i < nDevices; ++i) {
        printf("   %3d", i);
    }
    printf("\n");

    for (int j = 0; j < nDevices; ++j) {
        printf("%3d",j);
        for (int i = 0; i < nDevices; ++i) {
            if (i==j) {
                printf("  -   ");
            } else if (p2pOK[i][j] == 1) {
                printf("  Y   ");
            } else {
                printf("      ");
            }
        }
        printf("\n");
    }
}
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
program checkP2pAccess

    use cudafor

    implicit none
    integer, allocatable:: p2pOK(:,:)
    integer:: nDevices, i, j, istat
    type (cudaDeviceProp):: prop

    istat = cudaGetDeviceCount(nDevices)
    write(*,"('Number of CUDA -capable devices: ', i0,/)") nDevices

    do i = 0, nDevices -1
        istat = cudaGetDeviceProperties(prop, i)
        write(*,"('Device ', i0, ': ', a)") i, trim(prop%name)
    end do
    write(*,*)

    allocate(p2pOK (0: nDevices -1, 0: nDevices -1))
    p2pOK = 0

    do j = 0, nDevices -1
        do i = j+1, nDevices -1
            istat = cudaDeviceCanAccessPeer(p2pOK(i,j), i, j)
            p2pOK(j,i) = p2pOK(i,j)
        end do
    end do

    do i = 0, nDevices -1
        write(*,"(3x,i3)", advance='no') i
    end do
    write(*,*)

    do j = 0, nDevices -1
        write(*,"(i3)", advance='no') j
        do i = 0, nDevices -1
            if (i == j) then
            write(*,"(2x,'-',3x)", advance='no')
            else if (p2pOK(i,j) == 1) then
            write(*,"(2x, 'Y',3x)",advance='no')
            else
            write(*,"(6x)",advance='no')
            end if
        end do
        write(*,*)
    end do

end program checkP2pAccess
```
