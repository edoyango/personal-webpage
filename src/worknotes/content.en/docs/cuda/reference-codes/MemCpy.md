---
title: MemCpy
weight: 7
bookToc: false
---

# Memcpy

## Description

Alternatives to array assignment to transfer data between host and device.

Calls to the Memcpy function may be beneifical as transfers by array assignment can be implicitly broken up into multiple transfers, slowing down the transfer.

`cudaMemcpy` is used heaps in the other C++ example codes, so I didn't bother including a sample here.

## Code (Fortran)

```fortran {style=tango,linenos=false}
! for contiguous data
istat = cudaMemcpy(a_d , a_pageable , nElements)

! for 2D data
istat = cudaMemcpy2D(a_d(n1_l , n2_l), n, &
                     a(n1_l , n2_l), n, &
                     n1_u - n1_l + 1, n2_u - n2_l +1)

! there is also cudaMemcpy3D
```