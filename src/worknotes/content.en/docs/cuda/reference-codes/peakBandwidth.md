---
title: peakBandwidth
weight: 5
---

# peakBandwidth

## Description

Code to obtain theoretical peak memory bandwidth of GPUs on the system.

Effective bandwidth can be obtained with

{{< katex display >}}
bw_e = (r_B + w_B)/(t\cdot10^9)
{{< /katex >}}

where {{< katex >}}bw_e{{< /katex >}} is the effective bandiwdth, {{< katex >}}r_B{{< /katex >}} is the number of Bytes read, {{< katex >}}w_B{{< /katex >}} is the number of Bytes written, and {{< katex >}}t{{< /katex >}} is elapsed wall time in seconds.

The wall time of the simple `memory` kernel written in the [limitingFactor](/worknotes/docs/cuda/reference-codes/limitingFactor) code can be used.

Another note: the three memory bandwidth values we care about are:

* Theoretical: calculated in the code below
* Effective: can be measured with the approach above
* Actual: the realised bandwidth which is affected by memory access patterns

## Code (C++)
```c++ {style=tango,linenos=false}
#include <stdio.h>

int main() {

	int nDevices;
	cudaGetDeviceCount(&nDevices);

	cudaDeviceProp prop;
	for (int i = 0; i < nDevices; ++i) {
		cudaGetDeviceProperties(&prop, i);
		printf("  Device Number: %d\n", i);
		printf("    Memory Clock Rate (kHz): %d\n", prop.memoryClockRate);
		printf("    Memory Bush Width (bits): %d\n", prop.memoryBusWidth);
		printf("    Peak Memory Bandwidth (GB/s): %f\n\n", 
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8)*1.e-6);
	}
}
```

## Code (Fortran)
```fortran {style=tango,linenos=false}
program peakBandwidth

    use cudafor

    implicit none
    integer:: i, istat, nDevices=0
    type(cudaDeviceProp):: prop

    istat = cudaGetDeviceCount(nDevices)
    do i = 0, nDevices-1
        istat = cudaGetDeviceProperties(prop, i)
        write(*,"('  Device Number: ',i0)") i
        write(*,"('    Device name: ',a)") trim(prop%name)
        write(*,"('    Memory Clock Rate (KHz): ', i0)") prop%memoryClockRate
        write(*,"('    Memory Bush Width (bits): ', i0)") prop%memoryBusWidth
        write(*,"('    Peak Memory Bandwidth (GB/s): ', f6.2)") &
            2. *prop%memoryClockRate * (prop%memoryBusWidth/8.) * 1.e-6
        write(*,*)
    end do

end program peakBandwidth
```