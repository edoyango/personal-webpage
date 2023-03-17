---
title: precision_m
weight: 2
---

# precision_m

## Description

Module containing static types for single and double precision.
Note that `nvfortran` has the options that can be used to demote/promote the precision of real variable/parameter declarations e.g. `-r4` asks the compiler to interpret real declarations as `real(4)`, `-r8` interprets real as `real(8)`, and `-M[no]r8` will promote `real` declarations to `double precision`. `nvc`, `nvcc`, and `nvc++` don't have an equivalent as far as I know.

## Code (C++)
```c++ {style=tango,linenos=false}
#ifndef PRECISION
	#define PRECISION
	// Compile with -DDOUBLE to compile with double precision
	#ifdef DOUBLE
		typedef double userfp_t;
	#else
		typedef float userfp_t;
	#endif
#endif
```

## Code (Fortran)
```fortran {style=tango,linenos=false}
module precision_m
	integer, parameter:: sf = kind(0.)
	integer, parameter:: df = kind(0.d0)

	! compile with -DDOUBLE to compile with double precision
#ifdef DOUBLE
	integer, parameter:: f = df
#else
	integer, parameter:: f = sf
#endif

end module precision_m
```