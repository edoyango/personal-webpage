---
title: Cell lists pair search without "if"
weight: 5
---

# Cell lists pair search without `if`

## Description

So far, the speedup demonstrated from the [fixed-cutoff cell-lists pair search algorithm](grid-rows-spatial-hashing.md)
is pretty great. One last change we can make to improve things, is to remove any `if` statements when searching adjacent
cells. `if`'s are undesirable because they introduce [branching](https://en.wikipedia.org/wiki/Branch_(computer_science))
and harms performance. I've found that it can be beneficial to remove `if` statements, even if it means a bit more
computation/assignments are performed. 

The below code is the `update_pair_list` function from the code [shown previously](grid-rows-spatial-hashing.md), except
that the function is modified to remove the `if` statement used to check whether the two pairs are within the designated
`cutoff` distance.

The `if` statement is removed by modifying the `pair_i` and `pair_j` arrays to start from index 0. No matter what, 
`pair_i` and `pair_j` are updated at the given `npairs` index. By using the `merge` [Fortran intrinsic function](https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gfortran/MERGE.html),
we only increment `npairs` when the calculated distance is less than the cutoff. Note that the increment of `npairs` now
occurs *after* the index assignments.

The speedup observed with `-O3` compilation is about 1.2x over the [previous version](grid-rows-spatial-hashing.md).

## Code (Fortran)

Use the same `sort.cpp` code as previously and update the `update_pair_list` function as per the code below. Save the
code as `cll4.cpp` and compile with:

```bash {linenos=false,style=tango}
g++ -c sort.cpp
gfortran -o cll4.x cll4.F90 sort.o -lstdc++
```

```fortran {linenos=false,style=tango}

   !---------------------------------------------------------------------------
   subroutine update_pair_list(cutoff, jstart, jend, i, x, npairs, pair_i, pair_j)
      ! subroutine to loop over sequential grid cells and check if j point is
      ! within a given i point

      implicit none
      integer, intent(in):: jstart, jend, i
      real(f), intent(in):: cutoff, x(:, :)
      integer, intent(inout):: npairs, pair_i(0:), pair_j(0:)
      integer:: j
      real(f):: r2

      do j = jstart, jend
         r2 = sum((x(:, i) - x(:, j))**2)
         ! if (r2 <= cutoff*cutoff) then
            ! npairs = npairs + 1
            pair_i(npairs) = i
            pair_j(npairs) = j
            npairs = npairs + merge(1, 0, r2 <= cutoff*cutoff)
         ! end if
      end do

   end subroutine update_pair_list
```