---
title: grid dimension hashing
weight: 3
---

# Hashing grid cell indices based on grid dimensions

## Description

This strategy is based on [this NVIDIA article](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf).
The idea being that instead of storing indices of particles in a grid data structure, you can convert these 3-valued
indices to single hashes. These hashes can then be used to sort the particle data so that the particle data is ordered
based on their grid cell hash index. This is beneficial for GPUs, which is why it's mentioned in the above article, but
is also useful for CPUs as it iterating over the particle pairs more cache-friendly. 

Read the above article for details. The code makes use of the C++ sort function available in the `algorithm` header.

### Additional notes

I managed to obtain another 1.4x speedup for 1E5 points over [The half-search cell-list pair search](/worknotes/docs/useful/fixed-cutoff-cell-lists-pair-searchhalf-search/) when using no optimisations. A speedup of 4x over the previous version
when compiling all the codes with `-O3` (using total program time, measured roughly with `time`).

The significant speedup obtained is for a few reasons:

* When searching adjacent cells, points in 3 consecutive cells can be searched in a single loop. 
   * This makes the search much more cache-friendly.
* Branching is reduced due to 
   * the removed `if` statement when searching through the cell of interest, and
   * the fewer number of loops.
* In the case of compiling with more aggressive optimisations, the new structure is better able to leverage the compiler's automatic SIMD instructions.

The main downside of the code is that now the points are ordered differently from the input. If the order of the points
is not critical, then it is better to maintain this new order,as data locality would be much better. However, if
necessary, reordering the points can be reversed easily, and with only a modest time penalty.

Another benefit is the lower memory usage by the algorithm. This is because it doesn't need to allocate the
`cells` array which is `maxpercell*ngridx(1)*ngridx(2)*ngridx(3)`. 

## Code (Fortran and C++)

Save the C++ file as `sort.cpp`, and Fortran code as `cll3.F90`. Compile the code with

```bash {linenos=false,style=tango}
g++ -c sort.cpp
gfortran -o cll3.x cll3.F90 sort.o -lstdc++
```

```cpp {linenos=false,style=tango}
// sort.cpp
// C++ source code containing STL sort - modified for sort by indices

#include <algorithm>

extern "C" {
    
    void sort_hashes(int n, int* hashes, int* idx) {

        // filling idx with numbers 0:n-1
        for (int i = 0; i < n; ++i) {
            idx[i] = i;
        }

        // sorting idx based on hashes
        std::sort(idx, idx + n, 
            [&](const int& a, const int& b) {
                return (hashes[a] < hashes[b]);
            }
        );

        // adapting idx for 1-based indexing
        for (int i = 0; i < n; ++i) {
            idx[i]++;
        }
    }

}
```

```fortran {linenos=false,style=tango}
! cll3.F90
! Module to perform direct search for pairs of points within a fixed cutoff.
! cellList subroutine
!   dim:      An integer defining dimension of the coordinates.
!   npoints:  An integer with number of points.
!   maxpercell: An integer with max number of points in a cell.
!   x:        An real/real(f) array of dim x points dimension containing the list of points to find pairs of.
!   cutoff:   The distance (same type as x) which defines a pair
!   maxnpair: An integer of the maximum number of pairs expected.
!   npairs:   An integer with the number of pairs found.
!   pair_i:   An integer array of maxnpairs length with the "left-sided" point in a pair.
!   pair_j:   An integer array of maxnpairs length with the "right-sided" point in a pair.

module cellList_m

   use iso_fortran_env, only: error_unit, real32, real64
   use iso_c_binding

   public

#ifdef SINGLEPRECISION
   integer, parameter:: f = real32
#else
   integer, parameter:: f = real64
#endif

   interface sort_hashes

      subroutine sort_hashes(n, hashes, idx) bind(C)

         use iso_c_binding

         integer(c_int), value, intent(in):: n
         integer(c_int), intent(in):: hashes(*)
         integer(c_int), intent(out):: idx(*)

      end subroutine sort_hashes

   end interface sort_hashes

contains

   !---------------------------------------------------------------------------
   function coordsToHash(n, dim, x, minx, ngridx, cutoff) result(gridhash)
      ! converts raw 2/3D positions to gridhashes based on the grid dimensions
      ! described by minx (lowest grid coordinate), and ngridx (number of grid
      ! cells in xyz direction).

      implicit none
      integer, intent(in):: n, dim, ngridx(3)
      real(f), intent(in):: x(dim, n), minx(dim), cutoff
      integer:: i, icell(3)
      integer(c_int):: gridhash(n)

      if (dim == 2) icell(3) = 1

      do i = 1, n
         icell(1:dim) = int((x(1:dim, i) - minx(1:dim))/cutoff) + 1
         gridhash(i) = ngridx(1)*ngridx(2)*(icell(3) - 1) + ngridx(1)*(icell(2) - 1) + icell(1)
      end do

   end function coordsToHash

   !---------------------------------------------------------------------------
   function hashIncrement(ngridx, dcellx, dcelly, dcellz) result(dhash)
      ! calculates the change in hash, given a change in xyz cell indices

      implicit none
      integer, intent(in):: ngridx(3), dcellx, dcelly, dcellz
      integer:: dhash

      dhash = ngridx(1)*ngridx(2)*dcellx + ngridx(1)*dcelly + dcellz

   end function hashIncrement

   !---------------------------------------------------------------------------
   subroutine rearrange(n, dim, idx, gridhash, x)
      ! rearranges gridhash and x, using the indices obtained from a sort by
      ! key.

      implicit none
      integer, intent(in):: n, dim, idx(n)
      integer(c_int), intent(inout):: gridhash(n)
      real(f), intent(inout):: x(dim, n)
      integer(c_int), allocatable:: tmpgridhash(:)
      real(f), allocatable:: tmpx(:, :)
      integer:: i

      allocate (tmpgridhash(n), tmpx(dim, n))

      do i = 1, n
         tmpgridhash(i) = gridhash(idx(i))
         tmpx(:, i) = x(:, idx(i))
      end do

      gridhash(:) = tmpgridhash(:)
      x(:, :) = tmpx(:, :)

   end subroutine rearrange

   !---------------------------------------------------------------------------
   subroutine find_starts(n, ngridx, gridhash, starts)
      ! finds the starting index for each grid cell. See "Building the grid
      ! using Sorting" section in https://developer.download.nvidia.com/assets/cuda/files/particles.pdf

      implicit none
      integer, intent(in):: n, ngridx(3)
      integer(c_int), intent(in):: gridhash(n)
      integer, intent(out):: starts(:)
      integer:: i

      starts(1:gridhash(1)) = 1

      do i = 2, n
         if (gridhash(i) /= gridhash(i - 1)) starts(gridhash(i - 1) + 1:gridhash(i)) = i
      end do

      starts(gridhash(n) + 1:ngridx(1)*ngridx(2)*ngridx(3)) = n + 1

   end subroutine find_starts

   !---------------------------------------------------------------------------
   subroutine update_pair_list(cutoff, jstart, jend, i, x, npairs, pair_i, pair_j)
      ! subroutine to loop over sequential grid cells and check if j point is
      ! within a given i point

      implicit none
      integer, intent(in):: jstart, jend, i
      real(f), intent(in):: cutoff, x(:, :)
      integer, intent(inout):: npairs, pair_i(:), pair_j(:)
      integer:: j
      real(f):: r2

      do j = jstart, jend
         r2 = sum((x(:, i) - x(:, j))**2)
         if (r2 <= cutoff*cutoff) then
            npairs = npairs + 1
            pair_i(npairs) = i
            pair_j(npairs) = j
         end if
      end do

   end subroutine update_pair_list

   !---------------------------------------------------------------------------
   subroutine cellList(dim, npoints, x, cutoff, maxnpair, npairs, pair_i, pair_j)
      ! main subroutine to perform pair search

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      real(f), intent(in):: cutoff
      real(f), intent(inout):: x(dim, npoints)
      integer, intent(out)::  npairs, pair_i(maxnpair), pair_j(maxnpair)
      integer:: i, hashi, hashj, ngridx(3)
      real(f):: minx(3), maxx(3)
      integer(c_int), allocatable:: gridhash(:), idx(:), starts(:)

      ! determine grid min-max coordinates, with concessions made for dim=2 case
      minx(1:dim) = minval(x, dim=2)
      if (dim == 2) minx(3) = 0.d0
      maxx(1:dim) = maxval(x, dim=2)
      if (dim == 2) maxx(3) = 0.d0
      
      ! creating buffer layers
      minx(:) = minx(:) - 2.d0*cutoff
      maxx(:) = maxx(:) + 2.d0*cutoff

      ! determining no. of grid cells and adjusting maximum extent
      ngridx(:) = int((maxx(:) - minx(:))/cutoff) + 1
      maxx(:) = maxx(:) + ngridx(:)*cutoff

      ! convert coordinates to cell grid hashes
      allocate (gridhash(npoints))
      gridhash = coordsToHash(npoints, dim, x, minx, ngridx, cutoff)

      ! get sorting indices to reorder points by ascending grid hash
      allocate (idx(npoints))
      call sort_hashes(npoints, gridhash, idx)

      ! reorder points (x, gridhash) by ascending grid hash
      call rearrange(npoints, dim, idx, gridhash, x)

      ! find starting points of each grid cell
      allocate (starts(product(ngridx)))
      call find_starts(npoints, ngridx, gridhash, starts)

      ! perform pair search
      npairs = 0
      do hashi = gridhash(1), gridhash(npoints) ! for all relevant grid cell points
         do i = starts(hashi), starts(hashi + 1) - 1 ! loop over all points in the cell
            ! loop over "other" points in cell + next cell (1st adjacent tell: top of hashi cell)
            call update_pair_list(cutoff, i + 1, starts(hashi + 2) - 1, i, x, npairs, pair_i, pair_j)
            ! loop over 2nd-4rd adjacent cells (bottom-south-west to top-south-west)
            hashj = hashi + hashIncrement(ngridx, -1, -1, -1)
            call update_pair_list(cutoff, starts(hashj), starts(hashj + 3) - 1, i, x, npairs, pair_i, pair_j)
            ! loop over 5th-7th adjacent cells (bottom-centre-west to top-centre-west)
            hashj = hashi + hashIncrement(ngridx, -1, 0, -1)
            call update_pair_list(cutoff, starts(hashj), starts(hashj + 3) - 1, i, x, npairs, pair_i, pair_j)
            ! loop over 8th-10th adjacent cells (bottom-north-west to top-north-west)
            hashj = hashi + hashIncrement(ngridx, -1, 1, -1)
            call update_pair_list(cutoff, starts(hashj), starts(hashj + 3) - 1, i, x, npairs, pair_i, pair_j)
            ! loop over 11th-13th adjacent cells (bottom-south-centre to top-south-centre)
            hashj = hashi + hashIncrement(ngridx, 0, -1, -1)
            call update_pair_list(cutoff, starts(hashj), starts(hashj + 3) - 1, i, x, npairs, pair_i, pair_j)
         end do
      end do

   end subroutine cellList

end module cellList_m

#ifndef NOMAIN
program main

   use cellList_m, only: f, cellList

   implicit none
   integer, parameter:: n = 1e2, dim = 3, maxnpair = 60*n ! estimated using
   ! 2x the coaxial spacing if the points were arranged in a square
   real(f), parameter:: cutoff = 2*n**(-1.d0/dim)
   real(f):: x(dim, n)
   integer, allocatable:: pair_i(:), pair_j(:)
   integer:: npairs, k

   allocate (pair_i(maxnpair), pair_j(maxnpair))

   ! initialize positions with pseudo-random numbers
   call random_number(x)

   write (*, '(A)') 'Executing cellList'

   ! finding pairs
   call cellList(dim, n, x, cutoff, maxnpair, npairs, pair_i, pair_j)

   write (*, '(2x, A,I4,A)') 'Found ', npairs, ' pairs'
   write (*, '(2x, A)') 'First and last 5 pairs of points found:'
   write (*, '(2x, 4(A4, 1x))') 'Pair', 'i', 'j'
   do k = 1, npairs
      if (k <= 5 .or. k > npairs - 4) write (*, '(2x, 3(I4, 1x))') k, &
         pair_i(k), pair_j(k)
      if (k == 6) write (*, '(2x, A)') '...'
   end do

end program main
#endif
```