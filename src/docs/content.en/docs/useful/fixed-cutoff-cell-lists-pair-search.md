---
title: Cell list pair search
weight: 2
bookToc: false
---

# Point-pairs search with fixed cutoff distance (using cell-lists)

## Description
This is an improvement to the [direct search](/worknotes/docs/useful/fixed-cutoff-direct-pair-search/) algorithm for searching for pairs of points within a cutoff distance. It uses a grid of cells whose side-length is equal to the specified cutoff distance.

This is beneficial to performance since for any given point, all its neighbours are guaranteed to be within its own cell, or in adjacent cells. This means that instead of performing a comparison with all other points, comparisons only need to be made to points in the same or adjacent cells.

This results in O(N) time, assuming the grid cells or overall grid size, is resized to suit the number of points.

The below Fortran code is a relatively naive way to implement it. It looks like how someone trying it for the first time might attempt the implementation. Other pages will demonstrate how the algorithm can be sped up.

The main steps are:

1. Determine the min/max extents of the grid (based on point positions).
2. Map points to grid.
3. Sweep through the cells to find pairs.

The interface to the code looks similar to the direct search version, with the addition of a `maxpcell` variable that needs to be added to size the grid array.

## Code (Fortran)
```fortran {linenos=false,style=tango}
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

   use iso_fortran_env, only: error_unit

   public
   private:: coordsToCell

#ifdef SINGLEPRECISION
   parameter, integer:: f = kind(1.)
#else
   parameter, integer:: f = kind(1.d0)
#endif

contains

   !---------------------------------------------------------------------------
   function coordsToCell(dim, xi, minx, cutoff) result(icell)

      implicit none
      real(f), intent(in):: xi(dim), minx(3), cutoff
      integer, intent(in):: dim
      integer:: icell(3)

      icell(1:dim) = int((xi(:) - minx(1:dim))/cutoff) + 1
      if (dim == 2) icell(3) = 1

   end function coordsToCell

   !---------------------------------------------------------------------------
   subroutine cellList(dim, npoints, maxpercell, x, cutoff, maxnpair, npairs, pair_i, pair_j)

      implicit none
      integer, intent(in):: dim, npoints, maxnpair, maxpercell
      real(f), intent(in):: x(dim, npoints), cutoff
      integer, intent(out)::  npairs, pair_i(maxnpair), pair_j(maxnpair)
      integer:: i, j, k, ngridx(3), icell(3), xi, yi, zi, jcell(3)
      real(f):: r2, minx(3), maxx(3)
      integer, allocatable:: pincell(:, :, :), cells(:, :, :, :)

      minx(1:dim) = minval(x, dim=2)
      if (dim == 2) minx(3) = 0.d0
      maxx(1:dim) = maxval(x, dim=2)
      if (dim == 2) maxx(3) = 0.d0
      minx(:) = minx(:) - 2.d0*cutoff
      maxx(:) = maxx(:) + 2.d0*cutoff
      ngridx(:) = int((maxx(:) - minx(:))/cutoff) + 1
      maxx(:) = maxx(:) + ngridx(:)*cutoff

      allocate (pincell(ngridx(1), ngridx(2), ngridx(3)), &
                cells(maxpercell, ngridx(1), ngridx(2), ngridx(3)))

      pincell(:, :, :) = 0

      do i = 1, npoints
         icell = coordsToCell(dim, x(:, i), minx, cutoff)
         pincell(icell(1), icell(2), icell(3)) = &
            pincell(icell(1), icell(2), icell(3)) + 1
#ifdef DEBUG
         if (pincell(icell(1), icell(2), icell(3)) > maxpercell) then
            write (error_unit, '(A, 3(I2, 1x), A)') 'Number of particles in cell ', icell(1), icell(2), icell(3), &
               ' exceeds the input maxpercell value.'
            write (error_unit, '(A)') 'Terminating...'
            error stop 1
         end if
#endif
         cells(pincell(icell(1), icell(2), icell(3)), icell(1), icell(2), icell(3)) = i
      end do

      npairs = 0
      do i = 1, npoints
         icell(:) = coordsToCell(dim, x(:, i), minx, cutoff)
         do xi = -1, 1
            jcell(1) = icell(1) + xi
            do yi = -1, 1
               jcell(2) = icell(2) + yi
               do zi = -1, 1
                  jcell(3) = icell(3) + zi
                  do k = 1, pincell(jcell(1), jcell(2), jcell(3))
                     j = cells(k, jcell(1), jcell(2), jcell(3))
                     if (j > i) then
                        r2 = sum((x(:, i) - x(:, j))**2)
                        if (r2 <= cutoff**2) then
                           npairs = npairs + 1
                           pair_i(npairs) = i
                           pair_j(npairs) = j
                        end if
                     end if
                  end do
               end do
            end do
         end do
      end do

   end subroutine cellList

end module cellList_m

#ifndef NOMAIN
program main

   use cellList_m, only: f, cellList

   implicit none
   integer, parameter:: n = 100, dim = 3, maxnpair = 60*n ! estimated using
   ! 2x the coaxial spacing if the points were arranged in a square
   real(f), parameter:: cutoff = 2*n**(-1.d0/dim)
   real(f):: x(dim, n)
   integer:: pair_i(maxnpair), pair_j(maxnpair), npairs, startpos, i, j, k, endpos(n)

   ! initialize positions with pseudo-random numbers
   call random_number(x)

   ! finding pairs
   call cellList(dim, n, 27, x, cutoff, maxnpair, npairs, pair_i, pair_j)

   write (*, '(A)') 'Executing cellList'
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