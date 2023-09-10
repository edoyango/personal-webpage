---
title: Direct pair search
weight: 1
bookToc: false
---

# Point-pairs search with fixed cutoff distance (direct)

## Description
I work with point pair searches through particle-based simulations (mostly SPH and DEM). The algorithm here is the most basic way to perform a pair search. It is O(N2) time, so is not useful for any practical applications.

I use it frequently to server as a reference when investigating other ways to search for pairs. It's simple to code, so is harder to introduce conceptual and coding errors.

It does also have some real-world relevance, as the [cell-list pair-search algorithm](/worknotes/docs/useful/fixed-cutoff-cell-lists-pair-search) uses many smaller direct searches.

The Fortran code below contains the module `dsearch_m`, and a `main` program. To compile only the module, pass the `-DNOMAIN` option to the compiler, and the preprocessor will omit it.

The module has the `dsearch` and `dsearch_compact` interfaces. `dsearch` returns two lists, one with the "left-sided" points of the pairs, and `pair_j`, which returns the "right-sided" points in the pair list. In contrast, `dsearch_compact` returns a shorter list, `endpos`, where `endpos(i)` corresponds to the last pair with point `i` as the "left-sided" point. See the `main` program for an example of how to iterate over the lists.

The benefit of `dsearch_compact` is the output format has a smaller memory footprint, and iterating over the list requires fewer memory accesses and is consequently usually a bit faster. However, the drawback is that it is less intuitive and more awkward to iterate over.

## Code (Fortran)
```fortran {linenos=false,style=tango}
! Module to perform direct search for pairs of points within a fixed cutoff.
! dsearch subroutine
!   dim:      An integer defining dimension of the coordinates.
!   npoints:  An integer with number of points.
!   x:        A real/double precision array of dim x points dimension containing the list of points to find pairs of.
!   cutoff:   The distance (same type as x) which defines a pair
!   maxnpair: An integer of the maximum number of pairs expected.
!   npairs:   An integer with the number of pairs found.
!   pair_i:   An integer array of maxnpairs length with the "left-sided" point in a pair.
!   pair_j:   An integer array of maxnpairs length with the "right-sided" point in a pair.

! dsearch_compact subroutine
!   same as above, except except pair_i is replaced with endpos: a list of endlocations of the left-sided point in a
!   pair. The main program shows an example of how to iterate through along the list.

module dsearch_m

   public

#ifdef SINGLEPRECISION
   parameter, integer:: f = kind(1.)
#else
   parameter, integer:: f = kind(1.d0)
#endif

contains

   !---------------------------------------------------------------------------
   subroutine dsearch_dp(dim, npoints, x, cutoff, maxnpair, npairs, pair_i, pair_j)
      ! subroutine to find pairs of points using direct search. Output is 
      ! two lists describing which points are in the pair

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      real(f), intent(in):: x(dim, npoints), cutoff
      integer, intent(out)::  npairs, pair_i(maxnpair), pair_j(maxnpair)
      integer:: i, j
      real(f):: xi(dim), r2

      npairs = 0
      do i = 1, npoints - 1
         xi(:) = x(:, i)
         do j = i + 1, npoints
            r2 = sum((xi(:) - x(:, j))**2)
            if (r2 <= cutoff**2) then
               npairs = npairs + 1
               pair_i(npairs) = i
               pair_j(npairs) = j
            end if
         end do
      end do

   end subroutine dsearch_dp

   !---------------------------------------------------------------------------
   subroutine dsearch_compact_dp(dim, npoints, x, cutoff, maxnpair, npairs, endpos, pair_j)
      ! subroutine to find pairs using direct search. Output is two lists. One
      ! list describes the last pair with point `i` as the "left-sided" point.

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      real(f), intent(in):: x(dim, npoints), cutoff
      integer, intent(out):: endpos(npoints), npairs, pair_j(maxnpair)
      integer:: i, j
      real(f):: xi(dim), r2

      npairs = 0
      do i = 1, npoints
         xi(:) = x(:, i)
         do j = i + 1, npoints
            r2 = sum((xi(:) - x(:, j))**2)
            if (r2 <= cutoff**2) then
               npairs = npairs + 1
               pair_j(npairs) = j
            end if
         end do
         ! update end position when done with point i
         endpos(i) = npairs
      end do

   end subroutine dsearch_compact_dp

end module dsearch_m

#ifndef NOMAIN
program main

   use dsearch_m, only: f, dsearch, dsearch_compact

   implicit none
   integer, parameter:: n = 100, dim = 3, maxnpair = 60*n ! estimated using
   ! 2x the coaxial spacing if the points were arranged in a square
   real(f), parameter:: cutoff = 2*n**(-1.d0/dim)
   real(f):: x(dim, n)
   integer:: pair_i(maxnpair), pair_j(maxnpair), npairs, startpos, i, j, k, endpos(n)

   ! initialize positions with pseudo-random numbers
   call random_number(x)

   ! finding pairs
   call dsearch(dim, n, x, cutoff, maxnpair, npairs, pair_i, pair_j)

   write (*, '(A)') 'Executing "normal" dsearch'
   write (*, '(2x, A,I4,A)') 'Found ', n, ' pairs'
   write (*, '(2x, A)') 'First and last 5 pairs of points found:'
   write (*, '(2x, 4(A4, 1x))') 'Pair', 'i', 'j'
   do k = 1, npairs
      if (k <= 5 .or. k > npairs - 4) write (*, '(2x, 3(I4, 1x))') k, pair_i(k), pair_j(k)
      if (k == 6) write (*, '(2x, A)') '...'
   end do
   write (*, *)
   write (*, '(A)') 'Executing "compact" dsearch'
   write (*, '(2x, A,I4,A)') 'Found ', n, ' pairs'
   write (*, '(2x, A)') 'First and last 5 pairs of points found:'
   write (*, '(2x, 4(A4, 1x))') 'Pair', 'i', 'j'
   call dsearch_compact(dim, n, x, cutoff, maxnpair, npairs, endpos, pair_j)
   do i = 1, n
      if (i == 1) startpos = 1
      if (i > 1) startpos = endpos(i - 1) + 1
      do k = startpos, endpos(i)
         if (k <= 5 .or. k > npairs - 4) write (*, '(2x, 3(I4,1x))') k, i, pair_j(k)
         if (k == 6) write (*, '(2x, A)') '...'
      end do
   end do

end program main
#endif
```