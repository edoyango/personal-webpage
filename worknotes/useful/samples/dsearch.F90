! Module to perform direct search for pairs of points within a fixed cutoff.
! dsearch subroutine
!   dim:      An integer defining dimension of the coordinates.
!   npoints:  An integer with number of points.
!   x:        An real/double precision array of dim x points dimension containing the list of points to find pairs of.
!   cutoff:   The distance (same type as x) which defines a pair
!   maxnpair: An integer of the maximum number of pairs expected.
!   npairs:   An integer with the number of pairs found.
!   pair_i:   An integer array of maxnpairs length with the "left-sided" point in a pair.
!   pair_j:   An integer array of maxnpairs length with the "right-sided" point in a pair.

! dsearch_compact subroutine
!   same as above, except except pair_i is replaced with endpos: a list of endlocations of the left-sided point in a
!   pair. The main program shows an example of how to iterate through along the list.

module dsearch_m

   interface dsearch
      module procedure dsearch_sp, dsearch_dp
   end interface dsearch

   interface dsearch_compact
      module procedure dsearch_compact_sp, dsearch_compact_dp
   end interface dsearch_compact

contains

   !---------------------------------------------------------------------------
   subroutine dsearch_sp(dim, npoints, x, cutoff, maxnpair, npairs, pair_i, pair_j)

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      real, intent(in):: x(dim, npoints), cutoff
      integer, intent(out)::  npairs, pair_i(maxnpair), pair_j(maxnpair)
      integer:: i, j
      real:: xi(dim), r2

      npairs = 0
      do i = 1, npoints - 1
         xi(:) = x(:, i)
         do j = i + 1, npoints
            r2 = sum((xi(:) - x(:, j))**2)
            if (r2 <= cutoff) then
               npairs = npairs + 1
               pair_i(npairs) = i
               pair_j(npairs) = j
            end if
         end do
      end do

   end subroutine dsearch_sp

   !---------------------------------------------------------------------------
   subroutine dsearch_dp(dim, npoints, x, cutoff, maxnpair, npairs, pair_i, pair_j)

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      double precision, intent(in):: x(dim, npoints), cutoff
      integer, intent(out)::  npairs, pair_i(maxnpair), pair_j(maxnpair)
      integer:: i, j
      double precision:: xi(dim), r2

      npairs = 0
      do i = 1, npoints - 1
         xi(:) = x(:, i)
         do j = i + 1, npoints
            r2 = sum((xi(:) - x(:, j))**2)
            if (r2 <= cutoff) then
               npairs = npairs + 1
               pair_i(npairs) = i
               pair_j(npairs) = j
            end if
         end do
      end do

   end subroutine dsearch_dp

   !---------------------------------------------------------------------------
   subroutine dsearch_compact_sp(dim, npoints, x, cutoff, maxnpair, npairs, endpos, pair_j)

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      real, intent(in):: x(dim, npoints), cutoff
      integer, intent(out):: endpos(npoints), npairs, pair_j(maxnpair)
      integer:: i, j
      real:: xi(dim), r2

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
         endpos(i) = npairs
      end do

   end subroutine dsearch_compact_sp

   !---------------------------------------------------------------------------
   subroutine dsearch_compact_dp(dim, npoints, x, cutoff, maxnpair, npairs, endpos, pair_j)

      implicit none
      integer, intent(in):: dim, npoints, maxnpair
      double precision, intent(in):: x(dim, npoints), cutoff
      integer, intent(out):: endpos(npoints), npairs, pair_j(maxnpair)
      integer:: i, j
      double precision:: xi(dim), r2

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
         endpos(i) = npairs
      end do

   end subroutine dsearch_compact_dp

end module dsearch_m

#ifndef NOMAIN
program main

   use dsearch_m, only: dsearch, dsearch_compact

   implicit none
   integer, parameter:: n = 100, dim = 3, maxnpair = 60*n ! estimated using
   ! 2x the coaxial spacing if the points were arranged in a square
   double precision, parameter:: cutoff = 2*n**(-1.d0/dim)
   double precision:: x(dim, n)
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