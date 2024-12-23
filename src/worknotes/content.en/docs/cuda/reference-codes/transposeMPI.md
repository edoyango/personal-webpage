---
title: transposeMPI
weight: 17
---

# transposeMPI

## Description

Code showing the use of tiling and shared memory in transposing a matrix. The book uses it as an example of the performance difference between CUDA aware MPI vs non-MPI transfers (transfers between GPUs via their respective host CPUs). The code will work on GPUs communicating across node boundaries.

## Code (C++)

``` {style=tango,linenos=false}
To do...
```

## Code (Fortran)

```fortran {style=tango,linenos=false}
module transpose_m

    implicit none
    integer, parameter:: cudaTileDim = 32
    integer, parameter:: blockRows = 8

contains

    attributes(global) subroutine cudaTranspose(odata, ldo, idata, ldi)

        real, intent(out):: odata(ldo, *)
        real, intent(in):: idata(ldi, *)
        integer, value, intent(in):: ldo, ldi
        real, shared:: tile(cudaTileDim+1, cudaTileDim)
        integer:: x, y, j

        x = (blockIdx%x-1) * cudaTileDim + threadIdx%x
        y = (blockIdx%y-1) * cudaTileDim + threadIdx%y

        do j = 0, cudaTileDim-1, blockRows
            tile(threadIdx%x, threadIdx%y+j) = idata(x, y+j)
        end do

        call syncthreads()

        x = (blockIdx%y-1) * cudaTileDim + threadIdx%x
        y = (blockIdx%x-1) * cudaTileDim + threadIdx%y

        do j = 0, cudaTileDim-1, blockRows
            odata(x, y+j) = tile(threadIdx%y+j, threadIdx%x)
        end do

    end subroutine

end module transpose_m

!
! Main code
!

program transpose_MPI

    use cudafor
    use mpi
    use transpose_m
    use mpiDeviceUtil

    implicit none

    ! global array size
    integer, parameter:: nx = 2048, ny = 2048

    ! host arrays (global)
    real:: idata_h(nx, ny), tdata_h(ny, nx), gold(ny, nx)

    ! CUDA vars and device arrays
    integer:: deviceID, nDevices
    type(dim3):: dimGrid, dimBlock
    real, device, allocatable:: idata_d(:, :), tdata_d(:, :), sTile_d(:, :), &
        rTile_d(:, :)

    ! MPI stuff
    integer:: mpiTileDimX, mpiTileDimY
    integer:: procid, numprocs, tag, ierr, localRank
    integer:: nstages, stage, sRank, rRank
    integer:: status(MPI_STATUS_SIZE)
    double precision:: timeStart, timeStop
    character(len=10):: localRankStr

    integer:: i, j, nyl, jl, jg, p
    integer:: xOffset, yOffset

    ! MPI initialization
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, procid, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

    ierr = cudaGetDeviceCount(nDevices)

    ! check parameters and calculate execution configuration
    if (mod(nx, numprocs) == 0 .and. mod(ny, numprocs) == 0) then
        mpiTileDimX = nx/numprocs
        mpiTileDimY = ny/numprocs
    else
        write(*,*) "ny must be an integral multiple of numprocs"
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if

    if (mod(mpiTileDimX, cudaTileDim) /= 0 .or. mod(mpiTileDimY, cudaTileDim) &
        /= 0) then
        write(*,*) "mpiTileDimX and mpiTileDimY must be an integral multiple", &
            " of cudaTileDim"
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if

    if (numprocs > nDevices) then
        write(*,*) "numprcs must be <= number of GPUs on the system!"
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if

    ! each MPI process 
    call assignDevice(procid, numprocs, deviceID)

    dimGrid = dim3(mpiTileDimX/cudaTileDim, mpiTileDimY/cudaTileDim, 1)
    dimBlock = dim3(cudaTileDim, blockRows, 1)

    ! write parameters to terminal
    if (procid == 0) then
        write(*,*)
        write(*,"('Array size: ', i0,' x ',i0,/)") nx, ny
        write(*,"('CUDA block size: ', i0,' x ',i0,',  CUDA tile size: ', &
            i0,' x ',i0,/)") cudaTileDim, blockRows, cudaTileDim, cudaTileDim

        write(*,"('dimGrid: ', i0,' x ',i0,' x ',i0,'  dimBlock: ', i0,&
            ' x ',i0,' x ',i0,/)") dimGrid%x, dimGrid%y, dimGrid%x, &
            dimBlock%x, dimBlock%y, dimBlock%z

        write(*,"('numprocs: ', i0, ',  Local input array size: ', &
            i0,' x ',i0,/)") numprocs, nx, mpiTileDimY

        write(*,"('mpiTileDim: ', i0,' x ',i0,/)") mpiTileDimX, mpiTileDimY
    end if

    ! initalize data
    ! host - each process has entire array on host (for now)
    do p = 0, numprocs-1
        do jl = 1, mpiTileDimY
            jg = p*mpiTileDimY + jl
            do i = 1, nx
                idata_h(i, jg) = i + (jg-1)*nx
            end do
        end do
    end do

    gold = transpose(idata_h)

    ! device - each process has nx*mpiTileDimY = ny*mpiTileDimX elements
    allocate( idata_d(nx, mpiTileDimY), &
        tdata_d(ny, mpiTileDimX), &
        sTile_d(mpiTileDimX, mpiTileDimY), &
        rTile_d(mpiTileDimX, mpiTileDimY))

    yOffset = procid*mpiTileDimY
    idata_d(1:nx, 1:mpiTileDimY) = idata_h(1:nx, yOffset+1:yOffset+mpiTileDimY)
    tdata_d = -1.0

    ! ---------
    ! transpose
    ! ---------
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    timeStart = MPI_WTIME()

    ! 0th stage - local transpose
    call cudaTranspose<<<dimGrid, dimBlock>>>&
        (tdata_d(procid*mpiTileDimY+1, 1), ny, &
        idata_d(procid*mpiTileDimX+1, 1), nx)

    ! other stages that involve MPI transfers
    do stage = 1, numprocs-1
        ! sRank = the rank to which procid send data to
        ! rRank = the rank from which myrank receives data
        sRank = modulo(procid-stage, numprocs)
        rRank = modulo(procid+stage, numprocs)

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)

        ! pack tile so data to be sent is contiguous

        !$cuf kernel do (2) <<<*, *>>>
        do j = 1, mpiTileDimY
            do i = 1, mpiTileDimX
                sTile_d(i, j) = idata_d(sRank*mpiTileDimX+i, j)
            end do
        end do

        call MPI_SENDRECV(sTile_d, mpiTileDimX*mpiTileDimY, MPI_REAL, sRank, &
            procid, rTile_d, mpiTileDimX*mpiTileDimY, MPI_REAL, rRank, rRank, &
            MPI_COMM_WORLD, status, ierr)

        ! do transpose from receive tile into final array
        ! (no need to unpack)
        call cudaTranspose<<<dimGrid, dimBlock>>>&
            (tdata_d(rRank*mpiTileDimY+1, 1), ny, rTile_d, mpiTileDimX)

    end do

    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
    timeStop = MPI_WTIME()

    ! check results
    tdata_h = tdata_d

    xOffset = procid*mpiTileDimX
    if (all(tdata_h(1:ny, 1:mpiTileDimX) == &
        gold(1:ny, xOffset+1:xOffset+mpiTileDimX))) then
        if (procid == 0) then
            write(*,"('Bandwidth (GB/s): ', f7.2,/)") 2.*(nx*ny*4)/&
                (1.e9*(timeStop-timeStart))
        end if
    else
        write(*,"('[',i0,']', '*** Failed ***')") procid
    end if

    deallocate(idata_d, tdata_d, sTile_d, rTile_d)

    call MPI_FINALIZE(ierr)

end program transpose_MPI
```