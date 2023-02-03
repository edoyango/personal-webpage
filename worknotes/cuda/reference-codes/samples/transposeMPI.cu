#include <stdio.h>
#include <mpiDeviceUtil.h>

const int cudaTileDim = 32;
const int blockRows = 8;

__global__ void cudaTranspose(float* __restrict__ odata, 
    const int ldo, 
    const float* __restrict__ idata, 
    const int ldi) {

    if (threadIdx.x==0 & threadIdx.y==0) __shared__ float tile[cudaTileDim+1][cudaTileDim];

    int x = blockIdx.x * cudaTileDim + threadIdx.x;
    int y = blockIdx.y * cudaTileDim + threadIdx.y;

    __syncthreads();

    for (int j = 0; j < cudaTileDim; j += blockRows) {
        tile[threadIdx.x][threadIdx.y+j] = idata[x][y+j];
    }

    __syncthreads();

    x = blockIdx.y * cudaTileDim + threadIdx.x;
    y = blockIdx.x * cudaTileDim + threadIdx.y;

    for (int j = 0; j < cudaTileDim, j += blockRows) {
        odata[x][y+j] = tile[threadIdx.y+j][threadIdx.x];
    }

}

int main(int argc, char *argv[]) {

    // global array size
    const int nx = 2048, ny = 2048, n = nx*ny;
    const size_t nbytes = n*sizeof(float);

    // host arrays (global)
    float idata_h[n], tdata_h[n], gold[n];

    // MPI initialization
    int procid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // check parameters and calculate execution configuration
    const int mpiTileDimX = nx/numprocs;
    const int mpiTileDimY = ny/numprocs;
    if (nx % numprocs != 0 || ny % numprocs != 0) {
        printf("ny must be an integral multiple of numprocs\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (mpiTileDimX % cudaTileDim != 0 || mpiTileDimY % cudaTileDim != 0) {
        printf("mpiTileDimX and mpiTileDimY must be an integral multiple of cudaTileDim\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    if (numprocs > nDevices) {
        printf("numprocs must be <= number of GPUs on the system\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // assign MPI processes to GPUs
    const int deviceID = assignDevice(procid, numprocs);

    const dim3 dimGrid = dim3(mpiTileDimX/cudaTileDim, mpiTileDimY/cudaTileDim, 1);
    const dim3 dimBlock = dim3(cudaTileDim, blockRows, 1);

    // write parameters to terminal
    if (procid == 0) {
        printf("\nArray size: %d x %d\n\n", nx, ny);
        printf("CUDA block size: %d x %d CUDA tile size: %d x %d\n\n", 
            cudaTileDim, blockRows, cudaTileDim, cudaTileDim);
        
        printf("dimGrid: %d x %d x %d dimBlock: %d x %d x %d\n\n", 
            dimGrid.x, dimGrid.y, dimGrid.x, dimBlock.x, dimBlock.y);
        
        printf("numprocs: %d, Local input array size: %d x %d\n\n", 
            numprocs, nx, mpiTileDimY);

        printf("mpiTileDim: %d x %d\n\n", mpiTileDimX, mpiTileDimY);
    }

    // initialize data
    // host - each process has entire array on host (for now)
    int jg;
    for (int p = 0; p < numprocs; ++p) {
        for (int jl = 0; jl < mpiTileDimY; ++jl) {
            jg = p*mpiTileDimY + jl;
            for (int i = 0; i < nx; ++i) {
                idata_h[jg*nx + i] = i + jg*nx;
            }
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            gold[j*nx + i] = idata_h[i*nx + j];
        }
    }

    // CUDA vars and device arrays
    // each process has nx*mpiTileDimY = ny*mpiTileDimX elements
    float *idata_d, *tdata_d, *sTile_d, *rTile_d;

    cudaMalloc(&idata_d, nx*mpiTileDimY*sizeof(float));
    cudaMalloc(&tdata_d, ny*mpiTileDimX*sizeof(float));
    cudaMalloc(&sTile_d, mpiTileDimX*mpiTileDimY*sizeof(float));
    cudaMalloc(&rTile_d, mpiTileDimX*mpiTileDimY*sizeof(float));

    const int yOffset = procid*mpiTileDimY;
    cudaMemcpy2D(idata_d, )
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < mpiTileDimY; ++j) {
            idata_d[j*nx + i] = idata_h[i*ny + yOffset + j];
        }
    }

    //
    // transpose
    //

    MPI_Barrier(MPI_COMM_WORLD);
    double timeStart = MPI_WTIME();

    // 0th stage - local transpose
    // unfinished
    
}