#include <mpiDeviceUtil.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <limits.h>

int main(int argc, char *argv[]) {

    const int n = 1024*1024;
    
    int procid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int deviceID = assignDevice(procid, numprocs);

    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    for (int i = 0; i < numprocs; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == procid) printf(" [%d] host: %s, device: %d\n", 
            procid, hostname, deviceID);
    }

    // get memory use bafore large allocations
    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);

    // allocate memory on each device
    float* d;
    cudaMalloc(&d, n*sizeof(float));

    // get free memory after allocation
    size_t freeA, totalA;
    cudaMemGetInfo(&freeA, &totalA);

    for (int i = 0; i < numprocs; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == procid) printf(" [%d] device arrays allocated: %d\n", 
            procid, (freeB-freeA)/n/sizeof(float));
    }

    cudaFree(d);

    MPI_Finalize();
    
}
