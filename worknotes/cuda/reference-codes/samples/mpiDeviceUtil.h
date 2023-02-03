#ifndef MPIDEVICEUTIL
#define MPIDEVICEUTIL
#include <mpi.h>
#include <unistd.h>
#include <limits.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

int assignDevice(const int procid, const int numprocs) {

    // array for hostnames. C string arrays as that's what MPI accepts
    char hosts[numprocs][HOST_NAME_MAX];
    char hostname[HOST_NAME_MAX];

    // every process collects the hostname of all the nodes
    gethostname(hostname, HOST_NAME_MAX);

    MPI_Allgather(hostname, HOST_NAME_MAX, MPI_CHAR, hosts, HOST_NAME_MAX, MPI_CHAR, MPI_COMM_WORLD);

    std::vector<std::string> hosts_string(numprocs);
    std::string hostname_string = hostname;

    for (int i = 0; i < numprocs; ++i) {
        hosts_string[i] = hosts[i];
    }

    std::sort(std::begin(hosts_string), std::end(hosts_string));

    int colour = 0;
    for (int i = 0; i < numprocs; ++i) {
        if (i > 0) {
            if ( hosts_string[i-1] != hosts_string[i] ) colour += 1;
        }
        if ( hostname_string == hosts_string[i] ) break;
    }

    MPI_Comm newComm;
    int deviceID;
    MPI_Comm_split(MPI_COMM_WORLD, colour, 0, &newComm);
    MPI_Comm_rank(newComm, &deviceID);

    cudaSetDevice(deviceID);

    return deviceID;

}

#endif
