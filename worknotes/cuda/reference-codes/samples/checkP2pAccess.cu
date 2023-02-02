#include <stdio.h>

int main() {

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("Number of CUDA-capable devices: %d\n", nDevices);

    cudaDeviceProp prop;
    for (int i = 0; i < nDevices; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }

    int p2pOK[nDevices][nDevices];
    for (int i = 0; i < nDevices; ++i) {
        for (int j = i+1; j < nDevices; ++j) {
            cudaDeviceCanAccessPeer(&p2pOK[i][j], i, j);
            p2pOK[j][i] = p2pOK[i][j];
        }
    }
    printf("\n");

    for (int i = 0; i < nDevices; ++i) {
        printf("   %3d", i);
    }
    printf("\n");

    for (int j = 0; j < nDevices; ++j) {
        printf("%3d",j);
        for (int i = 0; i < nDevices; ++i) {
            if (i==j) {
                printf("  -   ");
            } else if (p2pOK[i][j] == 1) {
                printf("  Y   ");
            } else {
                printf("      ");
            }
        }
        printf("\n");
    }
}