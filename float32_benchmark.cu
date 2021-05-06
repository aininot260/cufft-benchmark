#include <cufft.h>       
#include <stdio.h>

#include "cufft_benchmark.h"

#define NX (1LL<<23)
#define BATCH 10

cufftComplex *data;

int main()
{
    CHECK(cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH));
    printf("size: %d\n", sizeof(cufftComplex)*NX*BATCH);

    cufftHandle plan;
    printf("cufft plan result: %d\n",cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH));

    start_time();

    for(int i=0;i<100;i++)
        printf("cufft execute result: %d\n",cufftExecC2C(plan, data, data, CUFFT_FORWARD));
    cudaDeviceSynchronize();
    end_time("float32 cufft time cost: ");

    cufftDestroy(plan);
    cudaFree(data);

    return 0;
}

