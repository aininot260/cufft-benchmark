#include <stdio.h>

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

#include "cufft_benchmark.h"

#define NX (1LL<<23)
#define BATCH 10

half2 *data; 

int main()
{
    CHECK(cudaMalloc((void**)&data, sizeof(half2)*NX*BATCH));
    printf("size: %d\n", sizeof(half2)*NX*BATCH);

    long long sample_size = NX;
    int block_size = BATCH;
    size_t workSize = 0;

    cufftHandle plan_inverse;
    cufftCreate(&plan_inverse);
    printf("cufftXt plan result: %d\n", cufftXtMakePlanMany(plan_inverse,
        1, &sample_size,
        NULL, 1, 1, CUDA_C_16F,
        NULL, 1, 1, CUDA_C_16F,
        block_size, &workSize, CUDA_C_16F));

    start_time();

    for(int i=0;i<100;i++)
        printf("cufftXt execute result: %d\n", cufftXtExec(plan_inverse, data, data, CUFFT_FORWARD));
    cudaDeviceSynchronize();
    end_time("half16 cufft time cost: ");

    cufftDestroy(plan_inverse);
    cudaFree(data);

    return 0;
}

