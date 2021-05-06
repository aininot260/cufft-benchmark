# cufft-benchmark
 This is a cufft benchmark comparing with half16 and float32.

## Usage
- For windows with Visual Studio

Initialize a Visual Studio developer command prompt with x64 support
```
nvcc float32_benchmark.cu utils.cu -o float32_benchmark -arch=sm_61 -lcufft
nvcc half16_benchmark.cu utils.cu -o half16_benchmark -arch=sm_61 -lcufft
```

- For linux
```
nvcc float32_benchmark.cu utils.cu -o float32_benchmark -arch=sm_70 -lcufft
nvcc half16_benchmark.cu utils.cu -o half16_benchmark -arch=sm_70 -lcufft
```

## Result
The test result on NVIDIA Geforce MX350, Pascal 6.1
```
float32 cufft time cost:  TIME COST: 8.342000s
half16 cufft time cost:  TIME COST: 56.931000s
```

The test result on NVIDIA Tesla V100, Volta 7.0
```
float32 cufft time cost:  TIME COST: 0.521416s
half16 cufft time cost:  TIME COST: 0.619594s
```