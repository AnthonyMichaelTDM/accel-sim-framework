/*
Some of the code is adopted from device query benchmark
from CUDA SDK
*/

#include <cuda_runtime.h>
#include <stdio.h>
// #include <helper_cuda.h>
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xc0, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char **argv) {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    // device
    printf("  Device : \"%s\"\n\n", deviceProp.name);
    printf("  CUDA version number                         : %d.%d\n",
           deviceProp.major, deviceProp.minor);

    // core
    printf("  GPU Max Clock rate                             : %.0f MHz \n",
           deviceProp.clockRate * 1e-3f);
    printf("  Multiprocessors Count                       : %d\n",
           deviceProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor: %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  CUDA Cores per multiprocessor               : %d \n",
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
    printf("  Registers per multiprocessor                : %d\n",
           deviceProp.regsPerMultiprocessor);
    printf("  Shared memory per multiprocessor            : %lu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Warp size                                   : %d\n",
           deviceProp.warpSize);

    // threadblock config
    printf("  Maximum number of threads per block         : %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Shared memory per block                     : %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Registers per block                         : %d\n",
           deviceProp.regsPerBlock);

    // L1 cache
    printf("  globalL1CacheSupported                      : %d\n",
           deviceProp.globalL1CacheSupported);
    printf("  localL1CacheSupported                       : %d\n",
           deviceProp.localL1CacheSupported);

    // L2 cache
    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size                             : %.0f MB\n",
             static_cast<float>(deviceProp.l2CacheSize / 1048576.0f));
    }

    // memory
    char msg[256];
    snprintf(msg, sizeof(msg),
             "  Global memory size                        : %.0f GB\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f));
    printf("%s", msg);
    printf("  Memory Clock rate                           : %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width                            : %d bit\n",
           deviceProp.memoryBusWidth);

    printf(" ////////////////////////// \n");
  }
}
