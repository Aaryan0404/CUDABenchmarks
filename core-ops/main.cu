#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>
#include <map>

using namespace std;

// init kernel
template <typename T> __global__ void initKernel(T *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

  // initialize array with 1.1
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }

}

// fma kernel that operates on M elements at a time
// M = number of elements processed
// N = number of fma operations on each element
template <typename T, int N, int M>
__global__ void FMA_mixed(T p, T *A, int iters) {
  #pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    T t[M];

    // operate on M elements at a time ***
    #pragma unroll
    for (int m = 0; m < M; m++) {
      t[m] = p + threadIdx.x + iter + m;
    }

    #pragma unroll
    for (int n = 0; n < N / M; n++) {
      
      #pragma unroll
      for (int m = 0; m < M; m++) {
        t[m] = t[m] * (T)0.9 + (T)0.5;
      }
    }
    
    #pragma unroll
    for (int m = 0; m < M; m++) {
      if (t[m] > (T)22313.0) {
        A[0] = t[m];
      }
    }
  }
}

// fma kernel that operates on one element at a time
// M = number of elements processed
// N = number of fma operations on each element
template <typename T, int N, int M>
__global__ void FMA_separated(T p, T *A, int iters) {

  for (int iter = 0; iter < iters; iter++) {

    // operate on one element at a time
    #pragma unroll
    for (int m = 0; m < M; m++) {

      // initialize t
      T t = p + threadIdx.x + iter + m;

      // N
      for (int n = 0; n < N; n++) {
        // actual fma operation
        t = t * (T)0.9 + (T)0.5;
      }

      // check and store
      if (t > (T)22313.0) {
        A[0] = t;
      }

    }
  }
}

unsigned int gpu_clock = 0;

template <typename T, int N, int M>
double measure(int warpCount, void (*kernel)(T, T *, int)) {
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(0, &device);

  // calculations for blocksize
  const int iters = 10000;
  const int blockSize = 32 * warpCount;
  const int blockCount = 1024 * 16;  /// constant

  MeasurementSeries time;

  T *dA;
  GPU_ERROR(cudaMalloc(&dA, iters * 2 * sizeof(T)));
  
  // initialize array of iters * 2 elements with 1.1
  initKernel<<<52, 256>>>(dA, iters * 2);
  
  GPU_ERROR(cudaDeviceSynchronize());

  // warmup
  kernel<<<blockCount, blockSize>>>((T)0.32, dA, iters);

  // actual measurement
  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 1; i++) {
    double t1 = dtime();
    kernel<<<blockCount, blockSize>>>((T)0.32, dA, iters);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }

  cudaFree(dA);

  // cycles per op calculation
  double rcpThru = time.value() * gpu_clock * 1.0e6 / N;
  rcpThru = rcpThru / iters;
  rcpThru = rcpThru / warpCount;

  return rcpThru;
}

template <typename T> void measureTabular(int maxWarpCount) {

  vector<map<pair<int, int>, double>> r(3);
  const int N = 1024;

  // vary warp count (32 threads) from 1 to 32
  for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {

    // measurements for 1 stream
    r[0][{warpCount, 1}] = measure<T, N, 1>(warpCount, FMA_mixed<T, N, 1>);
    r[1][{warpCount, 1}] = measure<T, N, 1>(warpCount, FMA_mixed<T, N, 1>);
    r[2][{warpCount, 1}] = measure<T, N, 1>(warpCount, FMA_mixed<T, N, 1>);

    // measurements for 2 streams
    r[0][{warpCount, 2}] = measure<T, N, 2>(warpCount, FMA_mixed<T, N, 2>);
    r[1][{warpCount, 2}] = measure<T, N, 2>(warpCount, FMA_mixed<T, N, 2>);
    r[2][{warpCount, 2}] = measure<T, N, 2>(warpCount, FMA_mixed<T, N, 2>);

    // measurements for 4 streams
    r[0][{warpCount, 4}] = measure<T, N, 4>(warpCount, FMA_mixed<T, N, 4>);
    r[1][{warpCount, 4}] = measure<T, N, 4>(warpCount, FMA_mixed<T, N, 4>);
    r[2][{warpCount, 4}] = measure<T, N, 4>(warpCount, FMA_mixed<T, N, 4>);

    // measurements for 8 streams
    r[0][{warpCount, 8}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 8>);
    r[1][{warpCount, 8}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 8>);
    r[2][{warpCount, 8}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 8>);

    // measurements for 16 streams
    r[0][{warpCount, 16}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 16>);
    r[1][{warpCount, 16}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 16>);
    r[2][{warpCount, 16}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 16>);
  
  }

  // warpcount varies number of threads             (TLP)
  // stream    varies number of instruction streams (ILP)
  
  for (int i = 0; i < 3; i++) {
    for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
      for (int streams = 1; streams <= 16; streams *= 2) {
        cout << setw(7) << setprecision(3) << r[i][{warpCount, streams}] << " ";
      }
      cout << "\n";
    }
    cout << "\n";
  }
}

int main(int argc, char **argv) {
  gpu_clock = getGPUClock();
  measureTabular<float>(32);
  // measureTabular<double>(32);
}

// TABLE cols = num streams
// TABLE rows = warp count

// FMA latency = 4 cycles

// Steady State Throughput of 1 warp   = 1/1    = 1 per cycle
// (runs on one SM quadrant), no dependencies

// Steady State Throughput of 32 warps = 1/0.25 = 4 per cycle
// (all SM quadrants), dependencies irrelevant


// From README: 
// Scans combinations of ILP = 1, 2, 4, 8, 16, by generating 1, 2, 4, 8, 16 independent dependency chains
// TLP by varying the warp count on a SM from 1 to 32

// The final output is a ILP/TLP table, with (cycles per op):

// 4090
// Num Blocks = 1
// 4.06    2.06    1.07    1.09    1.11 
// 2.03    1.03   0.535   0.543   0.557 
// 1.01   0.516   0.268   0.272   0.279 
// 0.508   0.258    0.26   0.266   0.273 
// 0.255   0.258   0.258   0.263   0.271 
// 0.257   0.254   0.258   0.263    0.27

// H100
// Num Blocks = 1