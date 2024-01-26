#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"

#include <iomanip>
#include <iostream>

using namespace std;

#ifdef __NVCC__
using dtype = int4; 
#else
using dtype = float4;
#endif

dtype *dA, *dB;

// kernel to initiate an array with 1.1
// array: array to initiate
// totalElements: number of elements in the array
__global__ void initKernel(dtype *array, size_t totalElements) {
    size_t threadId = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t elementIndex = threadId; elementIndex < totalElements; elementIndex += stride) {
      array[elementIndex] = make_int4(1.1f, 1.1f, 1.1f, 1.1f);
    }
}


// Each thread block repeatedly reads the contents of the same buffer

// parallel reduction kernel
// N: number of elements to sum
// iters: number of iterations
// BLOCKSIZE: number of threads in a block
template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int zero) {
  dtype localSum = make_int4(0, 0, 0, 0);

  // each thread's B ptr calculated
  B += threadIdx.x;

  // number of times to perform the reduction
  #pragma unroll N / BLOCKSIZE > 32 ? 1 : 32 / (N / BLOCKSIZE)
  for (int iter = 0; iter < iters; iter++) {

    // each thread adds zero to its pointer
    B += zero;

    // done in parallel by all threads in the block
    #pragma unroll N / BLOCKSIZE >= 64 ? 32 : (N / BLOCKSIZE)
    for (int i = 0; i < N; i += BLOCKSIZE) {
      localSum.x += B[i].x;
      localSum.y += B[i].y;
      localSum.z += B[i].z;
      localSum.w += B[i].w;
    }

    // each thread multiplies its local sum by 1.3
    localSum.x *= 1.3f;
    localSum.y *= 1.3f;
    localSum.z *= 1.3f;
    localSum.w *= 1.3f;
  }

  // each thread adds its local sum to the global sum
  // in A idx
  if (localSum.x == 1233) {
    A[threadIdx.x] = localSum;
  }
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
  return 0.0;
}

template <int N> void measure() {
  const size_t iters = (size_t)1000000000 / N + 2;

  const int blockSize = 256;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, sumKernel<N, iters, blockSize>, blockSize, 0));

  // we make sure to launch one block per SM
  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());

  // actual loop of measurements
  for (int i = 0; i < 15; i++) {
    const size_t bufferCount = N; // + i * 1282;

    // both A and B are of size N = vaires in test below
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, blockSize>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, blockSize>>>(dB, bufferCount);
    
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    callKernel<N, iters, blockSize>(blockCount);
    GPU_ERROR(cudaDeviceSynchronize());
    
    double t2 = dtime();
    time.add((t2 - t1));

    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }
  double blockDV = N * sizeof(dtype);

  // prints the results
  double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s"                                       //
       << setprecision(0) << setw(10)
       << dram_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << dram_write.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl; //
}

// exponential series
// power to which 1.17 is raised
size_t constexpr expSeries(size_t N) {
    size_t val = 32 * 512;
    for (size_t i = 0; i < N; i++) {
      val *= 1.17;
    }
    return (val / 512) * 512;
  }

int main(int argc, char **argv) {
  initMeasureMetric();
  unsigned int clock = getGPUClock();
  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw"    //
       << setw(16) << "DRAM read"  //
       << setw(16) << "DRAM write" //
       << setw(16) << "L2 read"    //
       << setw(16) << "L2 store\n";

  initMeasureMetric();

  measure<256>();
  measure<512>();
  measure<3 * 256>();
  measure<2 * 512>();
  measure<3 * 512>();
  measure<4 * 512>();
  measure<5 * 512>();
  measure<6 * 512>();
  measure<7 * 512>();
  measure<8 * 512>();
  measure<9 * 512>();
  measure<10 * 512>();
  measure<11 * 512>();
  measure<12 * 512>();
  measure<13 * 512>();
  measure<14 * 512>();
  measure<15 * 512>();
  measure<16 * 512>();
  measure<17 * 512>();
  measure<18 * 512>();
  measure<19 * 512>();
  measure<20 * 512>();
  measure<21 * 512>();
  measure<22 * 512>();
  measure<23 * 512>();
  measure<24 * 512>();
  measure<25 * 512>();
  measure<26 * 512>();
  measure<27 * 512>();
  measure<28 * 512>();
  measure<29 * 512>();
  measure<30 * 512>();
  measure<31 * 512>();
  measure<32 * 512>();

  measure<expSeries(1)>();
  measure<expSeries(2)>();
  measure<expSeries(3)>();
  measure<expSeries(4)>();
  measure<expSeries(5)>();
  measure<expSeries(6)>();
  measure<expSeries(7)>();
  measure<expSeries(8)>();
  measure<expSeries(9)>();
  measure<expSeries(10)>();
  measure<expSeries(11)>();
  measure<expSeries(12)>();
  measure<expSeries(13)>();
  measure<expSeries(14)>();
  measure<expSeries(16)>();
  measure<expSeries(17)>();
  measure<expSeries(18)>();
  measure<expSeries(19)>();
  measure<expSeries(20)>();
  measure<expSeries(21)>();
  measure<expSeries(22)>();
  measure<expSeries(23)>();
  measure<expSeries(24)>();
  measure<expSeries(25)>();
  measure<expSeries(26)>();
  measure<expSeries(27)>();
  measure<expSeries(28)>();
  measure<expSeries(29)>();
  measure<expSeries(30)>();
  measure<expSeries(31)>();
  measure<expSeries(32)>();
  measure<expSeries(33)>();
  measure<expSeries(34)>();
  measure<expSeries(35)>();
  measure<expSeries(36)>();
  measure<expSeries(37)>();
  measure<expSeries(38)>();
  measure<expSeries(39)>();
  measure<expSeries(40)>();
  measure<expSeries(41)>();
  measure<expSeries(42)>();
  measure<expSeries(43)>();
  measure<expSeries(44)>();
  measure<expSeries(45)>();
  measure<expSeries(46)>();
  measure<expSeries(47)>();
  measure<expSeries(48)>();
  measure<expSeries(49)>();
}


// Measures bandwidths of the first and second cache level. 
// Launches one thread block per SM. Each thread block repeatedly 
// reads the contents of the same buffer. Varying buffer sizes 
// changes the targeted cache level.

// Each thread block repeatedly reads the contents of the same buffer. 
// Varying buffer sizes changes the targeted cache level.
