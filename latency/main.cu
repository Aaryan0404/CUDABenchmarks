#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <sys/time.h>

using namespace std;

__device__ unsigned int smid() {
  unsigned int r;

  // ptx instruction that returns the SM id
  // (SM id is the index of the SM in the GPU)
  asm("mov.u32 %0, %%smid;" : "=r"(r));

  return r;
}

// Here we do pointer chasing = 
// sequence of instrs that involves 
// repeated series of irregular memory
// access patterns (accessed data 
// (determines next ptr addr to be accessed) 
// "serially-dependent chain of loads"

// hard for prefetchers to handle

template <typename T>
__global__ void pchase(T *buf, T *__restrict__ dummy_buf, int64_t N) {

    // Calculate the unique thread index within the entire grid
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize a pointer to navigate through the buffer
    int64_t *currentPointer = buf;

    // Define the number of times the inner loop will unroll
    const int UNROLL_COUNT = 32;

    // Loop over N in steps of UNROLL_COUNT, ensuring sequential memory access
    #pragma unroll 1
    for (int64_t step = 0; step < N; step += UNROLL_COUNT) {

        // Perform pointer chasing UNROLL_COUNT times
        #pragma unroll
        for (int i = 0; i < UNROLL_COUNT; i++) {
            // Update the currentPointer by dereferencing it
            currentPointer = (int64_t *)*currentPointer;
        }
    }

    // Write to dummy_buf if threadIndex is above a certain threshold
    // This step prevents the compiler from optimizing out the loop
    if (threadIndex > 12313) {
        dummy_buf[0] = (int64_t)currentPointer;
    }
}

int main(int argc, char **argv) {

    // initialize clock
    unsigned int clock = getGPUClock();
    
    typedef int64_t dtype;
    
    const int cl_size = 1;
    const int skip_factor = 8;
    
    for (int64_t LEN = 16; LEN < (1 << 24); LEN = LEN * 1.04 + 32) {
        if (LEN * skip_factor * cl_size * sizeof(dtype) > 120 * 1024 * 1024)
        
        LEN *= 1.5;
        const int64_t iters = max(LEN, (int64_t)1000000);

        vector<int64_t> order(LEN);
        int64_t *buf = NULL;
        int64_t *dbuf = NULL;
        dtype *dummy_buf = NULL;
        
        GPU_ERROR(cudaMallocManaged(&buf, skip_factor * cl_size * LEN * sizeof(dtype)));
        GPU_ERROR(cudaMalloc(&dbuf, skip_factor * cl_size * LEN * sizeof(dtype)));
        GPU_ERROR(cudaMallocManaged(&dummy_buf, sizeof(dtype)));
    
        for (int64_t i = 0; i < LEN; i++) {
            order[i] = i + 1;
        }
        order[LEN - 1] = 0;

        // randomize order
        std::random_device rd;
        std::mt19937 g(rd());
        shuffle(begin(order), end(order) - 1, g);

        // actual buff intiailization
        for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
      
            dtype idx = 0;
            for (int64_t i = 0; i < LEN; i++) {
                
                buf[(idx * cl_size + cl_lane) * skip_factor] = skip_factor * (order[i] * cl_size + cl_lane + (order[i] == 0 ? 1 : 0));
                idx = order[i];
            }
        }
    
        buf[skip_factor * (order[LEN - 2] * cl_size + cl_size - 1)] = 0;
        
        for (int64_t n = 0; n < LEN * cl_size * skip_factor; n++) {
            buf[n] = (int64_t)dbuf + buf[n] * sizeof(int64_t *);
        }

        // copy to device
        cudaMemcpy(dbuf, buf, skip_factor * cl_size * LEN * sizeof(dtype),
               cudaMemcpyHostToDevice);

        // run kernel
        pchase<dtype><<<1, 64>>>(buf, dummy_buf, iters);
        
        MeasurementSeries times;
        
        for (int i = 0; i < 7; i++) {
            GPU_ERROR(cudaDeviceSynchronize());
            
            double start = dtime();
            pchase<dtype><<<1, 64>>>(buf, dummy_buf, iters);
            
            GPU_ERROR(cudaDeviceSynchronize());
            
            double end = dtime();
            times.add(end - start);
        }
        
        GPU_ERROR(cudaGetLastError());
        
        double dt = times.minValue();
        
        cout << setw(9) << iters << " " << setw(5) << clock << " " //
         << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024
         << " "                                            //
         << fixed                                          //
         << setprecision(1) << setw(8) << dt / iters * 1000000000 << " " //
         << setw(7) << setprecision(1)
         << (double)dt / iters * clock * 1000 * 1000 << "\n"
         << flush;
         
         GPU_ERROR(cudaFree(buf));
         GPU_ERROR(cudaFree(dbuf));
         GPU_ERROR(cudaFree(dummy_buf));
    }
    
    cout << "\n";
}

//  Irregular mem accesses to expose latency
//  partitioning scheme is used to ensures
//  conformitity with cache lines (hit exactly once 
//  before subseq. accesses)
//  Latency in clock cycles (calculated based on current clock rate)

// first bump in graph is l1 cache trans
// second bump in graph is l2 cache trans
