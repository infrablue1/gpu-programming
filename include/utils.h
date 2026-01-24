#include <cstdio>
#include <chrono>
#include <memory>
#include <vector>
#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include <cuda.h>

constexpr int kWarupIters = 1;
constexpr int kBenchIters = 10;
constexpr float kDefaultAlpha = 1.0;
constexpr float kDefaultBeta = 0.0;
constexpr int kM = 4096;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr int kBlockSize = 32;


// This function is called by the checkCudaErrors macro
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    exit(EXIT_FAILURE);
  }
}


// The main macro to use
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

constexpr inline int ceilDiv(int m, int n) {
  return (m + n - 1) / n;
}