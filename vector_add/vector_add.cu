
#include "utils.h"

using namespace std;

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

void doVectorAdd(int numElements);

int main(void) {
    // 512 Bytes, 1 KiB, 4KiB, 64KiB, 1 MiB, 4 MiB
    std::vector<int> numElementsList{512,       1024,        4096,
                                     64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    for (auto numElements : numElementsList) {
        doVectorAdd(numElements);
    }
    return 0;
}

void doVectorAdd(int numElements) {

    // Print the vector length to be used, and compute its size
    size_t size = numElements * sizeof(float);
    constexpr int kWarmup = 10;
    constexpr int kCount = 100;

    // Allocate the host input vector A
    auto h_A = std::make_unique<float[]>(size);
    auto h_B = std::make_unique<float[]>(size);
    auto h_C = std::make_unique<float[]>(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in
    // device memory
    CUDA_CHECK(cudaMemcpy(d_A, h_A.get(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.get(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.get(), size, cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Warmup
    for (int i = 0; i < kWarmup; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                      numElements);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kCount; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                      numElements);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t2 = std::chrono::high_resolution_clock::now();
    auto time_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    CUDA_CHECK(cudaMemcpy(h_C.get(), d_C, size, cudaMemcpyDeviceToHost));

    printf(
        "VectorAdd %d elements Kernel average execution time is about %ld us\n",
        numElements, time_elapsed.count() / kCount);

    // Free device global memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}
