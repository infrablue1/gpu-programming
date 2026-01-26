#include "argparse/argparse.hpp"
#include "naive_gemm.cuh"
#include "utils.h"

void initMatrix(int totalSize, float *matrix);
void callGemm(int blockSize, int M, int N, int K, const std::string pass);

int main(int argc, char const *argv[]) {

    argparse::ArgumentParser program("bench_gemm");
    program.add_argument("-M")
        .help("Matrix A's rows")
        .scan<'i', int>()
        .default_value(kM);
    program.add_argument("-N")
        .help("Matrix B's columns")
        .scan<'i', int>()
        .default_value(kN);
    program.add_argument("-K")
        .help("Matrix A's columns")
        .scan<'i', int>()
        .default_value(kK);
    program.add_argument("--block-size")
        .help("Block size")
        .scan<'i', int>()
        .default_value(kBlockSize);
    program.add_argument("--pass").help("Gemm pass").default_value("naive");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto M = program.get<int>("-M");
    auto N = program.get<int>("-N");
    auto K = program.get<int>("-K");
    int block_size = program.get<int>("--block-size");
    auto pass = program.get<std::string>("--pass");
    printf("MatrixA(%d,%d), MatrixB(%d,%d), block_size: %d, pass: %s\n", M, K,
           K, N, block_size, pass.c_str());

    callGemm(block_size, M, N, K, pass);

    return 0;
}

void initMatrix(int totalSize, float *matrix) {
    // Use random float number to initialze matrix.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0, 100.0);
    for (int i = 0; i < totalSize; i++) {
        matrix[i] = dis(gen);
    }
}

void callGemm(int blockSize, int M, int N, int K, const std::string pass) {
    // Allocate host memory for matrix A, B and C
    // A: M x K
    // B: K x N
    // C: M x N
    float *hA;
    unsigned int sizeA = M * K;
    unsigned int memSizeA = sizeof(float) * sizeA;
    checkCudaErrors(cudaMallocHost(&hA, memSizeA));

    float *hB;
    unsigned int sizeB = K * N;
    unsigned int memSizeB = sizeof(float) * sizeB;
    checkCudaErrors(cudaMallocHost(&hB, memSizeB));

    float *hC;
    unsigned int sizeC = M * N;
    unsigned int memSizeC = sizeof(float) * sizeC;
    checkCudaErrors(cudaMallocHost(&hC, memSizeC));

    // Initialize matrix A and B
    initMatrix(sizeA, hA);
    initMatrix(sizeB, hB);

    // Allocate device memory
    float *dA, *dB, *dC;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dA), memSizeA));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dB), memSizeB));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dC), memSizeC));

    // Allocate CUDA events for timing
    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Copy host memory to device
    checkCudaErrors(
        cudaMemcpyAsync(dA, hA, memSizeA, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
        cudaMemcpyAsync(dB, hB, memSizeB, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(blockSize * blockSize);
    dim3 grid(ceilDiv(M, blockSize), ceilDiv(N, blockSize));

    // Method to launch kernel multiple times
    auto lauchKernelMultipleIters = [&](int nIters) {
        for (int i = 0; i < nIters; i++) {
            if (pass == "naive") {
                if (blockSize == 16) {
                    naiveGemmKernel<16><<<grid, threads, 0, stream>>>(
                        M, N, K, kDefaultAlpha, kDefaultBeta, dA, dB, dC);
                } else {
                    naiveGemmKernel<32><<<grid, threads, 0, stream>>>(
                        M, N, K, kDefaultAlpha, kDefaultBeta, dA, dB, dC);
                }
            } else if (pass == "coalesce") {
                if (blockSize == 16) {
                    coalescedGemmKernel<16><<<grid, threads, 0, stream>>>(
                        M, N, K, kDefaultAlpha, kDefaultBeta, dA, dB, dC);
                } else {
                    coalescedGemmKernel<32><<<grid, threads, 0, stream>>>(
                        M, N, K, kDefaultAlpha, kDefaultBeta, dA, dB, dC);
                }
            } else {
                fprintf(stderr, "Unknown gemm pass %s\n", pass.c_str());
                exit(EXIT_FAILURE);
            }
        }
    };

    // Warm up
    lauchKernelMultipleIters(kWarupIters);

    printf("Warmup with %d iterations done!\n", kWarupIters);
    checkCudaErrors(cudaEventSynchronize(stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Launch kernel
    lauchKernelMultipleIters(kBenchIters);

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    // Compute and print the performance
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float msecPerMatrixMul = msecTotal / kBenchIters;

    double flopsPerMatrixMul = 2.0 * static_cast<double>(M) *
                               static_cast<double>(N) * static_cast<double>(K);
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul,
           threads.x * threads.y);

    // Do clean up work
    checkCudaErrors(cudaFreeHost(hA));
    checkCudaErrors(cudaFreeHost(hB));
    checkCudaErrors(cudaFreeHost(hC));
    checkCudaErrors(cudaFree(dA));
    checkCudaErrors(cudaFree(dB));
    checkCudaErrors(cudaFree(dC));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}