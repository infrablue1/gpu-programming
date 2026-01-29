#include "argparse/argparse.hpp"
#include "naive_gemm.cuh"
#include "utils.h"

enum GemmPass {
    NAIVE = 0,
    COALESCED,
    SHARED_MEMORY,
    BLOCKTILING_1D,
};

void callGemm(int blockSize, int M, int N, int K, GemmPass pass,
              bool checkResult);

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
    program.add_argument("--check-result")
        .help("check result with cpu reference")
        .default_value(false)
        .implicit_value(true);

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
    auto checkResult = program.get<bool>("--check-result");
    printf("MatrixA(%d,%d), MatrixB(%d,%d), block_size: %d, pass: %s\n", M, K,
           K, N, block_size, pass.c_str());

    static std::unordered_map<std::string, GemmPass> name2value = {
        {"naive", GemmPass::NAIVE},
        {"coalesced", GemmPass::COALESCED},
        {"shared_memory", GemmPass::SHARED_MEMORY},
        {"blocktiling-1d", GemmPass::BLOCKTILING_1D},
    };

    auto it = name2value.find(pass);
    if (it != name2value.end()) {
        callGemm(block_size, M, N, K, it->second, checkResult);
    } else {
        fprintf(stderr, "Unkown gemm pass %s\n", pass.c_str());
    }

    return 0;
}

void initMatrix(int totalSize, float *matrix) {
    // Use random float number to initialze matrix.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0, 100.0);
    for (int i = 0; i < totalSize; i++) {
        matrix[i] = dis(gen);
        //matrix[i] = 1.0;
    }
}

void launchNaiveGemm(int M, int N, int K, int blockSize, float *A, float *B,
                     float *C, cudaStream_t stream) {
    dim3 threads(blockSize * blockSize);
    dim3 grid(ceilDiv(M, blockSize), ceilDiv(N, blockSize));
    if (blockSize == 16) {
        naiveGemmKernel<16><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    } else {
        naiveGemmKernel<32><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    }
}

void launchCoalescedGemm(int M, int N, int K, int blockSize, float *A, float *B,
                         float *C, cudaStream_t stream) {
    dim3 threads(blockSize * blockSize);
    dim3 grid(ceilDiv(M, blockSize), ceilDiv(N, blockSize));
    if (blockSize == 16) {
        coalescedGemmKernel<16><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    } else {
        coalescedGemmKernel<32><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    }
}

void launchSharedMemoryGemm(int M, int N, int K, int blockSize, float *A,
                            float *B, float *C, cudaStream_t stream) {
    dim3 threads(blockSize * blockSize);
    dim3 grid(ceilDiv(M, blockSize), ceilDiv(N, blockSize));
    if (blockSize == 16) {
        sharedMemoryGemmKernel<16><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    } else {
        sharedMemoryGemmKernel<32><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    }
}

void launchBlockTiling1DGemm(int M, int N, int K, int blockSize, float *A,
                            float *B, float *C, cudaStream_t stream) {
   constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    dim3 threads((BM * BN) / TM);
    dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));
    blockTiling1DGemmKernel<BM, BN, BK, TM><<<grid, threads, 0, stream>>>(
        M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchGemmKernel(int M, int N, int K, int blockSize, float *A, float *B,
                      float *C, cudaStream_t stream, GemmPass pass) {
    switch (pass) {
    case GemmPass::NAIVE:
        launchNaiveGemm(M, N, K, blockSize, A, B, C, stream);
        break;
    case GemmPass::COALESCED:
        launchCoalescedGemm(M, N, K, blockSize, A, B, C, stream);
        break;
    case GemmPass::SHARED_MEMORY:
        launchSharedMemoryGemm(M, N, K, blockSize, A, B, C, stream);
        break;
    case GemmPass::BLOCKTILING_1D:
        launchBlockTiling1DGemm(M, N, K, blockSize, A, B, C, stream);
        break;
    default:
        fprintf(stderr, "Unkown gemm pass %d\n", pass);
        exit(EXIT_FAILURE);
        break;
    }
}

void callGemm(int blockSize, int M, int N, int K, GemmPass pass,
              bool checkResult) {
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

    // Warm up
    for (int i = 0; i < kWarupIters; i++) {
        launchGemmKernel(M, N, K, blockSize, dA, dB, dC, stream, pass);
    }

    printf("Warmup with %d iterations done!\n", kWarupIters);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Launch kernel
    for (int i = 0; i < kBenchIters; i++) {
        launchGemmKernel(M, N, K, blockSize, dA, dB, dC, stream, pass);
    }

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
           blockSize * blockSize);

    if (checkResult) {
        checkCudaErrors(cudaMemcpy(hC, dC, memSizeC, cudaMemcpyDeviceToHost));
        printf("Calculate CPU reference result.\n");
        // std::vector<std::vector<float>> refC(M, std::vector<float>(N));
        auto refC = std::make_unique<float[]>(sizeC);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float tmp = 0.0;
                for (int k = 0; k < K; k++) {
                    tmp += hA[i * K + k] * hB[k * N + j];
                }
                refC[i * M + j] = tmp;
            }
        }

        printf("Compare cpu result and cuda result.\n");
        // test relative error by the formula
        //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

        double eps = 1.0e-2;
        bool correct = true;

        for (int i = 0; i < M * N; i++) {
            double absErr = fabs(hC[i] - refC[i]);
            double absVal = fabs(hC[i]);
            double relErr = absErr / absVal;

            if (relErr > eps) {
                printf(
                    "Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, hC[i], refC[i], eps);
                correct = false;
                break;
            }
        }

        printf("Result check %s\n",
               correct ? "result = PASS" : "result = FAIL");
    }

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