#include "argparse/argparse.hpp"
#include "utils.h"

#include "blocktiling_1d.cuh"
#include "blocktiling_2d.cuh"
#include "global_coalesced.cuh"
#include "naive.cuh"
#include "naive_tensorcore.cuh"
#include "shared_memory.cuh"
#include "vectorize.cuh"
#include "warptiling.cuh"

enum GemmPass {
    NAIVE = 0,
    COALESCED,
    SHARED_MEMORY,
    BLOCKTILING_1D,
    BLOCKTILING_2D,
    VECTORIZE,
    WARPTILING,
    NAIVE_TENSORCORE,
};

void callGemm(int M, int N, int K, GemmPass pass, bool checkResult);

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
    auto pass = program.get<std::string>("--pass");
    auto checkResult = program.get<bool>("--check-result");
    printf("MatrixA(%d,%d), MatrixB(%d,%d), pass: %s\n", M, K, K, N,
           pass.c_str());

    static std::unordered_map<std::string, GemmPass> name2value = {
        {"naive", GemmPass::NAIVE},
        {"coalesced", GemmPass::COALESCED},
        {"shared-memory", GemmPass::SHARED_MEMORY},
        {"blocktiling-1d", GemmPass::BLOCKTILING_1D},
        {"blocktiling-2d", GemmPass::BLOCKTILING_2D},
        {"vectorize", GemmPass::VECTORIZE},
        {"warptiling", GemmPass::WARPTILING},
        {"naive-tensorcore", GemmPass::NAIVE_TENSORCORE},
    };

    auto it = name2value.find(pass);
    if (it != name2value.end()) {
        callGemm(M, N, K, it->second, checkResult);
    } else {
        fprintf(stderr, "Unkown gemm pass %s\n", pass.c_str());
    }

    return 0;
}

void initMatrix(int totalSize, float *matrix) {
    // Use random float number to initialze matrix.
    unsigned int fixedSeed = 12345;
    std::mt19937 gen(fixedSeed);
    std::uniform_real_distribution<float> dis(-100.0, 100.0);
    for (int i = 0; i < totalSize; i++) {
        matrix[i] = dis(gen);
        // matrix[i] = 1.0;
    }
}

void initMatrixConst(int totalSize, float value, float *matrix) {
    for (int i = 0; i < totalSize; i++) {
        matrix[i] = value;
    }
}

void launchNaiveGemm(int M, int N, int K, float *A, float *B, float *C,
                     cudaStream_t stream) {
    dim3 threads(kBlockSize * kBlockSize);
    dim3 grid(ceilDiv(M, kBlockSize), ceilDiv(N, kBlockSize));

    naiveGemmKernel<kBlockSize><<<grid, threads, 0, stream>>>(
        M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchCoalescedGemm(int M, int N, int K, float *A, float *B, float *C,
                         cudaStream_t stream) {
    dim3 threads(kBlockSize * kBlockSize);
    dim3 grid(ceilDiv(M, kBlockSize), ceilDiv(N, kBlockSize));

    coalescedGemmKernel<kBlockSize><<<grid, threads, 0, stream>>>(
        M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchSharedMemoryGemm(int M, int N, int K, float *A, float *B, float *C,
                            cudaStream_t stream) {
    dim3 threads(kBlockSize * kBlockSize);
    dim3 grid(ceilDiv(M, kBlockSize), ceilDiv(N, kBlockSize));

    sharedMemoryGemmKernel<kBlockSize><<<grid, threads, 0, stream>>>(
        M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchBlockTiling1DGemm(int M, int N, int K, float *A, float *B, float *C,
                             cudaStream_t stream) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    dim3 threads((BM * BN) / TM);
    dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));
    blockTiling1DGemmKernel<BM, BN, BK, TM><<<grid, threads, 0, stream>>>(
        M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchBlockTiling2DGemm(int M, int N, int K, float *A, float *B, float *C,
                             cudaStream_t stream) {
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    if (M >= 128 && N >= 128) {
        constexpr int BM = 128;
        constexpr int BN = 128;

        dim3 threads((BM / TM) * (BN / TN));
        dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));

        blockTiling2DGemmKernel<BM, BN, BK, TM, TN>
            <<<grid, threads, 0, stream>>>(M, N, K, kDefaultAlpha, kDefaultBeta,
                                           A, B, C);
    } else {
        constexpr int BM = 64;
        constexpr int BN = 64;

        dim3 threads((BM / TM) * (BN / TN));
        dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));

        blockTiling2DGemmKernel<BM, BN, BK, TM, TN>
            <<<grid, threads, 0, stream>>>(M, N, K, kDefaultAlpha, kDefaultBeta,
                                           A, B, C);
    }
}

void launchVectorizeGemm(int M, int N, int K, float *A, float *B, float *C,
                         cudaStream_t stream) {
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    if (M >= 128 && N >= 128) {
        constexpr int BM = 128;
        constexpr int BN = 128;

        dim3 threads((BM / TM) * (BN / TN));
        dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));

        vectorizeGemmKernel<BM, BN, BK, TM, TN><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    } else {
        constexpr int BM = 64;
        constexpr int BN = 64;

        dim3 threads((BM / TM) * (BN / TN));
        dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));

        vectorizeGemmKernel<BM, BN, BK, TM, TN><<<grid, threads, 0, stream>>>(
            M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
    }
}

void launchWarpTilingGemm(int M, int N, int K, float *A, float *B, float *C,
                          cudaStream_t stream) {

    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WN = 64;
    constexpr int WM = 64;
    constexpr int WNITER = 4;
    constexpr int TM = 8;
    constexpr int TN = 4;

    dim3 threads(NUM_THREADS);
    dim3 grid(ceilDiv(M, BM), ceilDiv(N, BN));

    warpTilingGemmKernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, threads, 0, stream>>>(M, N, K, kDefaultAlpha, kDefaultBeta, A,
                                       B, C);
}

void launchNaiveTensorcoreGemm(int M, int N, int K, float *A, float *B,
                               float *C, cudaStream_t stream) {
    dim3 threads(kWarpSize);
    dim3 grid(ceilDiv(N, kWMMA_N), ceilDiv(M, kWMMA_M));

    naiveTensorCore<float>
        <<<grid, threads>>>(M, N, K, kDefaultAlpha, kDefaultBeta, A, B, C);
}

void launchGemmKernel(int M, int N, int K, float *A, float *B, float *C,
                      cudaStream_t stream, GemmPass pass) {
    switch (pass) {
    case GemmPass::NAIVE:
        launchNaiveGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::COALESCED:
        launchCoalescedGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::SHARED_MEMORY:
        launchSharedMemoryGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::BLOCKTILING_1D:
        launchBlockTiling1DGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::BLOCKTILING_2D:
        launchBlockTiling2DGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::VECTORIZE:
        launchVectorizeGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::WARPTILING:
        launchWarpTilingGemm(M, N, K, A, B, C, stream);
        break;
    case GemmPass::NAIVE_TENSORCORE:
        launchNaiveTensorcoreGemm(M, N, K, A, B, C, stream);
        break;
    default:
        fprintf(stderr, "Unkown gemm pass %d\n", pass);
        exit(EXIT_FAILURE);
        break;
    }
}

void callGemm(int M, int N, int K, GemmPass pass, bool checkResult) {
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
    if (pass == GemmPass::NAIVE_TENSORCORE) {
        // For float cuda only support tf32 MMA, which may cause lower accuracy.
        // So we use constant input value to avoid it.
        initMatrixConst(sizeA, 1.0f, hA);
        initMatrixConst(sizeA, 2.0f, hA);
    } else {
        initMatrix(sizeA, hA);
        initMatrix(sizeB, hB);
    }

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
    for (int i = 0; i < kWarmupIters; i++) {
        launchGemmKernel(M, N, K, dA, dB, dC, stream, pass);
    }

    printf("Warmup with %d iterations done!\n", kWarmupIters);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Launch kernel
    for (int i = 0; i < kBenchIters; i++) {
        launchGemmKernel(M, N, K, dA, dB, dC, stream, pass);
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
           kBlockSize * kBlockSize);

    if (checkResult) {
        checkCudaErrors(cudaMemcpy(hC, dC, memSizeC, cudaMemcpyDeviceToHost));

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

        double eps = 1.0e-4;

        bool correct = true;
        double dotLength = K;

        for (int i = 0; i < M * N; i++) {
            double absErr = fabs(hC[i] - refC[i]);
            double absVal = fabs(hC[i]);
            double relErr = absErr / absVal / dotLength;

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