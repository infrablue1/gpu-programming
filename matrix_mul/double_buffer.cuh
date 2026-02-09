#pragma once

#include "utils.h"
#include "warp_utils.cuh"

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ void doubleBufferingGemm(int M, int N, int K, float alpha,
                                    float beta, const float *A, const float *B,
                                    float *C) {
    // Block index
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    // Warp index
    const int warpIndex = threadIdx.x / kWarpSize;
    const int warpRow = warpIndex / (BN / WN);
    const int warpCol = warpIndex % (BN / WN);

    // Warp subtile size
    // WMITER * WNITER * TM * TN = (WM * WN) / kWarpSize
    // Each subWarp process WM * WN output elements and each warp have kWarpSize
    // threads. Thus each thread will process (WMITER * WNITER * TM * TN) output
    // elements.
    constexpr int WMITER = (WM * WN) / (kWarpSize * TM * TN * WNITER);

    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // ThreadIndex in warp
    const int threadIndexInWarp = threadIdx.x % kWarpSize; // [0, 31)
    const int threadRowInWarp = threadIndexInWarp / (WSUBN / TN);
    const int threadColInWarp = threadIndexInWarp % (WSUBN / TN);

    // Shared memory
    __shared__ float As[2 * BM * BK];
    __shared__ float Bs[2 * BK * BN];

    A += blockRow * BM * K; // row=blockRow, col=0
    B += blockCol * BN;     // row=0, col=blockCol
    C += (blockRow * BM + warpRow * WN) * N + blockCol * BN + warpCol * WN;

    // Pretending there's half as many threads as there actually are
    const int unit = NUM_THREADS / 2;
    const int innerRowA = (threadIdx.x % unit) / (BK / 4);
    const int innerColA = (threadIdx.x % unit) % (BK / 4);
    const int innerRowB = (threadIdx.x % unit) / (BN / 4);
    const int innerColB = (threadIdx.x % unit) % (BN / 4);

    const int strideA = (NUM_THREADS / 2) / (BK / 4);
    const int strideB = (NUM_THREADS / 2) / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    bool isSecondPart = (threadIdx.x >= unit);

    if (isSecondPart) {
        // load first (B0)
        loadFromGlobalMem<BM, BN, BK, strideA, strideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    }

    __syncthreads();

    for (int blockIndex = 0; blockIndex < K; blockIndex += 2 * BK) {
        if (isSecondPart == false) {
            // procss B0 as it's been loaded
            processFromSharedMem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM,
                                 WSUBN, TM, TN>(
                regM, regN, threadResults, As, Bs, warpRow, warpCol,
                threadRowInWarp, threadColInWarp);

            __syncthreads();

            if (blockIndex + BK < K) {
                // procss B1 as the load part is done by other thread part
                processFromSharedMem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM,
                                     WSUBN, TM, TN>(
                    regM, regN, threadResults, As + (BM * BK), Bs + (BK * BN),
                    warpRow, warpCol, threadRowInWarp, threadColInWarp);
            }

            __syncthreads();

            if (blockIndex + 2 * BK < K) {
                // load next B0
                loadFromGlobalMem<BM, BN, BK, strideA, strideB>(
                    N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA,
                    innerColA, innerRowB, innerColB);
            }

        } else {
            if (blockIndex + BK < K) {
                // load next B(B0)
                loadFromGlobalMem<BM, BN, BK, strideA, strideB>(
                    N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN),
                    innerRowA, innerColA, innerRowB, innerColB);
            }

            __syncthreads();

            // procss current B(B0)
            processFromSharedMem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM,
                                 WSUBN, TM, TN>(
                regM, regN, threadResults, As, Bs, warpRow, warpCol,
                threadRowInWarp, threadColInWarp);

            __syncthreads();

            // process next B(B1)
            if (blockIndex + BK < K) {
                processFromSharedMem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM,
                                     WSUBN, TM, TN>(
                    regM, regN, threadResults, As + (BM * BK), Bs + (BK * BN),
                    warpRow, warpCol, threadRowInWarp, threadColInWarp);
            }
        }

        A += 2 * BK;
        B += 2 * BK * N;
    }

    writeToGlobalMem<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        N, threadRowInWarp, threadColInWarp, alpha, beta, C, threadResults);
}
