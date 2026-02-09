
#pragma once

#include "utils.h"
#include "warp_utils.cuh"

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ void warpTilingGemmKernel(int M, int N, int K, float alpha,
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
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += blockRow * BM * K; // row=blockRow, col=0
    B += blockCol * BN;     // row=0, col=blockCol
    C += (blockRow * BM + warpRow * WN) * N + blockCol * BN + warpCol * WN;

    // Load 128 bits per operation
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    const int strideA = NUM_THREADS / (BK / 4);
    const int strideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {
        loadFromGlobalMem<BM, BN, BK, strideA, strideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        A += BK;
        B += BK * N;
        processFromSharedMem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                             TM, TN>(regM, regN, threadResults, As, Bs, warpRow,
                                     warpCol, threadRowInWarp, threadColInWarp);
        __syncthreads();
    }

    writeToGlobalMem<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        N, threadRowInWarp, threadColInWarp, alpha, beta, C, threadResults);
}
