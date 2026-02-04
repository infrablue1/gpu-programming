
#pragma once

#include "utils.h"

template <int BM, int BN, int BK, int strideA, int strideB>
__device__ void loadFromGlobalMem(int N, int K, const float *A, const float *B,
                                  float *As, float *Bs, int innerRowA,
                                  int innerColA, int innerRowB, int innerColB) {
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
    }

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(
                &B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WMITER, int WNITER,
          int WSUBM, int WSUBN, int TM, int TN>
__device__ void
processFromSharedMem(float *regM, float *regN, float *threadResults,
                     const float *As, const float *Bs, const int warpRow,
                     const int warpCol, const int threadRowInWarp,
                     const int threadColInWarp) {
    for (int dotIndex = 0; dotIndex < BK; dotIndex++) {
        for (int wSubRowIndex = 0; wSubRowIndex < WMITER; wSubRowIndex++) {
            for (int i = 0; i < TM; i++) {
                regM[wSubRowIndex * TM + i] =
                    As[(dotIndex * BM) + warpRow * WM + wSubRowIndex * WSUBM +
                       threadRowInWarp * TM + i];
            }
        }

        for (int wSubColIndex = 0; wSubColIndex < WNITER; wSubColIndex++) {
            for (int i = 0; i < TN; i++) {
                regN[wSubColIndex * TN + i] =
                    Bs[(dotIndex * BN) + warpCol * WN + wSubColIndex * WSUBN +
                       threadColInWarp * TN + i];
            }
        }

        for (int wSubRowIndex = 0; wSubRowIndex < WMITER; wSubRowIndex++) {
            for (int wSubColIndex = 0; wSubColIndex < WNITER; wSubColIndex++) {
                for (int row = 0; row < TM; row++) {
                    for (int col = 0; col < TN; col++) {
                        int resultIndex =
                            (wSubRowIndex * TM + row) * (WNITER * TN) +
                            (wSubColIndex * TN + col);
                        threadResults[resultIndex] +=
                            regM[wSubRowIndex * TM + row] *
                            regN[wSubColIndex * TN + col];
                    }
                }
            }
        }
    }
}

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

    // Wrap subtile size
    // WMITER * WNITER * TM * TN = (WM * WN) / kWarpSize
    // Each subWarp process WM * WN output elements and each warp have kWarpSize
    // threads. Thus each thread will process (WMITER * WNITER * TM * TN) output
    // elements.
    constexpr int WMITER = (WM * WN) / (kWarpSize * TM * TN * WNITER);

    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // ThreadIndex in warp
    const int threadIndexInWarp = threadIdx.x % kWarpSize; // [0, 31)
    const int threadRowInWrap = threadIndexInWarp / (WSUBN / TN);
    const int threadColInWrap = threadIndexInWarp % (WSUBN / TN);

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
                                     warpCol, threadRowInWrap, threadColInWrap);
        __syncthreads();
    }

    for (int wSubRowIndex = 0; wSubRowIndex < WMITER; wSubRowIndex++) {
        for (int wSubColIndex = 0; wSubColIndex < WNITER; wSubColIndex++) {
            // Move C to current warp subtile
            float *subC = C + (wSubRowIndex * WSUBM) * N + wSubColIndex * WSUBN;
            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col += 4) {
                    int CIndex = (threadRowInWrap * TM + row) * N +
                                 threadColInWrap * TN + col;

                    float4 tmp = reinterpret_cast<float4 *>(&subC[CIndex])[0];
                    const int resultIndex =
                        (wSubRowIndex * TM + row) * (WNITER * TN) +
                        wSubColIndex * TN + col;

                    tmp.x = alpha * threadResults[resultIndex] + beta * tmp.x;
                    tmp.y =
                        alpha * threadResults[resultIndex + 1] + beta * tmp.y;
                    tmp.z =
                        alpha * threadResults[resultIndex + 2] + beta * tmp.z;
                    tmp.w =
                        alpha * threadResults[resultIndex + 3] + beta * tmp.w;

                    reinterpret_cast<float4 *>(&subC[CIndex])[0] = tmp;
                }
            }
        }
    }
}
