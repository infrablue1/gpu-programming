
#pragma once

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

template <int WMITER, int WNITER, int WSUBM, int WSUBN, int TM, int TN>
__device__ void writeToGlobalMem(const int N, const int threadRowInWarp,
                                 const int threadColInWarp, float alpha,
                                 float beta, float *C,
                                 const float *threadResults) {
    for (int wSubRowIndex = 0; wSubRowIndex < WMITER; wSubRowIndex++) {
        for (int wSubColIndex = 0; wSubColIndex < WNITER; wSubColIndex++) {
            // Move C to current warp subtile
            float *subC = C + (wSubRowIndex * WSUBM) * N + wSubColIndex * WSUBN;
            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col += 4) {
                    int CIndex = (threadRowInWarp * TM + row) * N +
                                 threadColInWarp * TN + col;

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
