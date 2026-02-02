template <int BM, int BN, int BK, int TM>
__global__ void blockTiling1DGemmKernel(int M, int N, int K, float alpha,
                                        float beta, const float *A,
                                        const float *B, float *C) {
    // Block index
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    // Shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread index
    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN;

    // Global index
    const int globalRow = blockRow * BM + threadRow * TM;
    const int globalCol = blockCol * BN + threadCol;

    A += blockRow * BM * K;                 // row=blockRow, col=0
    B += blockCol * BN;                     // row=0, col=blockCol
    C += blockRow * BM * N + blockCol * BN; // row=blockRow, col=blockCol

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0f};

    for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {

        if (blockRow * BM + innerRowA < M && blockIndex + innerColA < K) {
            As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        } else {
            As[innerRowA * BK + innerColA] = 0.0f;
        }

        if (blockIndex + innerRowB < K && blockCol * BN + innerColB < N) {
            Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB * BN + innerColB] = 0.0f;
        }

        __syncthreads();

        A += BK;
        B += BK * N;

// The naive code for calculation is as follows, we reuse Bs entry by move
// dotproduct to outer loop.
#if 0
        for (int resIndex = 0; resIndex < TM; resIndex++) {
            for(int dotIndex = 0; dotIndex < BK; dotIndex++) {
                int aidx = (threadRow * TM + resIndex)* BK + dotIndex;
                int bidx = dotIndex * BN + threadCol;
                if (As[aidx] != 0.0 && Bs[bidx] != 0.0) {
                    printf("threadRow:%d, threadCol:%d, As[%d]=%f, Bs[%d]=%f\n",threadRow, threadCol, aidx, As[aidx], bidx, Bs[bidx]);
                }
                threadResults[resIndex] += As[(threadRow * TM + resIndex)* BK + dotIndex] * Bs[dotIndex * BN + threadCol];
            }
        }
#endif

        // Do matrix matmul for sub matrices
        for (int dotIndex = 0; dotIndex < BK; dotIndex++) {
            float BTmp = Bs[dotIndex * BN + threadCol];
            for (int resIndex = 0; resIndex < TM; resIndex++) {
                threadResults[resIndex] +=
                    As[(threadRow * TM + resIndex) * BK + dotIndex] * BTmp;
            }
        }

        __syncthreads();
    }

    for (int resIndex = 0; resIndex < TM; resIndex++) {
        if (globalRow + resIndex < M && globalCol < N) {
            int CIndex = (threadRow * TM + resIndex) * N + threadCol;
            C[CIndex] = alpha * threadResults[resIndex] + beta * C[CIndex];
        }
    }
}