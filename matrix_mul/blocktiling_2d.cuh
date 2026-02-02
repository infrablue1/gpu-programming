template <int BM, int BN, int BK, int TM, int TN>
__global__ void blockTiling2DGemmKernel(int M, int N, int K, float alpha,
                                        float beta, const float *A,
                                        const float *B, float *C) {
    // Block index
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;

    // Shared memory
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread index
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);
    const int numThreads = (BM / TM) * (BN / TN);

    A += blockRow * BM * K;                 // row=blockRow, col=0
    B += blockCol * BN;                     // row=0, col=blockCol
    C += blockRow * BM * N + blockCol * BN; // row=blockRow, col=blockCol

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA = numThreads / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {
        // Each thread will move BM / strideA = BM * BK / numthreads float
        // elements for matrix A
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }

        // Each thread will move BN / strideB = BM * BN /  numthreads float
        // elements for matrix B
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // Do matrix matmul for sub matrices
        for (int dotIndex = 0; dotIndex < BK; dotIndex++) {
            for (int i = 0; i < TM; i++) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIndex];
            }
            for (int i = 0; i < TN; i++) {
                regN[i] = Bs[dotIndex * BN + threadCol * TN + i];
            }
            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col++) {
                    threadResults[row * TN + col] += regM[row] * regN[col];
                }
            }
        }

        __syncthreads();
    }

    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            int globalRow = blockRow * BM + threadRow * TM + row;
            int globalCol = blockCol * BN + threadCol * TN + col;
            if (globalRow < M && globalCol < N) {
                int CIndex =
                    (threadRow * TM + row) * N + (threadCol * TN + col);
                C[CIndex] =
                    alpha * threadResults[row * TN + col] + beta * C[CIndex];
            }
        }
    }
}