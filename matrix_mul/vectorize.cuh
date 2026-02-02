template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorizeGemmKernel(int M, int N, int K, float alpha,
                                    float beta, const float *A, const float *B,
                                    float *C) {
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

    // Load 128 bits per operation
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    const int strideA = numThreads / (BK / 4);
    const int strideB = numThreads / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int blockIndex = 0; blockIndex < K; blockIndex += BK) {
        // Each thread will move BM / strideA = BM * BK / (4 * numthreads)
        // float4 elements for matrix A Transpose A while loading it
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            const float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + loadOffset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
        }

        // Each thread will move BN / strideB = BM * BN / (4 * numthreads)
        // float4 elements for matrix B
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            reinterpret_cast<float4 *>(
                &Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4 *>(
                    &B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // Do matrix matmul for sub matrices
        for (int dotIndex = 0; dotIndex < BK; dotIndex++) {
            for (int i = 0; i < TM; i++) {
                regM[i] = As[dotIndex * BM + threadRow * TM + i];
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
        for (int col = 0; col < TN; col += 4) {
            int globalRow = blockRow * BM + threadRow * TM + row;
            int globalCol = blockCol * BN + threadCol * TN + col;
            if (globalRow < M && globalCol < N) {
                int CIndex =
                    (threadRow * TM + row) * N + (threadCol * TN + col);

                float4 tmp = reinterpret_cast<float4 *>(&C[CIndex])[0];
                tmp.x = alpha * threadResults[row * TN + col] + beta * tmp.x;
                tmp.y =
                    alpha * threadResults[row * TN + col + 1] + beta * tmp.y;
                tmp.z =
                    alpha * threadResults[row * TN + col + 2] + beta * tmp.z;
                tmp.w =
                    alpha * threadResults[row * TN + col + 3] + beta * tmp.w;

                reinterpret_cast<float4 *>(&C[CIndex])[0] = tmp;
            }
        }
    }
}
