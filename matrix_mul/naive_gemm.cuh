
template <int BLOCK_SIZE>
__global__ void naiveGemmKernel(int M, int N, int K, float alpha, float beta,
                                const float *A, const float *B, float *C) {
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

template <int BLOCK_SIZE>
__global__ void coalescedGemmKernel(int M, int N, int K, float alpha,
                                    float beta, const float *A, const float *B,
                                    float *C) {
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

template <int BLOCK_SIZE>
__global__ void sharedMemoryGemmKernel(int M, int N, int K, float alpha,
                                       float beta, const float *A,
                                       const float *B, float *C) {
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
    const int tx = threadIdx.x / BLOCK_SIZE;
    const int ty = threadIdx.x % BLOCK_SIZE;

    // Global index
    const int x = bx * BLOCK_SIZE + tx;
    const int y = by * BLOCK_SIZE + ty;

    // Shared memory
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    A += bx * BLOCK_SIZE * K;                   // row=bx, col=0
    B += by * BLOCK_SIZE;                       // row=0, col=by
    C += bx * BLOCK_SIZE * N + by * BLOCK_SIZE; // row=bx, col=by

    float tmp = 0.0;

    for (int blockIndex = 0; blockIndex < K; blockIndex += BLOCK_SIZE) {
        if (x < M && (blockIndex + ty) < K) {
            As[tx * BLOCK_SIZE + ty] = A[tx * K + ty];
        } else {
            As[tx * BLOCK_SIZE + ty] = 0.0f;
        }

        if (y < N && (blockIndex + tx) < K) {
            Bs[tx * BLOCK_SIZE + ty] = B[tx * N + ty];
        } else {
            Bs[tx * BLOCK_SIZE + ty] = 0.0f;
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // Do matrix matmul for sub matrices
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp += As[tx * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + ty];
        }

        __syncthreads();
    }
    if (x < M and y < N) {
        C[tx * N + ty] = alpha * tmp + beta * C[tx * N + ty];
    }
}

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
