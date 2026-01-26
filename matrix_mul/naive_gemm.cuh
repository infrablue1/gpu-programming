
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
    const int ty = threadIdx.y % BLOCK_SIZE;

    // Shared memory
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    A += bx * BLOCK_SIZE * K;                   // row=bx, col=0
    B += by * BLOCK_SIZE;                       // row=0, col=by
    C += bx * BLOCK_SIZE * N + by * BLOCK_SIZE; // row=bx, col=by

    float tmp = 0.0;

    for (int blockIndex = 0; blockIndex < K; blockIndex += BLOCK_SIZE) {
        As[tx * BLOCK_SIZE + ty] = A[tx * K + ty];
        Bs[tx * BLOCK_SIZE + ty] = B[tx * N + ty];

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // Do matrix matmul for sub matrices
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp += As[tx * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + ty];
        }

        __syncthreads();
    }
    printf("tx=%d, ty=%d, tmp=%f\n", tx, ty, tmp);
    C[tx * N + ty] = alpha * tmp + beta * C[tx * N + ty];
}