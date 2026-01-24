
template <int BLOCK_SIZE>
__global__ void naiveGemmKernel(int M, int N, int K, float alpha, float beta,
                                const float *A, const float *B, float *C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= M && y <= N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
