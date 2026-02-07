
#include "utils.h"
#include <mma.h>

template <typename InputType>
__global__ void naiveTensorCore(int M, int N, int K, float alpha, float beta,
                                const float *A, const float *B, float *C) {
    // Each wrap computes one 16x16 WMMA tile of the output
    const size_t warpRow = blockIdx.y * kWMMA_M;
    const size_t warpCol = blockIdx.x * kWMMA_N;

    // Accumulator fragment for output (FP32)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                           float>
        fragC;
    nvcuda::wmma::fill_fragment(fragC, 0.0f);

    // K is divided by K / WMMA_K part and each unit has WMMA_K length.
    const size_t dotLength = (K + kWMMA_K - 1) / kWMMA_K;

    for (size_t dotIndex = 0; dotIndex < dotLength; dotIndex++) {
        // Use tf32 for float32 input type
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWMMA_M, kWMMA_N,
                               kWMMA_K, nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::row_major>
            fragA;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWMMA_M, kWMMA_N,
                               kWMMA_K, nvcuda::wmma::precision::tf32,
                               nvcuda::wmma::row_major>
            fragB;

        const InputType *subA = A + warpRow * K + dotIndex * kWMMA_K;
        nvcuda::wmma::load_matrix_sync(fragA, subA, K);

// The following part of code is not necessary as hardware do the cast work
// implicitly/
#if 0
        for(int i = 0; i < fragA.num_elements;i++){
            fragA.x[i] = nvcuda::wmma::__float_to_tf32(fragA.x[i]);
        }
#endif

        const InputType *subB = B + dotIndex * kWMMA_K * N + warpCol;
        nvcuda::wmma::load_matrix_sync(fragB, subB, N);

#if 0
        for(int i = 0; i < fragB.num_elements;i++){
            fragB.x[i] = nvcuda::wmma::__float_to_tf32(fragB.x[i]);
        }
#endif

        // Do matrix matmul
        nvcuda::wmma::mma_sync(fragC, fragA, fragB, fragC);
    }

    // Load current C tile for bet scaling
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWMMA_M, kWMMA_N, kWMMA_K,
                           float>
        oldFragC;
    float *subC = C + warpRow * N + warpCol;
    nvcuda::wmma::load_matrix_sync(oldFragC, subC, N,
                                   nvcuda::wmma::mem_row_major);

#pragma unroll
    for (int i = 0; i < fragC.num_elements; i++) {
        fragC.x[i] = alpha * fragC.x[i] + beta * oldFragC.x[i];
    }

    // Store result
    nvcuda::wmma::store_matrix_sync(subC, fragC, N,
                                    nvcuda::wmma::mem_row_major);
}
